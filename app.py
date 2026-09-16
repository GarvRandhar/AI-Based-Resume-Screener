"""FastAPI backend for the AI resume screening platform."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db
import job_manager
import ollama_client
import parsing
from config import ensure_config_file, get_settings

_settings = get_settings()
logging.basicConfig(
    level=getattr(logging, str(_settings.get("log_level", "INFO")).upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_config_file()
    db.init_db()
    db.ensure_dirs()
    settings = get_settings()
    job_manager.set_default_concurrency(
        int(settings.get("default_concurrency") or 4),
        embed_n=int(settings.get("embed_concurrency") or 8),
    )
    ollama_client.reload_settings()
    health = await ollama_client.check_ollama()
    if not health.get("ok"):
        logger.warning(
            "Ollama not reachable at %s — start it before processing jobs. (%s)",
            ollama_client.OLLAMA_BASE,
            health.get("error"),
        )
    else:
        if not health.get("has_extract_model"):
            logger.warning(
                "Model %s not found. Run: ollama pull %s",
                ollama_client.EXTRACT_MODEL,
                ollama_client.EXTRACT_MODEL,
            )
        if not health.get("has_score_model"):
            logger.warning(
                "Score model %s not found (fit falls back to extract). Optional: ollama pull %s",
                ollama_client.SCORE_MODEL,
                ollama_client.SCORE_MODEL,
            )
        if not health.get("has_embed_model"):
            logger.warning(
                "Model %s not found. Run: ollama pull %s",
                ollama_client.EMBED_MODEL,
                ollama_client.EMBED_MODEL,
            )
        logger.info(
            "Speed config: extract=%s score=%s concurrency=%s embed_concurrency=%s "
            "hybrid_first=%s refine=%s cross_job_dedupe=%s",
            ollama_client.EXTRACT_MODEL,
            ollama_client.SCORE_MODEL,
            settings.get("default_concurrency"),
            settings.get("embed_concurrency"),
            settings.get("hybrid_first_bulk"),
            settings.get("extract_refine_enabled"),
            settings.get("cross_job_dedupe"),
        )
    yield
    await ollama_client.close_http_client()
    try:
        import parsing as parsing_mod

        parsing_mod.shutdown_parse_executor()
    except Exception:
        pass


app = FastAPI(
    title="Resume Screener POC",
    description="Local AI-powered bulk resume screening (Ollama + SQLite)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HardFilterRule(BaseModel):
    field: str
    operator: str = "gte"
    value: Any
    label: Optional[str] = None


class BulkModeConfig(BaseModel):
    """Fast bulk: hybrid pre-rank, then LLM fit only top candidates."""
    enabled: bool = True
    llm_top_k: int = Field(default=20, ge=1, le=500)
    min_pre_score: float = Field(default=30.0, ge=0.0, le=100.0)
    always_full_below: int = Field(
        default=0,
        ge=0,
        le=1000,
        description="If total resumes < this, LLM-score everyone (0 = always top-K path)",
    )
    hybrid_first: bool = Field(
        default=True,
        description="Phase-1 hybrid-only extract (no Ollama chat until phase 2)",
    )
    phase2_llm_extract: bool = Field(
        default=False,
        description="Extra LLM structured extract in phase 2 (slow; fit works without it)",
    )


class ScoreWeights(BaseModel):
    """Relative weights for final score blend (normalized server-side)."""
    ats: float = Field(default=0.20, ge=0.0, le=1.0)
    embed: float = Field(default=0.30, ge=0.0, le=1.0)
    llm: float = Field(default=0.50, ge=0.0, le=1.0)


class CriteriaPayload(BaseModel):
    """
    Prompt-first screening: HR provides one hiring brief; ranking uses that text only.
    shortlist_size: how many top-ranked candidates to mark as shortlisted.
    min_final_score: optional quality floor for shortlist entry.
    hard_filters are optional advanced gates (default empty = rank everyone).
    """
    hiring_prompt: str = ""
    # Alias kept for older clients / saved templates
    job_description: str = ""
    shortlist_size: int = Field(
        default=5,
        ge=1,
        le=500,
        description="Number of top candidates to shortlist for this drive",
    )
    min_final_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum final score (0-100) to enter shortlist",
    )
    exclude_duplicates_from_shortlist: bool = True
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)
    hard_filters: list[HardFilterRule] = Field(default_factory=list)
    bulk_mode: BulkModeConfig = Field(default_factory=BulkModeConfig)

    def normalized(self) -> dict[str, Any]:
        prompt = (self.hiring_prompt or self.job_description or "").strip()
        return {
            "hiring_prompt": prompt,
            "job_description": prompt,
            "shortlist_size": max(1, min(500, int(self.shortlist_size or 5))),
            "min_final_score": max(0.0, min(100.0, float(self.min_final_score or 0))),
            "exclude_duplicates_from_shortlist": bool(
                self.exclude_duplicates_from_shortlist
            ),
            "score_weights": self.score_weights.model_dump(),
            "hard_filters": [h.model_dump() for h in self.hard_filters],
            "bulk_mode": self.bulk_mode.model_dump(),
        }


class CreateJobRequest(BaseModel):
    name: str
    criteria: CriteriaPayload
    concurrency: int = Field(default=4, ge=1, le=12)


class StartJobRequest(BaseModel):
    concurrency: int = Field(default=4, ge=1, le=12)


class ResumeOverrideRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    total_years_experience: Optional[float] = None
    notice_period_days: Optional[int] = None
    skills: Optional[Any] = None  # list or comma-separated string
    certifications: Optional[Any] = None
    summary: Optional[str] = None
    rescore: bool = True
    run_llm: bool = True


class RescoreRequest(BaseModel):
    run_llm: bool = True


class ShortlistSettingsRequest(BaseModel):
    shortlist_size: Optional[int] = Field(default=None, ge=1, le=500)
    min_final_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    exclude_duplicates_from_shortlist: Optional[bool] = None


class HrStatusRequest(BaseModel):
    hr_status: str = Field(default="none", pattern="^(none|starred|rejected)$")
    hr_notes: Optional[str] = None


class TemplateRequest(BaseModel):
    name: str
    criteria: CriteriaPayload


class UpdateCriteriaRequest(BaseModel):
    criteria: CriteriaPayload


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, Any]:
    """Friendly landing — browsers hitting :8000 are not a bug; use the React UI."""
    return {
        "service": "Resume Screener API",
        "status": "ok",
        "docs": "http://127.0.0.1:8000/docs",
        "health": "http://127.0.0.1:8000/health",
        "ui": "Run the recruiter UI with: cd frontend && npm run dev → http://localhost:5173",
    }


class BriefAnalyzeRequest(BaseModel):
    hiring_prompt: str = ""
    text: str = ""


@app.post("/brief/analyze")
async def analyze_brief(body: BriefAnalyzeRequest) -> dict[str, Any]:
    """Hiring brief quality checklist (same logic as the former Streamlit UI)."""
    from scoring import analyze_hiring_brief

    text = (body.hiring_prompt or body.text or "").strip()
    return analyze_hiring_brief(text)


@app.get("/health")
async def health() -> dict[str, Any]:
    ollama = await ollama_client.check_ollama()
    settings = get_settings()
    return {
        "status": "ok",
        "ollama": ollama,
        "db": str(db.DB_PATH),
        "config": {
            "api_host": settings.get("api_host"),
            "api_port": settings.get("api_port"),
            "extract_model": ollama_client.EXTRACT_MODEL,
            "score_model": ollama_client.SCORE_MODEL,
            "embed_model": ollama_client.EMBED_MODEL,
            "default_concurrency": settings.get("default_concurrency"),
            "embed_concurrency": settings.get("embed_concurrency"),
            "embed_cache_enabled": settings.get("embed_cache_enabled"),
            "extract_refine_enabled": settings.get("extract_refine_enabled"),
            "hybrid_first_bulk": settings.get("hybrid_first_bulk"),
            "cross_job_dedupe": settings.get("cross_job_dedupe"),
            "config_path": str(ensure_config_file()),
        },
        "ready": bool(
            ollama.get("ok")
            and ollama.get("has_extract_model")
            and ollama.get("has_embed_model")
            # score model optional: falls back to extract_model
        ),
    }


@app.get("/config")
async def get_config() -> dict[str, Any]:
    return {"config": get_settings(), "path": str(ensure_config_file())}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@app.post("/jobs")
async def create_job(body: CreateJobRequest) -> dict[str, Any]:
    criteria = body.criteria.normalized()
    if not criteria.get("hiring_prompt"):
        raise HTTPException(
            400,
            "hiring_prompt is required — paste the full hiring brief / instructions.",
        )
    job_id = db.create_job(body.name.strip() or "Untitled drive", criteria)
    job_manager.set_default_concurrency(body.concurrency)
    job = db.get_job(job_id)
    return {"job": job}


@app.get("/jobs")
async def list_jobs(limit: int = 50) -> dict[str, Any]:
    return {"jobs": db.list_jobs(limit=limit)}


@app.get("/jobs/{job_id}")
async def get_job(job_id: int) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job["is_running"] = job_manager.is_job_running(job_id)
    return {"job": job}


@app.patch("/jobs/{job_id}/criteria")
async def update_job_criteria(
    job_id: int, body: UpdateCriteriaRequest
) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "processing":
        raise HTTPException(400, "Cannot change criteria while job is processing")
    criteria = body.criteria.normalized()
    if not criteria.get("hiring_prompt"):
        raise HTTPException(400, "hiring_prompt is required")
    db.update_job(job_id, criteria=criteria)
    return {"job": db.get_job(job_id)}


@app.post("/jobs/{job_id}/upload")
async def upload_resumes(
    job_id: int,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "processing":
        raise HTTPException(400, "Cannot upload while job is processing")

    job_dir = db.UPLOADS_DIR / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for uf in files:
        original = Path(uf.filename or f"upload_{uuid.uuid4().hex}").name
        # Sanitize
        safe_name = "".join(
            c if c.isalnum() or c in "._- ()[]" else "_" for c in original
        )
        dest = job_dir / safe_name
        if dest.exists():
            dest = job_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        try:
            with open(dest, "wb") as out:
                shutil.copyfileobj(uf.file, out)
            saved.append(dest)
        except Exception as exc:
            logger.error("Failed to save %s: %s", original, exc)
        finally:
            await uf.close()

    # Expand zips and register resume rows
    resume_paths = parsing.collect_upload_files(saved, job_dir)
    created_ids: list[int] = []
    for rp in resume_paths:
        rid = db.create_resume(job_id, rp.name, str(rp))
        created_ids.append(rid)

    # Update total_count to current resume count
    all_resumes = db.list_resumes_for_job(job_id)
    db.update_job(job_id, total_count=len(all_resumes), status="pending")

    return {
        "uploaded_files": len(saved),
        "resumes_registered": len(created_ids),
        "resume_ids": created_ids,
        "total_for_job": len(all_resumes),
    }


@app.post("/jobs/{job_id}/start")
async def start_job(
    job_id: int,
    body: StartJobRequest = StartJobRequest(),
) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    resumes = db.list_resumes_for_job(job_id)
    if not resumes:
        raise HTTPException(400, "No resumes uploaded for this job")
    if job_manager.is_job_running(job_id):
        return {"started": False, "message": "Job already running", "job": job}

    ollama = await ollama_client.check_ollama()
    if not ollama.get("ok"):
        raise HTTPException(
            503,
            f"Ollama is not reachable at {ollama_client.OLLAMA_BASE}. "
            "Start Ollama and pull required models.",
        )

    job_manager.set_default_concurrency(body.concurrency)
    # Fire async task on the running event loop (single-process asyncio queue)
    started = await job_manager.start_job(job_id, concurrency=body.concurrency)
    job = db.get_job(job_id)
    return {
        "started": started,
        "message": "Processing started" if started else "Already running",
        "job": job,
    }


@app.get("/jobs/{job_id}/results")
async def job_results(job_id: int) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    results = db.get_results_for_job(job_id)
    shortlist_size = db.get_shortlist_size_for_job(job_id)
    # Drop raw_text from list payload for bandwidth (detail endpoint has full data)
    slim = []
    name_queue: list[dict[str, Any]] = []
    for r in results:
        item = {k: v for k, v in r.items() if k != "raw_text"}
        # Always include shortlist fields explicitly (avoid client confusion)
        item["shortlist_size"] = shortlist_size
        item["shortlisted"] = bool(item.get("shortlisted"))
        slim.append(item)
        if item.get("name_needs_review") and int(item.get("hard_filter_pass") or 0) == 1:
            name_queue.append(
                {
                    "resume_id": item.get("resume_id"),
                    "rank": item.get("rank"),
                    "candidate_name": item.get("candidate_name"),
                    "filename": item.get("source_filename") or item.get("filename"),
                    "name_confidence": item.get("name_confidence"),
                    "name_source": item.get("name_source"),
                    "name_warning": item.get("name_warning"),
                    "shortlisted": item.get("shortlisted"),
                    "final_score": item.get("final_score"),
                }
            )
    # Shortlist first, then by rank
    name_queue.sort(
        key=lambda x: (
            0 if x.get("shortlisted") else 1,
            x.get("rank") if x.get("rank") is not None else 9999,
        )
    )
    return {
        "job": {
            **job,
            "is_running": job_manager.is_job_running(job_id),
            "shortlist_size": shortlist_size,
        },
        "results": slim,
        "shortlist_size": shortlist_size,
        "shortlisted_count": sum(1 for r in slim if r.get("shortlisted")),
        "name_review_queue": name_queue,
        "name_review_count": len(name_queue),
    }


@app.get("/jobs/{job_id}/progress")
async def job_progress(job_id: int) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    total = job.get("total_count") or 0
    done = job.get("processed_count") or 0
    is_running = job_manager.is_job_running(job_id)
    status = job["status"]
    snap = job_manager.job_progress_snapshot(job_id)

    if status == "completed":
        pct = 100.0
        elapsed = None
        eta = None
        phase_done = None
        phase_total = None
        phase_percent = 100.0
    elif snap and (is_running or status in ("processing", "paused")):
        pct = min(99.0, snap["percent"]) if (is_running or status == "processing") else snap["percent"]
        elapsed = snap["elapsed_seconds"]
        eta = snap["eta_seconds"]
        phase_done = snap["phase_done"]
        phase_total = snap["phase_total"]
        phase_percent = snap["phase_percent"]
    else:
        if is_running or status in ("processing", "pending"):
            raw = (50.0 * done / total) if total else 0.0
            pct = min(95.0, raw)
        else:
            pct = (100.0 * done / total) if total else 0.0
        elapsed = None
        eta = None
        phase_done = None
        phase_total = None
        phase_percent = pct

    return {
        "job_id": job_id,
        "status": status,
        "processed_count": done,
        "total_count": total,
        "percent": round(pct, 1),
        "is_running": is_running,
        "error_message": job.get("error_message"),
        "current_stage": job.get("current_stage"),
        "current_filename": job.get("current_filename"),
        "control": job.get("control") or "run",
        "failed_count": job.get("failed_count") or 0,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "phase_done": phase_done,
        "phase_total": phase_total,
        "phase_percent": round(float(phase_percent or 0.0), 1),
    }


@app.patch("/jobs/{job_id}/shortlist-settings")
async def patch_shortlist_settings(
    job_id: int, body: ShortlistSettingsRequest
) -> dict[str, Any]:
    """Recompute shortlist flags without re-parsing (instant)."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    criteria = dict(job.get("criteria") or {})
    if body.shortlist_size is not None:
        criteria["shortlist_size"] = int(body.shortlist_size)
    if body.min_final_score is not None:
        criteria["min_final_score"] = float(body.min_final_score)
    if body.exclude_duplicates_from_shortlist is not None:
        criteria["exclude_duplicates_from_shortlist"] = bool(
            body.exclude_duplicates_from_shortlist
        )
    db.update_job(job_id, criteria=criteria)
    # Ranks stay; shortlisted is derived at read time from criteria
    return {
        "job": db.get_job(job_id),
        "shortlist_size": db.get_shortlist_size_for_job(job_id),
        "min_final_score": db.get_min_final_score_for_job(job_id),
    }


@app.post("/jobs/{job_id}/enrich-shortlist")
async def enrich_shortlist(job_id: int) -> dict[str, Any]:
    """
    LLM-extract skills + work history for current shortlist.
    Use after ranking or when shortlist size grows (profiles still hybrid-thin).
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job_manager.is_job_running(job_id):
        raise HTTPException(400, "Job is still processing — wait until ranking finishes")
    ollama = await ollama_client.check_ollama()
    if not ollama.get("ok"):
        raise HTTPException(503, "Ollama is not reachable")
    prev_status = job.get("status") or "completed"
    try:
        stats = await job_manager.enrich_shortlist_for_job(job_id)
    finally:
        db.update_job(
            job_id,
            status=prev_status if prev_status != "processing" else "completed",
            current_stage="done",
            current_filename="",
        )
    return {"ok": True, **stats, "job": db.get_job(job_id)}


@app.post("/jobs/{job_id}/pause")
async def pause_job(job_id: int) -> dict[str, Any]:
    if not db.get_job(job_id):
        raise HTTPException(404, "Job not found")
    job_manager.request_pause(job_id)
    return {"ok": True, "control": "pause", "job": db.get_job(job_id)}


@app.post("/jobs/{job_id}/resume")
async def resume_job_control(job_id: int) -> dict[str, Any]:
    if not db.get_job(job_id):
        raise HTTPException(404, "Job not found")
    job_manager.request_resume(job_id)
    # If not running, restart full remaining? resume only unblocks paused task
    if not job_manager.is_job_running(job_id):
        # Allow restarting a paused-but-dead process
        await job_manager.start_job(job_id)
    return {"ok": True, "control": "run", "job": db.get_job(job_id)}


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int) -> dict[str, Any]:
    if not db.get_job(job_id):
        raise HTTPException(404, "Job not found")
    job_manager.request_cancel(job_id)
    return {"ok": True, "control": "cancel", "job": db.get_job(job_id)}


@app.post("/jobs/{job_id}/rerun-failed")
async def rerun_failed_job(
    job_id: int, body: StartJobRequest = StartJobRequest()
) -> dict[str, Any]:
    if not db.get_job(job_id):
        raise HTTPException(404, "Job not found")
    if job_manager.is_job_running(job_id):
        raise HTTPException(400, "Job is still running")
    ollama = await ollama_client.check_ollama()
    if not ollama.get("ok"):
        raise HTTPException(503, "Ollama is not reachable")
    job_manager.set_default_concurrency(body.concurrency)
    result = await job_manager.rerun_failed(job_id, concurrency=body.concurrency)
    return {**result, "job": db.get_job(job_id)}


@app.get("/jobs/{job_id}/errors")
async def job_errors(job_id: int) -> dict[str, Any]:
    if not db.get_job(job_id):
        raise HTTPException(404, "Job not found")
    failed = db.list_failed_resumes(job_id)
    return {
        "job_id": job_id,
        "count": len(failed),
        "errors": [
            {
                "resume_id": r["id"],
                "filename": r.get("filename"),
                "status": r.get("status"),
                "stage": r.get("stage"),
                "error_message": r.get("error_message"),
            }
            for r in failed
        ],
    }


def _export_slim_row(r: dict[str, Any], *, rich: bool = True) -> dict[str, Any]:
    """Rich export row for hiring packs."""
    structured = r.get("structured") if isinstance(r.get("structured"), dict) else {}
    skills = structured.get("skills") or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    work = structured.get("work_history") or []
    work_lines: list[str] = []
    if isinstance(work, list):
        for w in work[:8]:
            if isinstance(w, dict):
                parts = [
                    str(w.get("title") or "").strip(),
                    str(w.get("company") or "").strip(),
                    str(w.get("duration") or "").strip(),
                ]
                line = " @ ".join(p for p in parts[:2] if p)
                if parts[2]:
                    line = f"{line} ({parts[2]})" if line else parts[2]
                if line:
                    work_lines.append(line)
            elif w:
                work_lines.append(str(w))
    row: dict[str, Any] = {
        "resume_id": r.get("resume_id"),
        "candidate_name": r.get("candidate_name"),
        "filename": r.get("source_filename") or r.get("filename"),
        "rank": r.get("rank"),
        "shortlisted": r.get("shortlisted"),
        "hr_status": r.get("hr_status"),
        "hr_notes": r.get("hr_notes"),
        "final_score": r.get("final_score"),
        "ats_score": r.get("ats_score"),
        "embedding_similarity": r.get("embedding_similarity"),
        "llm_score": r.get("llm_score"),
        "llm_justification": r.get("llm_justification"),
        "is_duplicate": r.get("is_duplicate"),
        "duplicate_reasons": r.get("duplicate_reasons"),
        "name_confidence": r.get("name_confidence"),
        "name_source": r.get("name_source"),
        "name_needs_review": r.get("name_needs_review"),
        "match_points": r.get("match_points"),
        "email": structured.get("email"),
        "phone": structured.get("phone"),
        "location": structured.get("location"),
        "years_experience": structured.get("total_years_experience"),
        "seen_elsewhere": r.get("seen_elsewhere"),
        "seen_in_other_jobs": r.get("seen_in_other_jobs") or [],
    }
    if rich:
        row["skills"] = skills[:30] if isinstance(skills, list) else []
        row["work_history"] = work_lines
        row["education"] = structured.get("education") or []
        row["summary"] = (structured.get("summary") or "")[:500] or None
        row["licenses"] = structured.get("licenses") or []
        row["certifications"] = structured.get("certifications") or []
        ext = structured.get("_extraction") or {}
        row["profile_enriched"] = bool(ext.get("shortlist_enriched"))
    return row


def _export_markdown(
    job: dict[str, Any],
    shortlist: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    """Human-readable hiring summary."""
    criteria = job.get("criteria") or {}
    brief = (
        (criteria.get("hiring_prompt") or criteria.get("job_description") or "")
        .strip()
    )
    lines = [
        f"# {job.get('name') or 'Hiring drive'} — export pack",
        "",
        f"- Job ID: {job.get('id')}",
        f"- Status: {job.get('status')}",
        f"- Generated: {_export_now()}",
        f"- Shortlist size setting: {(criteria.get('shortlist_size') or 5)}",
        f"- Candidates scored: {stats.get('scored', 0)} · shortlisted: {stats.get('shortlisted', 0)}",
        f"- Duplicates in drive: {stats.get('duplicates', 0)} · seen in other jobs: {stats.get('seen_elsewhere', 0)}",
        f"- Name verify queue: {stats.get('name_review', 0)}",
        "",
        "## Hiring brief",
        "",
        brief[:4000] if brief else "_(no brief)_",
        "",
        "## Shortlist",
        "",
    ]
    if not shortlist:
        lines.append("_No shortlisted candidates._")
    for r in shortlist:
        lines.append(
            f"### #{r.get('rank') or '—'} · {r.get('candidate_name') or 'Unknown'}"
        )
        lines.append(
            f"- Score: **{r.get('final_score')}** · ATS {r.get('ats_score')} · "
            f"Embed {r.get('embedding_similarity')} · LLM {r.get('llm_score')}"
        )
        if r.get("email") or r.get("phone"):
            lines.append(
                f"- Contact: {r.get('email') or '—'} · {r.get('phone') or '—'}"
            )
        if r.get("location") or r.get("years_experience") is not None:
            lines.append(
                f"- Location: {r.get('location') or '—'} · Years: {r.get('years_experience')}"
            )
        skills = r.get("skills") or []
        if skills:
            lines.append(f"- Skills: {', '.join(str(s) for s in skills[:20])}")
        work = r.get("work_history") or []
        if work:
            lines.append("- Work history:")
            for w in work[:6]:
                lines.append(f"  - {w}")
        if r.get("llm_justification"):
            lines.append(f"- Why: {r.get('llm_justification')}")
        if r.get("name_needs_review"):
            lines.append(
                f"- ⚠️ Verify name (source={r.get('name_source')}, conf={r.get('name_confidence')})"
            )
        if r.get("seen_elsewhere"):
            prior = r.get("seen_in_other_jobs") or []
            names = ", ".join(
                f"{p.get('job_name')} (job {p.get('job_id')})" for p in prior[:4]
            )
            lines.append(f"- Seen in other jobs: {names}")
        if r.get("hr_notes"):
            lines.append(f"- HR notes: {r.get('hr_notes')}")
        lines.append("")
    return "\n".join(lines)


def _export_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


@app.get("/jobs/{job_id}/export-pack")
async def export_pack(job_id: int) -> dict[str, Any]:
    """JSON export pack: brief + shortlist + ranked list + stats + markdown summary."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    results = db.get_results_for_job(job_id)
    shortlist = [r for r in results if r.get("shortlisted")]
    slim_all = [_export_slim_row(r) for r in results]
    slim_sl = [_export_slim_row(r) for r in shortlist]

    scored = sum(
        1
        for r in results
        if int(r.get("hard_filter_pass") or 0) == 1
    )
    stats = {
        "total": len(results),
        "scored": scored,
        "shortlisted": len(shortlist),
        "rejected": sum(1 for r in results if r.get("resume_status") == "rejected"),
        "failed": sum(1 for r in results if r.get("resume_status") == "failed"),
        "duplicates": sum(1 for r in results if r.get("is_duplicate")),
        "seen_elsewhere": sum(1 for r in results if r.get("seen_elsewhere")),
        "name_review": sum(1 for r in results if r.get("name_needs_review")),
        "starred": sum(1 for r in results if (r.get("hr_status") or "") == "starred"),
    }

    settings = get_settings()
    pack = {
        "version": 2,
        "exported_at": _export_now(),
        "models": {
            "extract_model": ollama_client.EXTRACT_MODEL,
            "score_model": ollama_client.SCORE_MODEL,
            "embed_model": ollama_client.EMBED_MODEL,
        },
        "job": {
            "id": job["id"],
            "name": job["name"],
            "status": job["status"],
            "criteria": job.get("criteria"),
            "created_at": job.get("created_at"),
            "total_count": job.get("total_count"),
            "processed_count": job.get("processed_count"),
        },
        "stats": stats,
        "shortlist": slim_sl,
        "all_ranked": slim_all,
        "starred": [
            _export_slim_row(r)
            for r in results
            if (r.get("hr_status") or "") == "starred"
        ],
        "rejected_by_hr": [
            _export_slim_row(r)
            for r in results
            if (r.get("hr_status") or "") == "rejected"
        ],
        "name_review_queue": [
            {
                "resume_id": r.get("resume_id"),
                "candidate_name": r.get("candidate_name"),
                "filename": r.get("filename"),
                "name_confidence": r.get("name_confidence"),
                "name_source": r.get("name_source"),
                "shortlisted": r.get("shortlisted"),
                "rank": r.get("rank"),
            }
            for r in results
            if r.get("name_needs_review") and int(r.get("hard_filter_pass") or 0) == 1
        ],
        "cross_job_hits": [
            {
                "resume_id": r.get("resume_id"),
                "candidate_name": r.get("candidate_name"),
                "email": (r.get("structured") or {}).get("email")
                if isinstance(r.get("structured"), dict)
                else None,
                "appearances": r.get("seen_in_other_jobs"),
            }
            for r in results
            if r.get("seen_elsewhere")
        ],
        "markdown_summary": _export_markdown(job, slim_sl, stats),
        "config_snapshot": {
            "hybrid_first_bulk": settings.get("hybrid_first_bulk"),
            "enrich_shortlist": settings.get("enrich_shortlist"),
            "cross_job_dedupe": settings.get("cross_job_dedupe"),
            "pdf_fast_mode": settings.get("pdf_fast_mode"),
        },
    }
    return pack


# ---------------------------------------------------------------------------
# Resumes
# ---------------------------------------------------------------------------


@app.get("/resumes/{resume_id}")
async def get_resume_detail(resume_id: int) -> dict[str, Any]:
    resume = db.get_resume(resume_id)
    if not resume:
        raise HTTPException(404, "Resume not found")
    # Attach score if present
    results = db.get_results_for_job(resume["job_id"])
    score_row = next(
        (r for r in results if r["resume_id"] == resume_id), None
    )
    return {"resume": resume, "score": score_row}


@app.patch("/resumes/{resume_id}/hr")
async def patch_hr_status(resume_id: int, body: HrStatusRequest) -> dict[str, Any]:
    """Star / reject / notes for HR workflow."""
    resume = db.get_resume(resume_id)
    if not resume:
        raise HTTPException(404, "Resume not found")
    notes = body.hr_notes if body.hr_notes is not None else resume.get("hr_notes")
    db.update_resume(
        resume_id,
        hr_status=body.hr_status,
        hr_notes=notes or "",
    )
    return {"resume": db.get_resume(resume_id)}


@app.patch("/resumes/{resume_id}/override")
async def override_resume_fields(
    resume_id: int, body: ResumeOverrideRequest
) -> dict[str, Any]:
    """
    Recruiter field corrections. Optionally re-run filters + scoring
    (no re-parse) so ranks update immediately.
    """
    resume = db.get_resume(resume_id)
    if not resume:
        raise HTTPException(404, "Resume not found")
    if job_manager.is_job_running(resume["job_id"]):
        raise HTTPException(
            400, "Cannot edit while the parent job is still processing"
        )

    overrides = body.model_dump(
        exclude={"rescore", "run_llm"}, exclude_none=True
    )
    if not overrides:
        raise HTTPException(400, "No override fields provided")

    try:
        structured = job_manager.apply_overrides_and_save(resume_id, overrides)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    score_result = None
    if body.rescore:
        try:
            score_result = await job_manager.rescore_resume(
                resume_id, run_llm=body.run_llm
            )
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    resume = db.get_resume(resume_id)
    results = db.get_results_for_job(resume["job_id"])
    score_row = next(
        (r for r in results if r["resume_id"] == resume_id), None
    )
    return {
        "resume": resume,
        "structured": structured,
        "score": score_row,
        "rescore": score_result,
    }


@app.post("/resumes/{resume_id}/rescore")
async def rescore_resume(
    resume_id: int, body: RescoreRequest = RescoreRequest()
) -> dict[str, Any]:
    """Re-run hard filters + ATS + embed + LLM without re-parsing."""
    resume = db.get_resume(resume_id)
    if not resume:
        raise HTTPException(404, "Resume not found")
    try:
        result = await job_manager.rescore_resume(
            resume_id, run_llm=body.run_llm
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    results = db.get_results_for_job(resume["job_id"])
    score_row = next(
        (r for r in results if r["resume_id"] == resume_id), None
    )
    return {"result": result, "score": score_row, "resume": db.get_resume(resume_id)}


# ---------------------------------------------------------------------------
# Criteria templates
# ---------------------------------------------------------------------------


@app.get("/templates")
async def list_templates() -> dict[str, Any]:
    return {"templates": db.list_templates()}


@app.post("/templates")
async def create_template(body: TemplateRequest) -> dict[str, Any]:
    tid = db.save_template(body.name.strip(), body.criteria.normalized())
    return {"template": db.get_template(tid)}


@app.get("/templates/{template_id}")
async def get_template(template_id: int) -> dict[str, Any]:
    t = db.get_template(template_id)
    if not t:
        raise HTTPException(404, "Template not found")
    return {"template": t}


@app.delete("/templates/{template_id}")
async def delete_template(template_id: int) -> dict[str, Any]:
    ok = db.delete_template(template_id)
    if not ok:
        raise HTTPException(404, "Template not found")
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=str(settings.get("api_host") or "127.0.0.1"),
        port=int(settings.get("api_port") or 8000),
        reload=False,
        log_level=str(settings.get("log_level") or "info").lower(),
    )
