"""Async job orchestration: parse → hard-filter → ATS → embed → (bulk) LLM → rank."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import db
import extract_rules
import filters
import ollama_client
import parsing
import scoring
from config import get_settings

logger = logging.getLogger(__name__)

# In-process registry of running job tasks (single FastAPI process)
_running_jobs: dict[int, asyncio.Task] = {}
_default_concurrency = 4
_embed_concurrency = 8

# In-process phase-aware progress tracking (single FastAPI process).
# Pipeline phases: p1 = parse + hybrid extract + ATS + embed, p2 = LLM fit for
# top candidates, p3 = shortlist profile enrichment. Each phase carries its own
# done/total + smoothed throughput so the UI can show a moving bar and an ETA
# even after processed_count hits 100% (LLM/enrich phases).
_PHASE_ORDER = ("p1", "p2", "p3")
_PHASE_WEIGHTS = {"p1": 0.55, "p2": 0.45, "p3": 1.0}
_job_progress: dict[int, dict[str, Any]] = {}


def _progress_init(job_id: int, total: int, p2_total: int = 0) -> None:
    _job_progress[job_id] = {
        "started_at": time.time(),
        "p1": {"total": max(0, int(total)), "done": 0, "t": None, "last": 0, "rate": None},
        "p2": {"total": max(0, int(p2_total)), "done": 0, "t": None, "last": 0, "rate": None},
        "p3": {"total": 0, "done": 0, "t": None, "last": 0, "rate": None},
    }


def _progress_set_total(job_id: int, phase: str, total: int) -> None:
    st = _job_progress.get(job_id)
    if st and phase in st:
        st[phase]["total"] = max(0, int(total))


def _progress_advance(job_id: int, phase: str, done: int) -> None:
    """Record completed work in a phase; keeps a smoothed rate (units/sec)."""
    st = _job_progress.get(job_id)
    if not st or phase not in st:
        return
    ph = st[phase]
    ph["done"] = int(done)
    now = time.time()
    if ph["t"] is not None and now > ph["t"] and ph["done"] > ph["last"]:
        inst = (ph["done"] - ph["last"]) / (now - ph["t"])
        if inst > 0:
            ph["rate"] = inst if ph["rate"] is None else 0.65 * ph["rate"] + 0.35 * inst
    ph["t"] = now
    ph["last"] = ph["done"]


def _progress_remove(job_id: int) -> None:
    _job_progress.pop(job_id, None)


def job_progress_snapshot(job_id: int) -> Optional[dict[str, Any]]:
    """
    Weighted progress + ETA across pipeline phases.
    Returns None when no tracker is present (e.g. process restarted mid-job);
    callers then fall back to processed_count / total_count.
    """
    st = _job_progress.get(job_id)
    if not st:
        return None
    elapsed = time.time() - st["started_at"]
    active = [k for k in _PHASE_ORDER if st[k]["total"] > 0]
    denom = sum(_PHASE_WEIGHTS[k] for k in active) or 1.0
    pct = 0.0
    current: Optional[str] = None
    for k in active:
        ph = st[k]
        frac = min(1.0, ph["done"] / ph["total"]) if ph["total"] > 0 else 0.0
        pct += _PHASE_WEIGHTS[k] * frac
        if current is None and frac < 1.0:
            current = k
    if current is None:
        pct = 100.0
    else:
        pct = (pct / denom) * 100.0

    eta = 0.0
    has_rate = False
    for k in active:
        ph = st[k]
        if ph["total"] > 0 and ph["done"] < ph["total"] and ph["rate"]:
            eta += (ph["total"] - ph["done"]) / ph["rate"]
            has_rate = True

    cur = st[current] if current else None
    return {
        "percent": round(pct, 1),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if has_rate else None,
        "phase_percent": round(100.0 * cur["done"] / cur["total"], 1) if (cur and cur["total"] > 0) else 100.0,
        "phase_done": int(cur["done"]) if cur else None,
        "phase_total": int(cur["total"]) if cur else None,
    }

# Default bulk mode: hybrid pre-rank + LLM fit only on top tier
DEFAULT_BULK_MODE: dict[str, Any] = {
    "enabled": True,
    "llm_top_k": 20,
    "min_pre_score": 30.0,
    # When total < this, LLM-score everyone (0 = always apply top-K / fast path)
    "always_full_below": 0,
    # Phase-1 hybrid only (no Ollama chat until phase 2)
    "hybrid_first": True,
    # Phase-2 LLM structured extract (extra call). Off = fit-only (much faster).
    "phase2_llm_extract": False,
}


def set_default_concurrency(n: int, embed_n: Optional[int] = None) -> None:
    global _default_concurrency, _embed_concurrency
    _default_concurrency = max(1, int(n))
    if embed_n is not None:
        _embed_concurrency = max(1, int(embed_n))
    else:
        settings = get_settings()
        _embed_concurrency = max(
            1, int(settings.get("embed_concurrency") or max(_default_concurrency * 2, 8))
        )
    ollama_client.set_concurrency(_default_concurrency, _embed_concurrency)


async def _wait_if_paused(job_id: int) -> str:
    """
    Block while job control is 'pause'. Returns current control: run|cancel.
    """
    while True:
        ctrl = db.get_job_control(job_id)
        if ctrl == "cancel":
            return "cancel"
        if ctrl != "pause":
            return "run"
        db.update_job(
            job_id,
            status="paused",
            current_stage="paused",
        )
        await asyncio.sleep(0.75)


def request_pause(job_id: int) -> None:
    db.update_job(job_id, control="pause", current_stage="pause_requested")


def request_resume(job_id: int) -> None:
    db.update_job(job_id, control="run", status="processing", current_stage="resuming")


def request_cancel(job_id: int) -> None:
    db.update_job(job_id, control="cancel", current_stage="cancel_requested")
    t = _running_jobs.get(job_id)
    if t and not t.done():
        t.cancel()


def resolve_bulk_mode(criteria: dict[str, Any], total: int) -> dict[str, Any]:
    """
    Merge criteria.bulk_mode with defaults.

    - hybrid_first: always available for small batches (not gated by top-K)
    - limit_llm (enabled): only top-K get phase-2 LLM fit when total is large enough
    - phase2_llm_extract: optional second LLM call for structured extract
    """
    settings = get_settings()
    raw = dict(DEFAULT_BULK_MODE)
    if "hybrid_first_bulk" in settings:
        raw["hybrid_first"] = bool(settings.get("hybrid_first_bulk", True))
    if "phase2_llm_extract" in settings:
        raw["phase2_llm_extract"] = bool(settings.get("phase2_llm_extract", False))
    if "always_full_below" in settings:
        raw["always_full_below"] = int(settings.get("always_full_below") or 0)
    user = criteria.get("bulk_mode") if isinstance(criteria.get("bulk_mode"), dict) else {}
    raw.update({k: v for k, v in (user or {}).items() if v is not None})

    bulk_on = bool(raw.get("enabled", True))
    always_full_below = max(0, int(raw.get("always_full_below", 0)))
    # Limit to top-K only when bulk is on AND batch is large enough (or threshold is 0)
    if not bulk_on:
        limit_llm = False
    elif always_full_below <= 0:
        limit_llm = True
    else:
        limit_llm = total >= always_full_below

    hybrid_first = bool(raw.get("hybrid_first", True))
    phase2_extract = bool(raw.get("phase2_llm_extract", False))
    return {
        "enabled": limit_llm,
        "llm_top_k": max(1, int(raw.get("llm_top_k", 20))),
        "min_pre_score": float(raw.get("min_pre_score", 30.0)),
        "always_full_below": always_full_below,
        "hybrid_first": hybrid_first,
        "phase2_llm_extract": phase2_extract,
    }


async def start_job(
    job_id: int,
    concurrency: Optional[int] = None,
    *,
    resume_ids: Optional[list[int]] = None,
) -> bool:
    """
    Kick off background processing for a job if not already running.
    If resume_ids is set, only those resumes are processed (rerun-failed).
    Returns True if a new task was started.
    """
    if job_id in _running_jobs and not _running_jobs[job_id].done():
        logger.info("Job %s already running", job_id)
        return False

    job = db.get_job(job_id)
    if job is None:
        raise ValueError(f"Job {job_id} not found")

    conc = concurrency if concurrency is not None else _default_concurrency
    set_default_concurrency(conc)
    db.update_job(job_id, control="run")

    task = asyncio.create_task(
        _run_job(job_id, only_resume_ids=resume_ids),
        name=f"screen-job-{job_id}",
    )
    _running_jobs[job_id] = task

    def _cleanup(t: asyncio.Task) -> None:
        _running_jobs.pop(job_id, None)
        try:
            exc = t.exception()
            if exc:
                logger.error("Job %s task failed: %s", job_id, exc)
        except asyncio.CancelledError:
            logger.info("Job %s cancelled", job_id)
            db.update_job(
                job_id,
                status="cancelled",
                current_stage="cancelled",
                current_filename="",
            )

    task.add_done_callback(_cleanup)
    return True


def is_job_running(job_id: int) -> bool:
    t = _running_jobs.get(job_id)
    return t is not None and not t.done()


async def rerun_failed(job_id: int, concurrency: Optional[int] = None) -> dict[str, Any]:
    """Reset failed resumes and reprocess only them."""
    failed = db.list_failed_resumes(job_id)
    if not failed:
        return {"started": False, "count": 0, "message": "No failed resumes"}
    ids = [int(r["id"]) for r in failed]
    for rid in ids:
        db.reset_resume_for_rerun(rid)
    # Adjust processed_count downward by failed count (best effort)
    job = db.get_job(job_id)
    if job:
        done = max(0, int(job.get("processed_count") or 0) - len(ids))
        db.update_job(
            job_id,
            processed_count=done,
            status="processing",
            control="run",
            failed_count=0,
            error_message="",
        )
    started = await start_job(job_id, concurrency=concurrency, resume_ids=ids)
    return {"started": started, "count": len(ids), "resume_ids": ids}


async def _run_job(
    job_id: int, *, only_resume_ids: Optional[list[int]] = None
) -> None:
    logger.info("Starting screening job %s", job_id)
    db.update_job(
        job_id,
        status="processing",
        error_message="",
        control="run",
        current_stage="starting",
        current_filename="",
    )

    try:
        job = db.get_job(job_id)
        if not job:
            return
        criteria = job.get("criteria") or {}
        if only_resume_ids:
            resumes = db.list_resumes_by_ids(only_resume_ids)
            # Keep job total_count as full set; don't reset processed to 0
            total = len(resumes)
            db.update_job(
                job_id,
                current_stage="rerun_failed",
                failed_count=0,
            )
        else:
            resumes = db.list_resumes_for_job(job_id)
            total = len(resumes)
            db.update_job(job_id, total_count=total, processed_count=0, failed_count=0)

        if total == 0:
            _progress_remove(job_id)
            db.update_job(
                job_id,
                status="completed",
                current_stage="done",
                current_filename="",
            )
            return

        bulk = resolve_bulk_mode(
            criteria, total if not only_resume_ids else total
        )
        # Fast path for ALL batch sizes unless explicitly disabled
        hybrid_first = bool(bulk.get("hybrid_first", True)) and not only_resume_ids
        # Defer LLM fit to phase 2 whenever hybrid-first OR top-K limiting is on
        defer_llm = (hybrid_first or bulk["enabled"]) and not only_resume_ids
        limit_llm = bool(bulk["enabled"]) and not only_resume_ids
        phase2_extract = bool(bulk.get("phase2_llm_extract", False))
        p2_est = min(total, int(bulk.get("llm_top_k", 20))) if limit_llm else (total if defer_llm else 0)
        _progress_init(job_id, total, p2_est)
        logger.info(
            "Job %s limit_llm=%s hybrid_first=%s phase2_extract=%s top_k=%s "
            "min_pre=%.1f total=%s only_ids=%s",
            job_id,
            limit_llm,
            hybrid_first,
            phase2_extract,
            bulk["llm_top_k"],
            bulk["min_pre_score"],
            total,
            bool(only_resume_ids),
        )

        # Pre-compute criteria embedding once (cached)
        criteria_blob = scoring.criteria_text_blob(criteria)
        criteria_embedding: list[float] = []
        try:
            db.update_job(job_id, current_stage="embedding_jd")
            criteria_embedding = await ollama_client.embed(criteria_blob)
        except Exception as exc:
            logger.warning("Criteria embedding failed (continuing): %s", exc)

        # Hybrid-first: larger batches so PDF parse + embeds fan out
        settings = get_settings()
        parse_workers = max(2, int(settings.get("parse_workers") or 8))
        if hybrid_first:
            batch_size = max(16, parse_workers * 2, _embed_concurrency * 2)
        else:
            batch_size = max(4, _default_concurrency * 2)
        phase1_passed: list[dict[str, Any]] = []
        failed_n = 0
        t_job = time.perf_counter()

        for i in range(0, total, batch_size):
            ctrl = await _wait_if_paused(job_id)
            if ctrl == "cancel":
                _progress_remove(job_id)
                db.update_job(
                    job_id,
                    status="cancelled",
                    current_stage="cancelled",
                    current_filename="",
                )
                logger.info("Job %s cancelled during phase1", job_id)
                return

            batch = resumes[i : i + batch_size]
            outcomes = await asyncio.gather(
                *[
                    _process_phase1(
                        job_id,
                        r,
                        criteria,
                        criteria_embedding,
                        defer_llm=defer_llm,
                        hybrid_first=hybrid_first,
                    )
                    for r in batch
                ],
                return_exceptions=True,
            )
            for out in outcomes:
                if isinstance(out, Exception):
                    logger.error("Phase1 exception: %s", out)
                    failed_n += 1
                    continue
                if out is None:
                    continue
                if out.get("eligible_for_llm"):
                    phase1_passed.append(out)
                if out.get("failed"):
                    failed_n += 1
            _progress_advance(job_id, "p1", min(i + len(batch), total))

        logger.info(
            "Job %s phase1 done in %.1fs (%s passed filters)",
            job_id,
            time.perf_counter() - t_job,
            len(phase1_passed),
        )

        if await _wait_if_paused(job_id) == "cancel":
            _progress_remove(job_id)
            db.update_job(job_id, status="cancelled", current_stage="cancelled")
            return

        # ----- Phase 2: LLM fit (optional extract) for selected candidates -----
        if defer_llm:
            ranked = sorted(
                phase1_passed,
                key=lambda x: float(x.get("pre_score") or 0.0),
                reverse=True,
            )
            if limit_llm and total > bulk["llm_top_k"]:
                selected = [
                    x
                    for x in ranked
                    if float(x.get("pre_score") or 0.0) >= bulk["min_pre_score"]
                ][: bulk["llm_top_k"]]
                # Never LLM-score zero people if anyone passed filters
                if not selected and ranked:
                    selected = ranked[: bulk["llm_top_k"]]
            else:
                # Small batches (≤ top_k): fit-score everyone who passed filters
                selected = list(ranked)
            selected_ids = {x["resume_id"] for x in selected}
            _progress_set_total(job_id, "p2", len(selected))
            logger.info(
                "Job %s phase2 LLM: %s of %s (hybrid_first=%s extract=%s)",
                job_id,
                len(selected_ids),
                len(phase1_passed),
                hybrid_first,
                phase2_extract,
            )

            skip_msg = (
                "Skipped (fast bulk — not in top tier by ATS+embed). "
                "Open candidate and click Re-score to run LLM fit."
            )
            for item in phase1_passed:
                if item["resume_id"] in selected_ids:
                    continue
                db.upsert_score(
                    item["resume_id"],
                    job_id,
                    hard_filter_pass=1,
                    ats_score=item.get("ats_score"),
                    embedding_similarity=item.get("embed_sim"),
                    llm_score=None,
                    llm_justification=skip_msg,
                    final_score=item.get("pre_score"),
                    force=True,
                )

            llm_batch = max(2, _default_concurrency)
            t_p2 = time.perf_counter()
            for i in range(0, len(selected), llm_batch):
                if await _wait_if_paused(job_id) == "cancel":
                    _progress_remove(job_id)
                    db.update_job(job_id, status="cancelled", current_stage="cancelled")
                    return
                chunk = selected[i : i + llm_batch]
                db.update_job(
                    job_id,
                    current_stage=(
                        "llm_extract_score" if phase2_extract else "llm_scoring"
                    ),
                    current_filename=chunk[0].get("filename") or "",
                )
                await asyncio.gather(
                    *[
                        _llm_score_one(
                            job_id,
                            item,
                            criteria,
                            full_extract=phase2_extract,
                            criteria_embedding=criteria_embedding,
                        )
                        for item in chunk
                    ],
                    return_exceptions=True,
                )
                _progress_advance(job_id, "p2", min(i + len(chunk), len(selected)))
            logger.info(
                "Job %s phase2 done in %.1fs (%s LLM fits)",
                job_id,
                time.perf_counter() - t_p2,
                len(selected),
            )

        db.assign_ranks(job_id)

        db.update_job(
            job_id,
            status="completed",
            current_stage="done",
            current_filename="",
            failed_count=failed_n,
            control="run",
        )
        _progress_remove(job_id)
        logger.info("Job %s completed", job_id)
        # Only cleanup files when full job (not partial rerun) completed
        if not only_resume_ids:
            _cleanup_job_files(job_id)

    except asyncio.CancelledError:
        _progress_remove(job_id)
        db.update_job(
            job_id,
            status="cancelled",
            current_stage="cancelled",
            current_filename="",
        )
        raise
    except Exception as exc:
        logger.exception("Job %s crashed: %s", job_id, exc)
        _progress_remove(job_id)
        db.update_job(
            job_id,
            status="failed",
            error_message=str(exc)[:1000],
            current_stage="error",
        )


async def _process_phase1(
    job_id: int,
    resume_row: dict[str, Any],
    criteria: dict[str, Any],
    criteria_embedding: list[float],
    *,
    defer_llm: bool,
    hybrid_first: bool = False,
) -> Optional[dict[str, Any]]:
    """
    Parse, extract, hard-filter, ATS, embed.
    hybrid_first=True: skip LLM extract (regex hybrid only) for speed.
    If defer_llm=False, also run full LLM extract (if needed) + fit immediately.
    Returns dict for bulk LLM selection when passed hard filters.
    """
    resume_id = resume_row["id"]
    filename = resume_row.get("filename") or f"resume-{resume_id}"
    file_path = resume_row.get("file_path")

    try:
        db.update_job(
            job_id, current_stage="parsing", current_filename=filename
        )
        db.update_resume(resume_id, status="parsing", stage="parsing")
        doc_meta: dict[str, Any] = {}
        sections: dict[str, str] = {}
        raw_text = ""

        if file_path and Path(file_path).exists():
            loop = asyncio.get_running_loop()
            doc = await loop.run_in_executor(
                parsing.get_parse_executor(),
                parsing.extract_document,
                file_path,
            )
            raw_text = doc.get("raw_text") or ""
            sections = doc.get("sections") or {}
            doc_meta = {
                "quality": doc.get("quality"),
                "ocr_used": doc.get("ocr_used"),
                "method": doc.get("method"),
                "text_chars": doc.get("text_chars"),
                "section_hits": doc.get("section_hits") or [],
                "page_count": doc.get("page_count"),
                "multi_column": doc.get("multi_column"),
                "multi_col_pages": doc.get("multi_col_pages"),
            }
        elif resume_row.get("raw_text"):
            raw_text = resume_row["raw_text"]
            sections = extract_rules.split_sections(raw_text)
            doc_meta = {
                "quality": "medium",
                "ocr_used": False,
                "method": "cached_text",
                "text_chars": len("".join(raw_text.split())),
                "section_hits": [
                    k
                    for k in (
                        "summary",
                        "experience",
                        "education",
                        "skills",
                        "certifications",
                    )
                    if sections.get(k)
                ],
            }

        # Skip expensive LLM when text is empty/near-empty
        if not raw_text or len(raw_text.strip()) < 40:
            db.update_resume(
                resume_id,
                status="failed",
                stage="failed_text",
                raw_text=raw_text or "",
                error_message="Could not extract enough text from file",
            )
            db.upsert_score(
                resume_id,
                job_id,
                hard_filter_pass=0,
                hard_filter_results=[
                    {
                        "label": "text_extraction",
                        "passed": False,
                        "reason": "Insufficient text extracted",
                    }
                ],
                force=True,
            )
            db.increment_processed(job_id)
            return {"failed": True, "resume_id": resume_id}

        db.update_resume(resume_id, raw_text=raw_text, stage="extracting")
        db.update_job(
            job_id,
            current_stage="hybrid_extract" if hybrid_first else "extracting",
            current_filename=filename,
        )

        try:
            loop = asyncio.get_running_loop()

            def _run_hybrid() -> dict[str, Any]:
                return extract_rules.hybrid_extract(
                    raw_text, sections, filename=filename
                )

            hybrid = await loop.run_in_executor(
                parsing.get_parse_executor(), _run_hybrid
            )
            if hybrid_first:
                # Fast path: no Ollama chat until top-K phase
                structured = extract_rules.structured_from_hybrid(
                    hybrid,
                    resume_text=raw_text,
                    doc_meta=doc_meta,
                    filename=filename,
                )
            else:
                structured = await ollama_client.extract_structured(
                    raw_text,
                    sections=sections,
                    hybrid=hybrid,
                    doc_meta=doc_meta,
                    filename=filename,
                )
        except Exception as exc:
            logger.warning("Extraction failed for %s: %s", filename, exc)
            db.update_resume(
                resume_id,
                status="failed",
                stage="failed_extract",
                error_message=f"Structured extraction failed: {exc}"[:500],
            )
            db.upsert_score(
                resume_id,
                job_id,
                hard_filter_pass=0,
                hard_filter_results=[
                    {
                        "label": "llm_extraction",
                        "passed": False,
                        "reason": str(exc)[:200],
                    }
                ],
                force=True,
            )
            db.increment_processed(job_id)
            return {"failed": True, "resume_id": resume_id}

        db.update_resume(
            resume_id,
            structured=structured,
            status="parsed",
            stage="filtering",
            error_message="",
        )
        db.update_job(
            job_id, current_stage="filtering", current_filename=filename
        )

        hard_rules = criteria.get("hard_filters") or []
        hf = filters.apply_hard_filters(
            structured, hard_rules, raw_text=raw_text
        )

        if not hf["passed"]:
            reason = filters.rejection_summary(hf)
            db.update_resume(
                resume_id,
                status="rejected",
                stage="rejected",
                error_message=reason[:500],
            )
            db.upsert_score(
                resume_id,
                job_id,
                hard_filter_pass=0,
                hard_filter_results=hf["results"],
                final_score=0.0,
                force=True,
            )
            db.increment_processed(job_id)
            db.update_resume(resume_id, clear_file_path=True)
            return {"failed": False, "resume_id": resume_id, "rejected": True}

        db.update_resume(resume_id, stage="scoring")
        db.update_job(
            job_id, current_stage="scoring", current_filename=filename
        )
        match_text = scoring.resume_text_for_matching(structured, raw_text)
        ats = scoring.ats_keyword_score(match_text, criteria)

        embed_sim = 0.0
        try:
            res_emb = await ollama_client.embed(match_text)
            if criteria_embedding and res_emb:
                embed_sim = scoring.cosine_similarity(criteria_embedding, res_emb)
        except Exception as exc:
            logger.warning("Embed failed for %s: %s", filename, exc)

        pre_score = scoring.combine_scores(
            ats, embed_sim, None, criteria=criteria
        )
        # Brief-point explainability (matched vs missing)
        match_info = scoring.match_brief_points(criteria, structured, raw_text)
        structured = dict(structured or {})
        structured["_ranking"] = match_info

        if not defer_llm:
            llm_score = None
            llm_just = None
            try:
                fit = await ollama_client.score_fit(structured, raw_text, criteria)
                llm_score = fit["score"]
                llm_just = fit["justification"]
            except Exception as exc:
                logger.warning("LLM score failed for %s: %s", filename, exc)
                llm_just = f"LLM scoring unavailable: {exc}"[:300]
            final = scoring.combine_scores(
                ats, embed_sim, llm_score, criteria=criteria
            )
            db.update_resume(
                resume_id,
                structured=structured,
                status="scored",
                stage="done",
                error_message="",
            )
            db.upsert_score(
                resume_id,
                job_id,
                hard_filter_pass=1,
                hard_filter_results=hf["results"],
                ats_score=ats,
                embedding_similarity=embed_sim,
                llm_score=llm_score,
                llm_justification=llm_just,
                final_score=final,
                force=True,
            )
            db.increment_processed(job_id)
            db.update_resume(resume_id, clear_file_path=True)
            return {"failed": False, "resume_id": resume_id}

        # Bulk mode: store pre-scores; LLM later
        db.update_resume(
            resume_id,
            structured=structured,
            status="scored",
            stage="awaiting_llm",
            error_message="",
        )
        db.upsert_score(
            resume_id,
            job_id,
            hard_filter_pass=1,
            hard_filter_results=hf["results"],
            ats_score=ats,
            embedding_similarity=embed_sim,
            llm_score=None,
            llm_justification=(
                "Pending bulk LLM tier (hybrid pre-rank)…"
                if hybrid_first
                else "Pending bulk LLM tier…"
            ),
            final_score=pre_score,
            force=True,
        )
        db.increment_processed(job_id)
        db.update_resume(resume_id, clear_file_path=True)

        return {
            "resume_id": resume_id,
            "eligible_for_llm": True,
            "ats_score": ats,
            "embed_sim": embed_sim,
            "pre_score": pre_score,
            "structured": structured,
            "raw_text": raw_text,
            "filename": filename,
            "doc_meta": doc_meta,
            "sections": sections,
            "hybrid_first": hybrid_first,
        }

    except Exception as exc:
        logger.exception("Unhandled error processing resume %s: %s", resume_id, exc)
        try:
            db.update_resume(
                resume_id,
                status="failed",
                error_message=str(exc)[:500],
            )
            db.upsert_score(
                resume_id,
                job_id,
                hard_filter_pass=0,
                hard_filter_results=[
                    {"label": "pipeline", "passed": False, "reason": str(exc)[:200]}
                ],
                force=True,
            )
            db.increment_processed(job_id)
        except Exception:
            logger.exception("Failed to record error for resume %s", resume_id)
        return None


async def _llm_score_one(
    job_id: int,
    item: dict[str, Any],
    criteria: dict[str, Any],
    *,
    full_extract: bool = False,
    criteria_embedding: Optional[list[float]] = None,
) -> None:
    """
    Phase-2 for a top-tier candidate.
    full_extract=True: run LLM structured extract (after hybrid pre-rank), then fit.
    """
    resume_id = item["resume_id"]
    structured = item.get("structured") or {}
    raw_text = item.get("raw_text") or ""
    ats = item.get("ats_score")
    embed_sim = item.get("embed_sim")
    filename = item.get("filename") or f"resume-{resume_id}"
    doc_meta = item.get("doc_meta") or {}
    sections = item.get("sections") or {}

    if full_extract and raw_text:
        try:
            db.update_job(
                job_id, current_stage="llm_extract", current_filename=filename
            )
            if not sections:
                sections = extract_rules.split_sections(raw_text)
            hybrid = await asyncio.to_thread(
                extract_rules.hybrid_extract,
                raw_text,
                sections,
                filename=filename,
            )
            structured = await ollama_client.extract_structured(
                raw_text,
                sections=sections,
                hybrid=hybrid,
                doc_meta=doc_meta,
                filename=filename,
            )
            match_text = scoring.resume_text_for_matching(structured, raw_text)
            ats = scoring.ats_keyword_score(match_text, criteria)
            try:
                res_emb = await ollama_client.embed(match_text)
                if criteria_embedding and res_emb:
                    embed_sim = scoring.cosine_similarity(
                        criteria_embedding, res_emb
                    )
            except Exception as exc:
                logger.warning("Re-embed after extract failed for %s: %s", resume_id, exc)
            match_info = scoring.match_brief_points(criteria, structured, raw_text)
            structured = dict(structured or {})
            structured["_ranking"] = match_info
            db.update_resume(resume_id, structured=structured, stage="llm_scoring")
        except Exception as exc:
            logger.warning(
                "Full LLM extract failed for resume %s (using hybrid profile): %s",
                resume_id,
                exc,
            )

    try:
        fit = await ollama_client.score_fit(structured, raw_text, criteria)
        llm_score = fit["score"]
        llm_just = fit["justification"]
    except Exception as exc:
        logger.warning("LLM score failed for resume %s: %s", resume_id, exc)
        llm_score = None
        llm_just = f"LLM scoring unavailable: {exc}"[:300]
    final = scoring.combine_scores(
        ats, embed_sim, llm_score, criteria=criteria
    )
    db.update_resume(resume_id, status="scored", stage="done", error_message="")
    db.upsert_score(
        resume_id,
        job_id,
        hard_filter_pass=1,
        ats_score=ats,
        embedding_similarity=embed_sim,
        llm_score=llm_score,
        llm_justification=llm_just,
        final_score=final,
        force=True,
    )


def _profile_needs_enrich(structured: dict[str, Any] | None) -> bool:
    """True if skills/work history look empty (hybrid-only profile)."""
    if not structured or not isinstance(structured, dict):
        return True
    meta = structured.get("_extraction") or {}
    if meta.get("shortlist_enriched"):
        return False
    if meta.get("hybrid_only") and not meta.get("llm_used"):
        return True
    skills = structured.get("skills") or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    work = structured.get("work_history") or []
    if not skills and not work:
        return True
    # Thin profile: no work history and very few skills
    if not work and len(skills) < 2:
        return True
    return False


async def _enrich_one_resume(resume_id: int) -> dict[str, Any]:
    """LLM extract skills/work history for one resume; preserve hard-won identity fields."""
    resume = db.get_resume(resume_id)
    if not resume:
        return {"resume_id": resume_id, "ok": False, "error": "not found"}
    raw_text = resume.get("raw_text") or ""
    if len(raw_text.strip()) < 40:
        return {"resume_id": resume_id, "ok": False, "error": "no text"}
    filename = resume.get("filename") or ""
    old = resume.get("structured") or {}
    if not _profile_needs_enrich(old):
        return {"resume_id": resume_id, "ok": True, "skipped": True, "reason": "already_rich"}

    sections = extract_rules.split_sections(raw_text)
    hybrid = await asyncio.to_thread(
        extract_rules.hybrid_extract, raw_text, sections, filename=filename
    )
    try:
        structured = await ollama_client.extract_structured(
            raw_text,
            sections=sections,
            hybrid=hybrid,
            filename=filename,
            refine=False,
        )
    except Exception as exc:
        return {"resume_id": resume_id, "ok": False, "error": str(exc)[:200]}

    # Prefer prior high-trust identity if new extract is weaker
    for field in ("name", "email", "phone", "location", "total_years_experience"):
        old_v = old.get(field)
        new_v = structured.get(field)
        if old_v not in (None, "", [], {}) and new_v in (None, "", [], {}):
            structured[field] = old_v

    # Re-resolve name with filename
    resolved = extract_rules.resolve_candidate_name(
        llm_name=structured.get("name"),
        hybrid_name=old.get("name"),
        email=structured.get("email") or old.get("email"),
        filename=filename,
        resume_text=raw_text[:2000],
    )
    if resolved.get("name"):
        structured["name"] = resolved["name"]

    meta = dict(structured.get("_extraction") or {})
    meta["shortlist_enriched"] = True
    meta["hybrid_only"] = False
    meta["llm_used"] = True
    structured["_extraction"] = meta
    if old.get("_ranking"):
        structured["_ranking"] = old["_ranking"]

    db.update_resume(
        resume_id,
        structured=structured,
        status="scored",
        stage="enriched",
        error_message="",
    )
    return {
        "resume_id": resume_id,
        "ok": True,
        "skipped": False,
        "skills": len(structured.get("skills") or []),
        "work_history": len(structured.get("work_history") or []),
        "name": structured.get("name"),
    }


async def enrich_shortlist_for_job(
    job_id: int,
    *,
    resume_ids: Optional[list[int]] = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    LLM-enrich shortlisted candidates (skills + work history).
    Safe to call after ranking or when shortlist size changes.
    """
    if is_job_running(job_id) and resume_ids is None:
        # Allow when called from inside the job runner (same task)
        pass

    results = db.get_results_for_job(job_id)
    if resume_ids is not None:
        want = set(int(x) for x in resume_ids)
        targets = [r for r in results if int(r["resume_id"]) in want]
    else:
        targets = [r for r in results if r.get("shortlisted")]

    if force:
        to_run = targets
    else:
        to_run = [
            r
            for r in targets
            if _profile_needs_enrich(r.get("structured") if isinstance(r.get("structured"), dict) else None)
        ]

    if not to_run:
        return {"enriched": 0, "skipped": len(targets), "failed": 0, "total_shortlist": len(targets)}

    # Progress tracker: running inside a job it continues p3; standalone calls
    # (e.g. /enrich-shortlist endpoint) get a fresh tracker here.
    if job_id not in _job_progress:
        _progress_init(job_id, 0)
    _progress_set_total(job_id, "p3", len(to_run))
    db.update_job(job_id, current_stage="enrich_shortlist", status="processing")
    batch = max(2, _default_concurrency)
    enriched = 0
    skipped = 0
    failed = 0
    details: list[dict[str, Any]] = []

    for i in range(0, len(to_run), batch):
        if await _wait_if_paused(job_id) == "cancel":
            _progress_remove(job_id)
            break
        chunk = to_run[i : i + batch]
        db.update_job(
            job_id,
            current_stage="enrich_shortlist",
            current_filename=chunk[0].get("filename") or chunk[0].get("source_filename") or "",
        )
        outs = await asyncio.gather(
            *[_enrich_one_resume(int(r["resume_id"])) for r in chunk],
            return_exceptions=True,
        )
        for out in outs:
            if isinstance(out, Exception):
                failed += 1
                details.append({"ok": False, "error": str(out)[:200]})
                continue
            details.append(out)
            if out.get("skipped"):
                skipped += 1
            elif out.get("ok"):
                enriched += 1
            else:
                failed += 1
        _progress_advance(job_id, "p3", min(i + len(chunk), len(to_run)))

    db.update_job(
        job_id,
        status="completed",
        current_stage="done",
        current_filename="",
    )
    _progress_remove(job_id)

    return {
        "enriched": enriched,
        "skipped": skipped,
        "failed": failed,
        "total_shortlist": len(targets),
        "attempted": len(to_run),
        "details": details[:50],
    }


async def rescore_resume(
    resume_id: int,
    *,
    run_llm: bool = True,
) -> dict[str, Any]:
    """
    Re-run hard filters + ATS + embed + (optional) LLM for one resume
    using current structured JSON (after recruiter overrides). Does not re-parse.
    """
    resume = db.get_resume(resume_id)
    if not resume:
        raise ValueError(f"Resume {resume_id} not found")
    job_id = resume["job_id"]
    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    if is_job_running(job_id):
        raise RuntimeError("Cannot rescore while the parent job is still processing")

    criteria = job.get("criteria") or {}
    structured = resume.get("structured") or {}
    raw_text = resume.get("raw_text") or ""

    if not structured:
        raise ValueError("No structured data to score — re-run the job instead")

    # Hard filters (fuzzy match)
    hard_rules = criteria.get("hard_filters") or []
    hf = filters.apply_hard_filters(
        structured, hard_rules, raw_text=raw_text
    )

    if not hf["passed"]:
        reason = filters.rejection_summary(hf)
        db.update_resume(resume_id, status="rejected", error_message=reason[:500])
        db.upsert_score(
            resume_id,
            job_id,
            hard_filter_pass=0,
            hard_filter_results=hf["results"],
            ats_score=None,
            embedding_similarity=None,
            llm_score=None,
            llm_justification=None,
            final_score=0.0,
            rank=None,
            force=True,
        )
        db.assign_ranks(job_id)
        return {"resume_id": resume_id, "status": "rejected", "reason": reason}

    match_text = scoring.resume_text_for_matching(structured, raw_text)
    ats = scoring.ats_keyword_score(match_text, criteria)

    embed_sim = 0.0
    try:
        criteria_blob = scoring.criteria_text_blob(criteria)
        criteria_embedding = await ollama_client.embed(criteria_blob)
        res_emb = await ollama_client.embed(match_text)
        if criteria_embedding and res_emb:
            embed_sim = scoring.cosine_similarity(criteria_embedding, res_emb)
    except Exception as exc:
        logger.warning("Rescore embed failed: %s", exc)

    llm_score = None
    llm_just = None
    if run_llm:
        try:
            fit = await ollama_client.score_fit(structured, raw_text, criteria)
            llm_score = fit["score"]
            llm_just = fit["justification"]
        except Exception as exc:
            llm_just = f"LLM scoring unavailable: {exc}"[:300]

    match_info = scoring.match_brief_points(criteria, structured, raw_text)
    structured = dict(structured or {})
    structured["_ranking"] = match_info
    final = scoring.combine_scores(
        ats, embed_sim, llm_score, criteria=criteria
    )
    db.upsert_score(
        resume_id,
        job_id,
        hard_filter_pass=1,
        hard_filter_results=hf["results"],
        ats_score=ats,
        embedding_similarity=embed_sim,
        llm_score=llm_score,
        llm_justification=llm_just,
        final_score=final,
        force=True,
    )
    db.update_resume(resume_id, structured=structured, status="scored", error_message="")
    db.assign_ranks(job_id)
    return {
        "resume_id": resume_id,
        "status": "scored",
        "ats_score": ats,
        "embedding_similarity": embed_sim,
        "llm_score": llm_score,
        "final_score": final,
    }


def apply_overrides_and_save(
    resume_id: int, overrides: dict[str, Any]
) -> dict[str, Any]:
    """Merge recruiter overrides into structured JSON and persist."""
    resume = db.get_resume(resume_id)
    if not resume:
        raise ValueError(f"Resume {resume_id} not found")
    structured = resume.get("structured") or {}
    updated = extract_rules.apply_recruiter_overrides(structured, overrides)
    # Preserve extraction meta if present
    if structured.get("_extraction"):
        meta = dict(structured["_extraction"])
        meta["recruiter_edited"] = True
        updated["_extraction"] = meta
    db.update_resume(resume_id, structured=updated, error_message="")
    return updated


def _cleanup_job_files(job_id: int) -> None:
    """Remove job upload directory after processing."""
    job_dir = db.UPLOADS_DIR / f"job_{job_id}"
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.info("Cleaned upload dir for job %s", job_id)
        except Exception as exc:
            logger.warning("Cleanup failed for job %s: %s", job_id, exc)
