"""SQLite database layer for the resume screening platform."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

DB_PATH = Path(__file__).parent / "data" / "screener.db"
UPLOADS_DIR = Path(__file__).parent / "uploads"

_local = threading.local()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with row factory."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        ensure_dirs()
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        _local.conn = conn
    return conn


@contextmanager
def db_cursor() -> Generator[sqlite3.Cursor, None, None]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, decl: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    cols = {row[1] for row in cur.fetchall()}
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_db() -> None:
    """Create tables if they do not exist; migrate additive columns."""
    ensure_dirs()
    with db_cursor() as cur:
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                criteria_json TEXT NOT NULL DEFAULT '{}',
                total_count INTEGER NOT NULL DEFAULT 0,
                processed_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS criteria_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                criteria_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT,
                raw_text TEXT,
                structured_json TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER NOT NULL UNIQUE,
                job_id INTEGER NOT NULL,
                hard_filter_pass INTEGER NOT NULL DEFAULT 0,
                hard_filter_results TEXT,
                ats_score REAL,
                embedding_similarity REAL,
                llm_score REAL,
                llm_justification TEXT,
                final_score REAL,
                rank INTEGER,
                FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE CASCADE,
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_resumes_job ON resumes(job_id);
            CREATE INDEX IF NOT EXISTS idx_scores_job ON scores(job_id);
            CREATE INDEX IF NOT EXISTS idx_scores_final ON scores(job_id, final_score DESC);
            """
        )
        # Jobs: live progress + control flags
        for col, decl in (
            ("current_stage", "TEXT"),
            ("current_filename", "TEXT"),
            ("control", "TEXT DEFAULT 'run'"),  # run | pause | cancel
            ("failed_count", "INTEGER DEFAULT 0"),
        ):
            _ensure_column(cur, "jobs", col, decl)
        # Resumes: HR workflow fields + processing stage
        for col, decl in (
            ("hr_status", "TEXT DEFAULT 'none'"),  # none | starred | rejected
            ("hr_notes", "TEXT"),
            ("stage", "TEXT"),  # parsing | extracting | filtering | scoring | done
        ):
            _ensure_column(cur, "resumes", col, decl)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def create_job(name: str, criteria: dict[str, Any]) -> int:
    now = _utc_now()
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (name, status, criteria_json, total_count,
                              processed_count, created_at, updated_at)
            VALUES (?, 'pending', ?, 0, 0, ?, ?)
            """,
            (name, json.dumps(criteria), now, now),
        )
        return int(cur.lastrowid)


def get_job(job_id: int) -> Optional[dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_job(row)


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [_row_to_job(r) for r in cur.fetchall()]


def update_job(
    job_id: int,
    *,
    status: Optional[str] = None,
    total_count: Optional[int] = None,
    processed_count: Optional[int] = None,
    error_message: Optional[str] = None,
    criteria: Optional[dict[str, Any]] = None,
    current_stage: Optional[str] = None,
    current_filename: Optional[str] = None,
    control: Optional[str] = None,
    failed_count: Optional[int] = None,
) -> None:
    fields: list[str] = ["updated_at = ?"]
    values: list[Any] = [_utc_now()]
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if total_count is not None:
        fields.append("total_count = ?")
        values.append(total_count)
    if processed_count is not None:
        fields.append("processed_count = ?")
        values.append(processed_count)
    if error_message is not None:
        fields.append("error_message = ?")
        values.append(error_message)
    if criteria is not None:
        fields.append("criteria_json = ?")
        values.append(json.dumps(criteria))
    if current_stage is not None:
        fields.append("current_stage = ?")
        values.append(current_stage)
    if current_filename is not None:
        fields.append("current_filename = ?")
        values.append(current_filename)
    if control is not None:
        fields.append("control = ?")
        values.append(control)
    if failed_count is not None:
        fields.append("failed_count = ?")
        values.append(failed_count)
    values.append(job_id)
    with db_cursor() as cur:
        cur.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values
        )


def get_job_control(job_id: int) -> str:
    job = get_job(job_id)
    if not job:
        return "cancel"
    return (job.get("control") or "run").lower()


def increment_processed(job_id: int) -> None:
    with db_cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET processed_count = processed_count + 1,
                updated_at = ?
            WHERE id = ?
            """,
            (_utc_now(), job_id),
        )


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["criteria"] = json.loads(d.pop("criteria_json") or "{}")
    except json.JSONDecodeError:
        d["criteria"] = {}
    return d


# ---------------------------------------------------------------------------
# Resumes
# ---------------------------------------------------------------------------


def create_resume(job_id: int, filename: str, file_path: str) -> int:
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO resumes (job_id, filename, file_path, status, created_at)
            VALUES (?, ?, ?, 'pending', ?)
            """,
            (job_id, filename, file_path, _utc_now()),
        )
        return int(cur.lastrowid)


def get_resume(resume_id: int) -> Optional[dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_resume(row)


def list_resumes_for_job(job_id: int) -> list[dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM resumes WHERE job_id = ? ORDER BY id", (job_id,)
        )
        return [_row_to_resume(r) for r in cur.fetchall()]


def update_resume(
    resume_id: int,
    *,
    status: Optional[str] = None,
    raw_text: Optional[str] = None,
    structured: Optional[dict[str, Any]] = None,
    error_message: Optional[str] = None,
    clear_file_path: bool = False,
    hr_status: Optional[str] = None,
    hr_notes: Optional[str] = None,
    stage: Optional[str] = None,
) -> None:
    fields: list[str] = []
    values: list[Any] = []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if raw_text is not None:
        fields.append("raw_text = ?")
        values.append(raw_text)
    if structured is not None:
        fields.append("structured_json = ?")
        values.append(json.dumps(structured))
    if error_message is not None:
        fields.append("error_message = ?")
        values.append(error_message)
    if clear_file_path:
        fields.append("file_path = NULL")
    if hr_status is not None:
        fields.append("hr_status = ?")
        values.append(hr_status)
    if hr_notes is not None:
        fields.append("hr_notes = ?")
        values.append(hr_notes)
    if stage is not None:
        fields.append("stage = ?")
        values.append(stage)
    if not fields:
        return
    values.append(resume_id)
    with db_cursor() as cur:
        cur.execute(
            f"UPDATE resumes SET {', '.join(fields)} WHERE id = ?", values
        )


def list_failed_resumes(job_id: int) -> list[dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM resumes
            WHERE job_id = ? AND status = 'failed'
            ORDER BY id
            """,
            (job_id,),
        )
        return [_row_to_resume(r) for r in cur.fetchall()]


def list_resumes_by_ids(resume_ids: list[int]) -> list[dict[str, Any]]:
    if not resume_ids:
        return []
    placeholders = ",".join("?" * len(resume_ids))
    with db_cursor() as cur:
        cur.execute(
            f"SELECT * FROM resumes WHERE id IN ({placeholders}) ORDER BY id",
            resume_ids,
        )
        return [_row_to_resume(r) for r in cur.fetchall()]


def reset_resume_for_rerun(resume_id: int) -> None:
    """Clear scores/status so a failed resume can be reprocessed."""
    with db_cursor() as cur:
        cur.execute("DELETE FROM scores WHERE resume_id = ?", (resume_id,))
        cur.execute(
            """
            UPDATE resumes
            SET status = 'pending', error_message = NULL, stage = NULL
            WHERE id = ?
            """,
            (resume_id,),
        )


def get_embedding_cache(cache_key: str) -> Optional[list[float]]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT embedding_json FROM embedding_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return list(json.loads(row["embedding_json"]))
        except json.JSONDecodeError:
            return None


def set_embedding_cache(cache_key: str, model: str, embedding: list[float]) -> None:
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO embedding_cache (cache_key, model, embedding_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                embedding_json = excluded.embedding_json,
                model = excluded.model,
                created_at = excluded.created_at
            """,
            (cache_key, model, json.dumps(embedding), _utc_now()),
        )


def _row_to_resume(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    raw = d.pop("structured_json", None)
    if raw:
        try:
            d["structured"] = json.loads(raw)
        except json.JSONDecodeError:
            d["structured"] = None
    else:
        d["structured"] = None
    return d


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def upsert_score(
    resume_id: int,
    job_id: int,
    *,
    hard_filter_pass: int = 0,
    hard_filter_results: Optional[list | dict] = None,
    ats_score: Optional[float] = None,
    embedding_similarity: Optional[float] = None,
    llm_score: Optional[float] = None,
    llm_justification: Optional[str] = None,
    final_score: Optional[float] = None,
    rank: Optional[int] = None,
    force: bool = False,
) -> None:
    """
    Upsert score row. When force=True, overwrite all score fields
    (needed for rescore / bulk-mode updates). Otherwise COALESCE keeps
    existing non-null values when new ones are None.
    """
    hf_json = (
        json.dumps(hard_filter_results) if hard_filter_results is not None else None
    )
    if force:
        update_sql = """
            ON CONFLICT(resume_id) DO UPDATE SET
                hard_filter_pass = excluded.hard_filter_pass,
                hard_filter_results = excluded.hard_filter_results,
                ats_score = excluded.ats_score,
                embedding_similarity = excluded.embedding_similarity,
                llm_score = excluded.llm_score,
                llm_justification = excluded.llm_justification,
                final_score = excluded.final_score,
                rank = excluded.rank
        """
    else:
        update_sql = """
            ON CONFLICT(resume_id) DO UPDATE SET
                hard_filter_pass = excluded.hard_filter_pass,
                hard_filter_results = COALESCE(excluded.hard_filter_results, scores.hard_filter_results),
                ats_score = COALESCE(excluded.ats_score, scores.ats_score),
                embedding_similarity = COALESCE(excluded.embedding_similarity, scores.embedding_similarity),
                llm_score = COALESCE(excluded.llm_score, scores.llm_score),
                llm_justification = COALESCE(excluded.llm_justification, scores.llm_justification),
                final_score = COALESCE(excluded.final_score, scores.final_score),
                rank = COALESCE(excluded.rank, scores.rank)
        """
    with db_cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO scores (
                resume_id, job_id, hard_filter_pass, hard_filter_results,
                ats_score, embedding_similarity, llm_score, llm_justification,
                final_score, rank
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            {update_sql}
            """,
            (
                resume_id,
                job_id,
                hard_filter_pass,
                hf_json,
                ats_score,
                embedding_similarity,
                llm_score,
                llm_justification,
                final_score,
                rank,
            ),
        )


def get_shortlist_size_for_job(job_id: int) -> int:
    """Read shortlist_size from job criteria (default 5)."""
    job = get_job(job_id)
    if not job:
        return 5
    criteria = job.get("criteria") or {}
    try:
        n = int(criteria.get("shortlist_size") or 5)
    except (TypeError, ValueError):
        n = 5
    return max(1, min(500, n))


def get_min_final_score_for_job(job_id: int) -> float:
    job = get_job(job_id)
    if not job:
        return 0.0
    try:
        from scoring import get_min_final_score

        return get_min_final_score(job.get("criteria") or {})
    except Exception:
        return 0.0


def _name_needs_review(
    *,
    name: Any,
    filename: Any,
    confidence: Any,
    source: Any,
    warning: Any,
    structured: Any,
) -> bool:
    """True when HR should verify the candidate name before outreach."""
    conf = (confidence or "").lower()
    src = (source or "").lower()
    name_s = (str(name).strip() if name else "") or ""
    fn = (str(filename).strip() if filename else "") or ""

    if conf == "low":
        return True
    if warning:
        return True
    if not name_s:
        return True
    # Heuristic-only with no email/filename consensus
    if src in ("heuristic", "none", "rejected", ""):
        return True
    if src == "llm" and conf != "high":
        return True
    # Single-token name is weak for outreach
    if len(name_s.split()) < 2 and "email" not in src:
        return True
    # Structured extraction trust warnings on name
    if isinstance(structured, dict):
        ext = structured.get("_extraction") or {}
        field_conf = ext.get("field_confidence") or structured.get("_confidence") or {}
        if field_conf.get("name") == "low":
            return True
        for w in ext.get("trust_warnings") or []:
            if isinstance(w, str) and "name" in w.lower():
                return True
    return False


def get_results_for_job(job_id: int) -> list[dict[str, Any]]:
    """Return resumes joined with scores, ranked by final_score desc."""
    shortlist_size = get_shortlist_size_for_job(job_id)
    min_final = get_min_final_score_for_job(job_id)
    exclude_dups = True
    job = get_job(job_id)
    if job and isinstance(job.get("criteria"), dict):
        exclude_dups = bool(job["criteria"].get("exclude_duplicates_from_shortlist", True))
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT
                r.id AS resume_id,
                r.job_id,
                r.filename,
                r.status AS resume_status,
                r.error_message,
                r.structured_json,
                r.raw_text,
                r.hr_status,
                r.hr_notes,
                r.stage,
                s.hard_filter_pass,
                s.hard_filter_results,
                s.ats_score,
                s.embedding_similarity,
                s.llm_score,
                s.llm_justification,
                s.final_score,
                s.rank
            FROM resumes r
            LEFT JOIN scores s ON s.resume_id = r.id
            WHERE r.job_id = ?
            ORDER BY
                CASE WHEN s.final_score IS NULL THEN 1 ELSE 0 END,
                s.final_score DESC,
                r.id ASC
            """,
            (job_id,),
        )
        results = []
        for row in cur.fetchall():
            d = dict(row)
            for key, default in (
                ("structured_json", None),
                ("hard_filter_results", None),
            ):
                raw = d.pop(key, None)
                out_key = "structured" if key == "structured_json" else "hard_filter_results"
                if raw:
                    try:
                        d[out_key] = json.loads(raw)
                    except json.JSONDecodeError:
                        d[out_key] = None
                else:
                    d[out_key] = default
            # Display name: re-resolve from structured + filename (portal PDFs)
            name = None
            structured = d.get("structured") if isinstance(d.get("structured"), dict) else None
            name_conf = None
            name_src = None
            name_warn = None
            try:
                from extract_rules import resolve_candidate_name

                resolved = resolve_candidate_name(
                    llm_name=(structured or {}).get("name"),
                    hybrid_name=None,
                    email=(structured or {}).get("email"),
                    filename=d.get("filename"),
                    resume_text=(d.get("raw_text") or "")[:1500],
                )
                name = resolved.get("name")
                name_conf = resolved.get("confidence")
                name_src = resolved.get("source")
                name_warn = resolved.get("warning")
                d["name_confidence"] = name_conf
                d["name_source"] = name_src
                if name_warn:
                    d["name_warning"] = name_warn
            except Exception:
                if structured:
                    name = structured.get("name")
            d["candidate_name"] = name or d.get("filename") or "Unknown"
            d["source_filename"] = d.get("filename")
            # Low-confidence name queue flag (for HR verify UI)
            d["name_needs_review"] = _name_needs_review(
                name=name,
                filename=d.get("filename"),
                confidence=name_conf,
                source=name_src,
                warning=name_warn,
                structured=structured,
            )
            # ranking explainability from structured
            if isinstance(d.get("structured"), dict):
                d["match_points"] = (d["structured"].get("_ranking") or {})
            else:
                d["match_points"] = {}
            d["shortlist_size"] = shortlist_size
            d["min_final_score"] = min_final
            results.append(d)

        # Duplicates + shortlist eligibility (rank + min score + de-dupe)
        try:
            from scoring import find_duplicates, is_shortlist_eligible

            dup_map = find_duplicates(results)
        except Exception:
            dup_map = {}

        for d in results:
            rid = d.get("resume_id")
            dup = dup_map.get(int(rid), {}) if rid is not None else {}
            d["is_duplicate"] = bool(dup.get("is_duplicate"))
            d["duplicate_of"] = dup.get("duplicate_of") or []
            d["duplicate_reasons"] = dup.get("reasons") or []
            d["duplicate_peers"] = dup.get("duplicate_peers") or []
            try:
                rank_i = int(d["rank"]) if d.get("rank") is not None else None
            except (TypeError, ValueError):
                rank_i = None
            try:
                from scoring import is_shortlist_eligible as _elig

                d["shortlisted"] = _elig(
                    rank=rank_i,
                    final_score=d.get("final_score"),
                    hard_filter_pass=d.get("hard_filter_pass"),
                    shortlist_size=shortlist_size,
                    min_final_score=min_final,
                    is_duplicate=d["is_duplicate"],
                    exclude_duplicates=exclude_dups,
                )
            except Exception:
                d["shortlisted"] = bool(
                    rank_i is not None
                    and rank_i <= shortlist_size
                    and int(d.get("hard_filter_pass") or 0) == 1
                )
            # Below min score flag for UI
            try:
                d["below_min_score"] = (
                    d.get("final_score") is not None
                    and float(d["final_score"]) < min_final
                    and min_final > 0
                )
            except (TypeError, ValueError):
                d["below_min_score"] = False

        # Cross-job history (email/phone seen in other drives)
        try:
            from config import get_settings as _gs

            if bool(_gs().get("cross_job_dedupe", True)):
                for d in results:
                    structured = (
                        d.get("structured")
                        if isinstance(d.get("structured"), dict)
                        else {}
                    )
                    prior = find_prior_appearances(
                        email=structured.get("email"),
                        phone=structured.get("phone"),
                        exclude_job_id=job_id,
                        exclude_resume_id=int(d["resume_id"])
                        if d.get("resume_id") is not None
                        else None,
                        limit=8,
                    )
                    d["seen_in_other_jobs"] = prior
                    d["seen_elsewhere"] = len(prior) > 0
            else:
                for d in results:
                    d["seen_in_other_jobs"] = []
                    d["seen_elsewhere"] = False
        except Exception:
            for d in results:
                d.setdefault("seen_in_other_jobs", [])
                d.setdefault("seen_elsewhere", False)

        return results


def find_prior_appearances(
    *,
    email: Any = None,
    phone: Any = None,
    exclude_job_id: Optional[int] = None,
    exclude_resume_id: Optional[int] = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Find the same person (email or phone) in other jobs.
    Returns list of {job_id, job_name, resume_id, candidate_name, final_score, rank, status}.
    """
    import re as _re

    email_s = str(email or "").strip().lower()
    digits = _re.sub(r"\D", "", str(phone or ""))
    phone_s = digits[-10:] if len(digits) >= 10 else ""
    if not email_s and not phone_s:
        return []

    with db_cursor() as cur:
        cur.execute(
            """
            SELECT
                r.id AS resume_id,
                r.job_id,
                r.filename,
                r.status AS resume_status,
                r.structured_json,
                j.name AS job_name,
                j.created_at AS job_created_at,
                s.final_score,
                s.rank,
                s.hard_filter_pass
            FROM resumes r
            JOIN jobs j ON j.id = r.job_id
            LEFT JOIN scores s ON s.resume_id = r.id
            ORDER BY j.created_at DESC, r.id DESC
            LIMIT 800
            """
        )
        rows = cur.fetchall()

    out: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    for row in rows:
        rid = int(row["resume_id"])
        jid = int(row["job_id"])
        if exclude_resume_id is not None and rid == int(exclude_resume_id):
            continue
        if exclude_job_id is not None and jid == int(exclude_job_id):
            continue
        structured = {}
        raw = row["structured_json"]
        if raw:
            try:
                structured = json.loads(raw) or {}
            except json.JSONDecodeError:
                structured = {}
        e = str(structured.get("email") or "").strip().lower()
        p_digits = _re.sub(r"\D", "", str(structured.get("phone") or ""))
        p = p_digits[-10:] if len(p_digits) >= 10 else ""
        match_reasons: list[str] = []
        if email_s and e and e == email_s:
            match_reasons.append("same email")
        if phone_s and p and p == phone_s:
            match_reasons.append("same phone")
        if not match_reasons:
            continue
        key = (jid, rid)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        name = structured.get("name")
        out.append(
            {
                "job_id": jid,
                "job_name": row["job_name"],
                "job_created_at": row["job_created_at"],
                "resume_id": rid,
                "filename": row["filename"],
                "candidate_name": name,
                "final_score": row["final_score"],
                "rank": row["rank"],
                "resume_status": row["resume_status"],
                "match_reasons": match_reasons,
            }
        )
        if len(out) >= limit:
            break
    return out


def assign_ranks(job_id: int) -> None:
    """Recompute rank for all scored/passed resumes in a job."""
    with db_cursor() as cur:
        # Clear ranks first so rejected/failed are not left with stale ranks
        cur.execute(
            "UPDATE scores SET rank = NULL WHERE job_id = ?",
            (job_id,),
        )
        cur.execute(
            """
            SELECT resume_id FROM scores
            WHERE job_id = ? AND hard_filter_pass = 1 AND final_score IS NOT NULL
            ORDER BY final_score DESC, resume_id ASC
            """,
            (job_id,),
        )
        ranked = cur.fetchall()
        for i, row in enumerate(ranked, start=1):
            cur.execute(
                "UPDATE scores SET rank = ? WHERE resume_id = ?",
                (i, row["resume_id"]),
            )


# ---------------------------------------------------------------------------
# Criteria templates
# ---------------------------------------------------------------------------


def save_template(name: str, criteria: dict[str, Any]) -> int:
    now = _utc_now()
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO criteria_templates (name, criteria_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                criteria_json = excluded.criteria_json,
                created_at = excluded.created_at
            """,
            (name, json.dumps(criteria), now),
        )
        cur.execute(
            "SELECT id FROM criteria_templates WHERE name = ?", (name,)
        )
        return int(cur.fetchone()["id"])


def list_templates() -> list[dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM criteria_templates ORDER BY name"
        )
        out = []
        for row in cur.fetchall():
            d = dict(row)
            try:
                d["criteria"] = json.loads(d.pop("criteria_json") or "{}")
            except json.JSONDecodeError:
                d["criteria"] = {}
            out.append(d)
        return out


def get_template(template_id: int) -> Optional[dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM criteria_templates WHERE id = ?", (template_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["criteria"] = json.loads(d.pop("criteria_json") or "{}")
        except json.JSONDecodeError:
            d["criteria"] = {}
        return d


def delete_template(template_id: int) -> bool:
    with db_cursor() as cur:
        cur.execute(
            "DELETE FROM criteria_templates WHERE id = ?", (template_id,)
        )
        return cur.rowcount > 0


# Initialize on import
init_db()
