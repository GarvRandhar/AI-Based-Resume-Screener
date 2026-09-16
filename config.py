"""Central runtime configuration (env overrides optional)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent
_CONFIG_PATH = _ROOT / "config.json"

_DEFAULTS: dict[str, Any] = {
    "api_host": "127.0.0.1",
    "api_port": 8000,
    "ollama_base": "http://localhost:11434",
    "extract_model": "llama3.2:3b",
    # Stronger model for fit scoring only (extract stays small/fast)
    "score_model": "llama3.1:8b",
    "embed_model": "nomic-embed-text",
    # Concurrent Ollama chat (extract + fit) calls
    "default_concurrency": 4,
    # Concurrent embedding calls (separate from chat; embeddings are cheaper)
    "embed_concurrency": 8,
    "llm_timeout_seconds": 120,
    "embed_cache_enabled": True,
    # Extra LLM repair / identity refine passes (slower; off for speed)
    "extract_refine_enabled": False,
    # Extract JSON parse retries (0 = single attempt)
    "extract_max_retries": 1,
    # Hybrid extract everyone first; LLM ranking in phase 2
    "hybrid_first_bulk": True,
    # Extra LLM structured-extract in phase 2 (slow). Fit uses hybrid+raw text when false.
    "phase2_llm_extract": False,
    # If total resumes < this, LLM-score everyone (0 = always use top-K limits)
    "always_full_below": 0,
    # After ranking: LLM-extract skills/work history for shortlisted only (manual trigger)
    "enrich_shortlist": False,
    # Parallel PDF/DOCX parse workers (thread pool)
    "parse_workers": 8,
    # Prefer cheap PyMuPDF text first; full layout only if thin/garbled
    "pdf_fast_mode": True,
    # find_tables() is slow — off by default for speed
    "pdf_extract_tables": False,
    # When fast path quality is low, re-parse with multi-column + tables
    "pdf_auto_tables_on_low_quality": True,
    # Flag candidates seen in other jobs (email/phone)
    "cross_job_dedupe": True,
    # OCR is very slow; only when digital text is missing
    "ocr_enabled": True,
    "ocr_dpi": 150,
    "ocr_max_pages": 3,
    "log_level": "INFO",
}


def _load_file() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_settings() -> dict[str, Any]:
    """Merge defaults ← config.json ← environment."""
    cfg = dict(_DEFAULTS)
    cfg.update(_load_file())
    # Env overrides
    if os.getenv("SCREENER_API_HOST"):
        cfg["api_host"] = os.environ["SCREENER_API_HOST"]
    if os.getenv("SCREENER_API_PORT"):
        try:
            cfg["api_port"] = int(os.environ["SCREENER_API_PORT"])
        except ValueError:
            pass
    if os.getenv("OLLAMA_BASE") or os.getenv("SCREENER_OLLAMA_BASE"):
        cfg["ollama_base"] = os.getenv("SCREENER_OLLAMA_BASE") or os.environ["OLLAMA_BASE"]
    if os.getenv("SCREENER_EXTRACT_MODEL"):
        cfg["extract_model"] = os.environ["SCREENER_EXTRACT_MODEL"]
    if os.getenv("SCREENER_SCORE_MODEL"):
        cfg["score_model"] = os.environ["SCREENER_SCORE_MODEL"]
    if os.getenv("SCREENER_EMBED_MODEL"):
        cfg["embed_model"] = os.environ["SCREENER_EMBED_MODEL"]
    if os.getenv("SCREENER_CONCURRENCY"):
        try:
            cfg["default_concurrency"] = int(os.environ["SCREENER_CONCURRENCY"])
        except ValueError:
            pass
    if os.getenv("SCREENER_EMBED_CONCURRENCY"):
        try:
            cfg["embed_concurrency"] = int(os.environ["SCREENER_EMBED_CONCURRENCY"])
        except ValueError:
            pass
    if os.getenv("SCREENER_EXTRACT_REFINE") is not None:
        cfg["extract_refine_enabled"] = os.environ["SCREENER_EXTRACT_REFINE"].lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    if os.getenv("SCREENER_HYBRID_FIRST") is not None:
        cfg["hybrid_first_bulk"] = os.environ["SCREENER_HYBRID_FIRST"].lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return cfg


def ensure_config_file() -> Path:
    """Write config.json with defaults if missing (user-editable)."""
    if not _CONFIG_PATH.exists():
        _CONFIG_PATH.write_text(
            json.dumps(_DEFAULTS, indent=2) + "\n", encoding="utf-8"
        )
    return _CONFIG_PATH


# Eager defaults for importers
SETTINGS = get_settings()
