"""Async HTTP client for local Ollama (extraction, scoring, embeddings)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

import httpx

import extract_rules
from config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
OLLAMA_BASE = _settings.get("ollama_base") or "http://localhost:11434"
EXTRACT_MODEL = _settings.get("extract_model") or "llama3.2:3b"
SCORE_MODEL = _settings.get("score_model") or _settings.get("extract_model") or "llama3.1:8b"
EMBED_MODEL = _settings.get("embed_model") or "nomic-embed-text"
DEFAULT_TIMEOUT = float(_settings.get("llm_timeout_seconds") or 120)
EXTRACT_REFINE_ENABLED = bool(_settings.get("extract_refine_enabled", False))
EXTRACT_MAX_RETRIES = int(_settings.get("extract_max_retries", 1))

# Shared HTTP client (keep-alive) — recreated when base/timeout changes
_http_client: Optional[httpx.AsyncClient] = None
_http_client_key: Optional[tuple[str, float]] = None

# Separate semaphores: chat (extract/score) vs embeddings
_llm_semaphore: Optional[asyncio.Semaphore] = None
_embed_semaphore: Optional[asyncio.Semaphore] = None


def reload_settings() -> None:
    """Reload models/base from config (after config.json edit)."""
    global OLLAMA_BASE, EXTRACT_MODEL, SCORE_MODEL, EMBED_MODEL, DEFAULT_TIMEOUT
    global EXTRACT_REFINE_ENABLED, EXTRACT_MAX_RETRIES, _settings, _http_client_key
    _settings = get_settings()
    OLLAMA_BASE = _settings.get("ollama_base") or OLLAMA_BASE
    EXTRACT_MODEL = _settings.get("extract_model") or EXTRACT_MODEL
    SCORE_MODEL = (
        _settings.get("score_model")
        or _settings.get("extract_model")
        or SCORE_MODEL
    )
    EMBED_MODEL = _settings.get("embed_model") or EMBED_MODEL
    DEFAULT_TIMEOUT = float(_settings.get("llm_timeout_seconds") or DEFAULT_TIMEOUT)
    EXTRACT_REFINE_ENABLED = bool(_settings.get("extract_refine_enabled", False))
    EXTRACT_MAX_RETRIES = max(0, int(_settings.get("extract_max_retries", 1)))
    # Force client rebuild on next request if base/timeout changed
    _http_client_key = None


def set_concurrency(n: int, embed_n: Optional[int] = None) -> None:
    """Set concurrent Ollama chat limit and optional separate embed limit."""
    global _llm_semaphore, _embed_semaphore
    llm_n = max(1, int(n))
    emb_n = max(1, int(embed_n)) if embed_n is not None else max(llm_n * 2, 8)
    _llm_semaphore = asyncio.Semaphore(llm_n)
    _embed_semaphore = asyncio.Semaphore(emb_n)
    logger.info("Ollama concurrency: llm=%s embed=%s", llm_n, emb_n)


def get_llm_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        n = int(_settings.get("default_concurrency") or 4)
        _llm_semaphore = asyncio.Semaphore(max(1, n))
    return _llm_semaphore


def get_embed_semaphore() -> asyncio.Semaphore:
    global _embed_semaphore
    if _embed_semaphore is None:
        n = int(_settings.get("embed_concurrency") or 8)
        _embed_semaphore = asyncio.Semaphore(max(1, n))
    return _embed_semaphore


# Back-compat alias used by older call sites
def get_semaphore() -> asyncio.Semaphore:
    return get_llm_semaphore()


async def get_http_client() -> httpx.AsyncClient:
    """Shared AsyncClient with connection pooling."""
    global _http_client, _http_client_key
    key = (OLLAMA_BASE, DEFAULT_TIMEOUT)
    if (
        _http_client is None
        or _http_client.is_closed
        or _http_client_key != key
    ):
        if _http_client is not None and not _http_client.is_closed:
            try:
                await _http_client.aclose()
            except Exception:
                pass
        _http_client = httpx.AsyncClient(
            base_url=OLLAMA_BASE,
            timeout=DEFAULT_TIMEOUT,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
        _http_client_key = key
    return _http_client


async def close_http_client() -> None:
    global _http_client, _http_client_key
    if _http_client is not None:
        try:
            await _http_client.aclose()
        except Exception:
            pass
        _http_client = None
        _http_client_key = None


async def check_ollama() -> dict[str, Any]:
    """Health check: is Ollama up and do we have required models?"""
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE, timeout=5.0) as client:
            r = await client.get("/api/tags")
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            def _has(model: str) -> bool:
                return any(
                    model in m or m.split(":")[0] == model.split(":")[0]
                    for m in models
                )

            has_llm = _has(EXTRACT_MODEL)
            has_score = _has(SCORE_MODEL)
            has_embed = _has(EMBED_MODEL)
            return {
                "ok": True,
                "base": OLLAMA_BASE,
                "extract_model": EXTRACT_MODEL,
                "score_model": SCORE_MODEL,
                "embed_model": EMBED_MODEL,
                "models": models,
                "has_extract_model": has_llm,
                "has_score_model": has_score,
                "has_embed_model": has_embed,
                "timeout_seconds": DEFAULT_TIMEOUT,
                "extract_refine_enabled": EXTRACT_REFINE_ENABLED,
                "hybrid_first_bulk": bool(_settings.get("hybrid_first_bulk", True)),
                "default_concurrency": int(_settings.get("default_concurrency") or 4),
                "embed_concurrency": int(_settings.get("embed_concurrency") or 8),
            }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "base": OLLAMA_BASE,
            "extract_model": EXTRACT_MODEL,
            "score_model": SCORE_MODEL,
            "embed_model": EMBED_MODEL,
            "models": [],
            "has_extract_model": False,
            "has_score_model": False,
            "has_embed_model": False,
            "timeout_seconds": DEFAULT_TIMEOUT,
            "extract_refine_enabled": EXTRACT_REFINE_ENABLED,
            "hybrid_first_bulk": bool(_settings.get("hybrid_first_bulk", True)),
            "default_concurrency": int(_settings.get("default_concurrency") or 4),
            "embed_concurrency": int(_settings.get("embed_concurrency") or 8),
        }


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort parse of JSON from model output (handles fences / noise)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse JSON from model output: {text[:300]!r}")


EXTRACTION_SYSTEM = """You are an expert multi-industry resume parser for HR screening
(any domain: tech, sales, healthcare, operations, campus, finance, trades, etc.).

Return ONLY a valid JSON object with exactly these keys:
{
  "name": string or null,
  "email": string or null,
  "phone": string or null,
  "location": string or null,
  "total_years_experience": number (float),
  "education": [{"degree": string, "field": string, "institution": string, "year": string or null}],
  "licenses": [string],
  "certifications": [string],
  "skills": [string],
  "work_history": [{"title": string, "company": string, "duration": string, "years": number or null}],
  "notice_period_days": number or null,
  "summary": string
}

STRICT RULES (industry-agnostic):
- Never invent facts not supported by the resume text. Prefer null over guessing.
- name: real person name only. NEVER skills, tools, section headers, or job titles alone.
- location: city / region / country only. NEVER skills, tools, or course lists.
- total_years_experience: professional work only (jobs/internships). Never school year spans.
- work_history: only real employment or internships (title and/or company). Not education or projects.
- education: degrees, diplomas, fields, institutions — any country/system.
- licenses: professional licenses (RN, CA, CDL, teaching license, etc.) when stated.
- certifications: non-license certificates (PMP, Six Sigma, product certs, etc.).
- skills: short phrases relevant to the candidate's field (domain-appropriate; not tech-only).
- notice_period_days: immediate→0, 1 month→30, 2 weeks→14; null if unknown.
- If PRE-EXTRACTED FACTS are given, trust email/phone/years when marked high-confidence.
"""

# Compact few-shots (2 examples) — shorter prompts = faster generation on small models
EXTRACTION_FEW_SHOT = """
Example A: "Riya Sharma | riya@mail.com | Pune. B.Tech CSE VIT 2025. Intern FinServe Jun–Aug 2024. Skills: Python, SQL."
→ {"name":"Riya Sharma","email":"riya@mail.com","phone":null,"location":"Pune","total_years_experience":0.3,"education":[{"degree":"B.Tech","field":"CSE","institution":"VIT","year":"2025"}],"licenses":[],"certifications":[],"skills":["Python","SQL"],"work_history":[{"title":"Intern","company":"FinServe","duration":"Jun–Aug 2024","years":0.3}],"notice_period_days":0,"summary":"Fresher with short internship; Python/SQL."}

Example B: "Marcus Lee | marcus@email.com | Chicago. Regional Sales Manager NorthCo 2019–Present; Account Exec BrightSales 2015–2019. B.A. Marketing UIC. Salesforce, B2B."
→ {"name":"Marcus Lee","email":"marcus@email.com","phone":null,"location":"Chicago","total_years_experience":9,"education":[{"degree":"B.A.","field":"Marketing","institution":"UIC","year":null}],"licenses":[],"certifications":[],"skills":["B2B sales","Salesforce"],"work_history":[{"title":"Regional Sales Manager","company":"NorthCo","duration":"2019–Present","years":null},{"title":"Account Exec","company":"BrightSales","duration":"2015–2019","years":4}],"notice_period_days":null,"summary":"~9 years B2B sales leadership."}
"""


def _build_sectioned_prompt(
    resume_text: str,
    sections: Optional[dict[str, str]],
    hybrid: Optional[dict[str, Any]],
    *,
    max_chars: int = 14000,
) -> str:
    """Prefer sectioned content so experience/education aren't truncated away."""
    parts: list[str] = []
    if hybrid:
        # Industry-light hybrid: only high-trust identity / years / education hints
        # Skills come from the LLM + hiring brief, not a tech lexicon.
        allow = {
            "name",
            "email",
            "phone",
            "location",
            "total_years_experience",
            "education",
            "notice_period_days",
            "linkedin",
            "github",
        }
        hints = {
            k: v
            for k, v in hybrid.items()
            if k in allow and v not in (None, "", [], {})
        }
        parts.append("PRE-EXTRACTED FACTS (trust email/phone/years when present; fix name/location if wrong):")
        parts.append(json.dumps(hints, indent=2)[:2000])
        parts.append("")

    sections = sections or {}
    order = [
        ("header", "HEADER / CONTACT"),
        ("summary", "SUMMARY"),
        ("experience", "EXPERIENCE"),
        ("education", "EDUCATION"),
        ("skills", "SKILLS"),
        ("certifications", "CERTIFICATIONS"),
        ("projects", "PROJECTS"),
    ]
    used = 0
    budget = max_chars
    for key, label in order:
        body = (sections.get(key) or "").strip()
        if not body:
            continue
        chunk = body[: max(500, budget // 3)]
        block = f"### {label}\n{chunk}"
        parts.append(block)
        used += len(block)
        if used >= budget:
            break

    if used < 500:
        # Fallback to full text
        parts.append("### FULL RESUME TEXT")
        parts.append((resume_text or "")[:max_chars])
    elif sections.get("full") and used < budget * 0.6:
        # Append remaining body tail if sections were thin
        remaining = budget - used
        if remaining > 800:
            parts.append("### ADDITIONAL TEXT")
            parts.append((resume_text or "")[:remaining])

    return "\n\n".join(parts)


async def _chat(
    messages: list[dict[str, str]],
    *,
    model: Optional[str] = None,
    format_json: bool = True,
    temperature: float = 0.0,
) -> str:
    payload: dict[str, Any] = {
        "model": model or EXTRACT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if format_json:
        payload["format"] = "json"

    sem = get_llm_semaphore()
    async with sem:
        client = await get_http_client()
        r = await client.post("/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "") or ""


async def extract_structured(
    resume_text: str,
    *,
    max_retries: Optional[int] = None,
    sections: Optional[dict[str, str]] = None,
    hybrid: Optional[dict[str, Any]] = None,
    doc_meta: Optional[dict[str, Any]] = None,
    filename: Optional[str] = None,
    refine: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Hybrid + LLM structured extraction with validation and optional repair.

    1. Run / accept hybrid regex-dict facts
    2. Ask extract model for full JSON (sectioned prompt, temp 0)
    3. Merge hybrid-over-LLM for high-confidence fields
    4. Optionally repair / refine identity (extract_refine_enabled)

    Set refine=False (default via config) for bulk speed.
    """
    if max_retries is None:
        max_retries = EXTRACT_MAX_RETRIES
    if refine is None:
        refine = EXTRACT_REFINE_ENABLED

    sections = sections or extract_rules.split_sections(resume_text)
    if hybrid is None:
        hybrid = extract_rules.hybrid_extract(
            resume_text, sections, filename=filename
        )
    elif filename and not hybrid.get("_filename"):
        hybrid = dict(hybrid)
        hybrid["_filename"] = filename
    if filename:
        hybrid = dict(hybrid or {})
        hybrid["_filename"] = filename
        hybrid["_resume_text"] = (resume_text or "")[:2000]

    sectioned = _build_sectioned_prompt(resume_text, sections, hybrid, max_chars=10000)
    user_prompt = (
        f"{EXTRACTION_FEW_SHOT}\n\n"
        f"Extract structured data from this resume:\n\n{sectioned}"
    )
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    last_err: Optional[Exception] = None
    llm_data: Optional[dict[str, Any]] = None

    for attempt in range(max_retries + 1):
        try:
            content = await _chat(messages, format_json=True, temperature=0.0)
            llm_data = _extract_json_object(content)
            llm_data = extract_rules.sanitize_structured(llm_data)
            break
        except Exception as exc:
            last_err = exc
            logger.warning(
                "extract_structured attempt %s failed: %s", attempt + 1, exc
            )
            if attempt < max_retries:
                messages = [
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON. "
                            "Return ONLY a JSON object with the required keys. "
                            "Do not invent facts.\n\n"
                            f"{sectioned[:7000]}"
                        ),
                    },
                ]
                await asyncio.sleep(0.35 * (attempt + 1))

    if llm_data is None:
        # Fall back to hybrid-only so batch can still hard-filter on contact/years
        logger.error(
            "LLM extraction failed after retries (%s); using hybrid-only", last_err
        )
        merged = extract_rules.structured_from_hybrid(
            hybrid,
            resume_text=resume_text,
            doc_meta=doc_meta,
            filename=filename,
        )
        if last_err:
            meta = dict(merged.get("_extraction") or {})
            meta["validation_issues"] = [str(last_err)[:200]]
            merged["_extraction"] = meta
        return merged

    merged = extract_rules.merge_hybrid_and_llm(hybrid, llm_data)
    merged = extract_rules.apply_two_pass_years(
        merged, hybrid=hybrid, resume_text=resume_text
    )
    merged = extract_rules.sanitize_structured(merged)
    issues = extract_rules.validate_structured(merged)

    # Targeted repair + identity refine only when enabled (extra LLM calls)
    if refine and issues and max_retries > 0:
        try:
            repaired = await _repair_fields(
                merged, issues, sectioned[:6000], hybrid
            )
            if repaired:
                merged = extract_rules.merge_hybrid_and_llm(hybrid, repaired)
                merged = extract_rules.apply_two_pass_years(
                    merged, hybrid=hybrid, resume_text=resume_text
                )
                merged = extract_rules.sanitize_structured(merged)
                issues = extract_rules.validate_structured(merged)
        except Exception as exc:
            logger.warning("targeted repair failed: %s", exc)

    if refine:
        try:
            conf = merged.get("_confidence") or {}
            name_low = conf.get("name") == "low" or not merged.get("name")
            years_low = conf.get("total_years_experience") in ("low", None) and (
                not merged.get("work_history")
            )
            if name_low or years_low:
                refined = await _refine_identity_fields(
                    merged, resume_text[:5000], filename=filename
                )
                if refined:
                    if refined.get("name") and extract_rules._is_plausible_person_name(
                        refined["name"], allow_single=True
                    ):
                        merged["name"] = refined["name"]
                        conf = dict(merged.get("_confidence") or {})
                        conf["name"] = "medium"
                        merged["_confidence"] = conf
                        src = dict(merged.get("_sources") or {})
                        src["name"] = "llm_refine"
                        merged["_sources"] = src
                    if refined.get("total_years_experience") is not None and years_low:
                        try:
                            y = float(refined["total_years_experience"])
                            if 0 <= y <= 50:
                                merged["total_years_experience"] = y
                        except (TypeError, ValueError):
                            pass
                    merged = extract_rules.apply_two_pass_years(
                        merged, hybrid=hybrid, resume_text=resume_text
                    )
                    resolved = extract_rules.resolve_candidate_name(
                        llm_name=merged.get("name"),
                        hybrid_name=hybrid.get("name"),
                        email=merged.get("email") or hybrid.get("email"),
                        filename=filename or hybrid.get("_filename"),
                        resume_text=resume_text,
                    )
                    if resolved.get("name"):
                        merged["name"] = resolved["name"]
                        conf = dict(merged.get("_confidence") or {})
                        conf["name"] = resolved.get("confidence") or "medium"
                        merged["_confidence"] = conf
        except Exception as exc:
            logger.warning("identity refine pass failed: %s", exc)

    return _attach_meta(merged, hybrid, doc_meta, llm_used=True, issues=issues)


async def _refine_identity_fields(
    current: dict[str, Any],
    resume_excerpt: str,
    *,
    filename: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Focused LLM pass for name + professional years only."""
    messages = [
        {
            "role": "system",
            "content": (
                "Extract only identity fields from the resume. Return JSON: "
                '{"name": string|null, "total_years_experience": number}. '
                "name must be a real person's name (not skills or job titles). "
                "total_years_experience is professional work only (0 for students/freshers)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Filename hint: {filename or 'n/a'}\n"
                f"Current guess name: {current.get('name')!r}\n"
                f"Email: {current.get('email')!r}\n\n"
                f"Resume:\n{resume_excerpt}"
            ),
        },
    ]
    content = await _chat(messages, format_json=True, temperature=0.0)
    return _extract_json_object(content)


async def _repair_fields(
    current: dict[str, Any],
    issues: list[str],
    resume_excerpt: str,
    hybrid: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Ask the model to fix only invalid fields."""
    prompt = {
        "current": {
            k: current.get(k)
            for k in (
                "name",
                "email",
                "phone",
                "location",
                "total_years_experience",
                "education",
                "certifications",
                "skills",
                "work_history",
                "notice_period_days",
                "summary",
            )
        },
        "issues": issues,
        "hybrid_hints": {
            k: hybrid.get(k)
            for k in (
                "name",
                "email",
                "phone",
                "total_years_experience",
                "skills",
                "notice_period_days",
            )
            if hybrid.get(k) not in (None, "", [])
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Fix the resume JSON. Return ONLY the full corrected JSON object "
                "with the same keys. Resolve the listed issues using the resume text. "
                "Do not invent data."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Issues: {json.dumps(issues)}\n"
                f"Current + hints: {json.dumps(prompt)[:3500]}\n\n"
                f"Resume excerpt:\n{resume_excerpt}"
            ),
        },
    ]
    content = await _chat(messages, format_json=True, temperature=0.0)
    data = _extract_json_object(content)
    return extract_rules.sanitize_structured(data)


def _attach_meta(
    data: dict[str, Any],
    hybrid: dict[str, Any],
    doc_meta: Optional[dict[str, Any]],
    *,
    llm_used: bool,
    issues: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Attach extraction quality metadata under _extraction (filters ignore _ keys)."""
    conf = data.get("_confidence") or hybrid.get("_confidence") or {}
    sources = data.get("_sources") or hybrid.get("_sources") or {}
    section_hits = data.get("_section_hits") or hybrid.get("_section_hits") or []
    doc_meta = doc_meta or {}

    # Overall extraction confidence for UI
    high_fields = sum(1 for v in conf.values() if v == "high")
    quality = doc_meta.get("quality") or "medium"
    if not llm_used:
        overall = "low"
    elif quality == "low" or (issues and len(issues) > 2):
        overall = "low"
    elif high_fields >= 3 and quality == "high":
        overall = "high"
    else:
        overall = "medium"

    # Production trust flags for HR UI
    trust_warnings: list[str] = []
    name = data.get("name")
    loc = data.get("location")
    try:
        from extract_rules import (
            _is_plausible_person_name,
            _is_plausible_location,
            _looks_like_tech_or_skills_blob,
        )

        if not name:
            trust_warnings.append(
                "Name missing or rejected — verify candidate identity from the file name or resume."
            )
            conf["name"] = "low"
        elif not _is_plausible_person_name(name, allow_single=True):
            trust_warnings.append(
                f"Name looks unreliable ({name!r}) — may be a section header or skills line."
            )
            conf["name"] = "low"
        if loc and not _is_plausible_location(loc):
            trust_warnings.append(
                f"Location looks unreliable ({loc!r}) — may be skills/tools misread."
            )
            conf["location"] = "low"
            data["location"] = None
        if conf.get("total_years_experience") == "low":
            trust_warnings.append("Years of experience is low-confidence — verify manually.")
    except Exception:
        pass

    if issues:
        trust_warnings.extend(f"Validation: {i}" for i in issues[:5])

    data["_extraction"] = {
        "overall": overall if not trust_warnings else (
            "low" if any("unreliable" in w or "missing" in w for w in trust_warnings) else overall
        ),
        "llm_used": llm_used,
        "ocr_used": bool(doc_meta.get("ocr_used")),
        "method": doc_meta.get("method"),
        "text_chars": doc_meta.get("text_chars"),
        "text_quality": quality,
        "section_hits": section_hits,
        "multi_column": bool(doc_meta.get("multi_column")),
        "multi_col_pages": doc_meta.get("multi_col_pages") or 0,
        "years_method": data.get("_years_method"),
        "field_confidence": conf,
        "field_sources": sources,
        "validation_issues": issues or [],
        "trust_warnings": trust_warnings,
    }
    data["_confidence"] = conf
    data["_sources"] = sources
    return data


def _normalize_extraction(data: dict[str, Any]) -> dict[str, Any]:
    """Legacy helper — sanitize only."""
    return extract_rules.sanitize_structured(data)


SCORING_SYSTEM = """You are an expert technical recruiter.
The hiring manager wrote a single hiring brief (instructions for who to hire).
Score how well this candidate's resume matches THAT brief only.
Return ONLY a valid JSON object:
{
  "score": number from 1 to 10 (decimals allowed, e.g. 7.5),
  "justification": string (1-2 sentences citing concrete evidence from the resume vs the brief)
}
Rules:
- Follow the hiring brief as the source of truth (skills, experience, education, location, seniority, etc.).
- Reward clear evidence in the resume; penalize missing or contradictory signals.
- Do not invent resume facts. Be fair and specific.
"""


async def score_fit(
    resume_structured: dict[str, Any],
    resume_text: str,
    criteria: dict[str, Any],
    *,
    max_retries: int = 0,
) -> dict[str, Any]:
    """LLM fit score 1-10 + short justification against HR hiring prompt only."""
    hiring_prompt = (
        (criteria.get("hiring_prompt") or "").strip()
        or (criteria.get("job_description") or "").strip()
    )
    # Compact profile — smaller prompts = faster generation
    public = {
        k: v
        for k, v in (resume_structured or {}).items()
        if not str(k).startswith("_")
        and k
        in (
            "name",
            "location",
            "total_years_experience",
            "skills",
            "education",
            "work_history",
            "licenses",
            "certifications",
            "summary",
        )
    }
    # Trim long work_history
    wh = public.get("work_history") or []
    if isinstance(wh, list) and len(wh) > 6:
        public["work_history"] = wh[:6]
    candidate_blob = json.dumps(public, separators=(",", ":"))[:2500]
    text_snip = (resume_text or "")[:2800]
    user = (
        f"HIRING BRIEF:\n{hiring_prompt[:2500]}\n\n"
        f"PROFILE:\n{candidate_blob}\n\n"
        f"RESUME EXCERPT:\n{text_snip}\n\n"
        "Score 1-10 for fit. JSON: score + justification (1-2 sentences)."
    )

    messages = [
        {"role": "system", "content": SCORING_SYSTEM},
        {"role": "user", "content": user},
    ]
    score_retries = max(0, int(max_retries))
    last_err: Optional[Exception] = None
    for attempt in range(score_retries + 1):
        try:
            # Prefer dedicated score_model; fall back to extract model if unset
            content = await _chat(
                messages,
                model=SCORE_MODEL or EXTRACT_MODEL,
                format_json=True,
                temperature=0.1,
            )
            data = _extract_json_object(content)
            score = float(data.get("score", 5))
            score = max(1.0, min(10.0, score))
            just = str(data.get("justification") or "").strip() or "No justification provided."
            return {"score": score, "justification": just}
        except Exception as exc:
            last_err = exc
            logger.warning("score_fit attempt %s failed: %s", attempt + 1, exc)
            if attempt < score_retries:
                await asyncio.sleep(0.25 * (attempt + 1))
            # If stronger score model missing, retry once with extract model
            if attempt == 0 and SCORE_MODEL != EXTRACT_MODEL:
                try:
                    content = await _chat(
                        messages,
                        model=EXTRACT_MODEL,
                        format_json=True,
                        temperature=0.1,
                    )
                    data = _extract_json_object(content)
                    score = float(data.get("score", 5))
                    score = max(1.0, min(10.0, score))
                    just = (
                        str(data.get("justification") or "").strip()
                        or "No justification provided."
                    )
                    logger.warning(
                        "score_fit fell back to extract_model %s", EXTRACT_MODEL
                    )
                    return {"score": score, "justification": just}
                except Exception as exc2:
                    last_err = exc2
    raise RuntimeError(f"LLM scoring failed after retries: {last_err}")


async def embed(text: str, *, use_cache: bool = True) -> list[float]:
    """Get embedding vector from nomic-embed-text (SQLite-cached when enabled)."""
    import hashlib

    import db
    from config import get_settings

    text = (text or "").strip()
    if not text:
        return []
    text = text[:8000]
    settings = get_settings()
    cache_on = use_cache and bool(settings.get("embed_cache_enabled", True))
    cache_key = ""
    if cache_on:
        cache_key = hashlib.sha256(
            f"{EMBED_MODEL}:{text}".encode("utf-8")
        ).hexdigest()
        cached = db.get_embedding_cache(cache_key)
        if cached:
            return cached

    sem = get_embed_semaphore()
    async with sem:
        client = await get_http_client()
        r = await client.post(
            "/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        r.raise_for_status()
        emb = list(r.json().get("embedding") or [])
    if cache_on and emb and cache_key:
        try:
            db.set_embedding_cache(cache_key, EMBED_MODEL, emb)
        except Exception as exc:
            logger.warning("embed cache write failed: %s", exc)
    return emb


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts concurrently under the embed semaphore."""
    async def _one(t: str) -> list[float]:
        try:
            return await embed(t)
        except Exception as exc:
            logger.warning("embed failed: %s", exc)
            return []

    return list(await asyncio.gather(*[_one(t) for t in texts]))
