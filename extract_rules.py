"""Deterministic hybrid extraction: regex + dictionaries (no LLM).

Used to ground LLM extraction — contact fields, skills, dates, notice period,
and years estimates from work-history text.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
# International + US-style phones (loose)
PHONE_RE = re.compile(
    r"(?:(?:\+|00)\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-.]?)?\d{3,5}[\s\-.]?\d{3,5}(?:[\s\-.]?\d{2,5})?"
)
# Prefer phones with enough digits
PHONE_MIN_DIGITS = 10

LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_%/]+", re.I
)
GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9\-]+/?", re.I
)

# "5+ years", "8 years of experience", "Experience: 6 yrs"
YEARS_EXPLICIT_RE = re.compile(
    r"(?i)(?:(?:over|about|approx(?:imately)?|around)\s+)?"
    r"(\d{1,2}(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+experience)?"
)
YEARS_EXP_PHRASE_RE = re.compile(
    r"(?i)(?:total\s+)?(?:experience|exp\.?)\s*[:=\-]?\s*(\d{1,2}(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)?"
)

# Date ranges in work history: 2020-2024, Jan 2019 – Present, 2018 to 2021
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_RANGE_RE = re.compile(
    rf"(?i)(?:{_MONTH}\s+)?((?:19|20)\d{{2}})"
    rf"\s*(?:[-–—]|to)\s*"
    rf"(?:{_MONTH}\s+)?((?:19|20)\d{{2}}|Present|Current|Now|Till\s+Date|Ongoing)"
)

NOTICE_RE = re.compile(
    r"(?i)notice\s*period\s*[:=\-]?\s*"
    r"(immediate|serving\s+notice|"
    r"\d+\s*(?:days?|weeks?|months?)|"
    r"\d+\s*-\s*\d+\s*(?:days?|weeks?|months?))"
)
NOTICE_IMMEDIATE_RE = re.compile(r"(?i)\b(?:immediate(?:ly)?\s+available|available\s+immediate(?:ly)?)\b")

LOCATION_LINE_RE = re.compile(
    r"(?i)\b(?:based\s+in|location|relocating\s+to|residing\s+in)\s*[:=\-]?\s*([A-Za-z .,\-/]{3,60})"
)

# Common city, ST or City, Country patterns near top of resume
CITY_STATE_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*),\s*"
    r"([A-Z]{2}|[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b"
)

SECTION_HEADERS = {
    "experience": re.compile(
        r"(?im)^(?:professional\s+)?(?:work\s+)?experience|"
        r"^employment(?:\s+history)?|^work\s+history|^career\s+history|"
        r"^professional\s+background|^positions?\s+held\s*$"
    ),
    "education": re.compile(
        r"(?im)^education|^academic\s+(?:background|qualifications?)|"
        r"^qualifications?$|^degrees?\s*$"
    ),
    "skills": re.compile(
        r"(?im)^(?:technical\s+)?skills|^core\s+competenc(?:y|ies)|"
        r"^technologies|^tech\s+stack|^key\s+skills|^expertise\s*$"
    ),
    "certifications": re.compile(
        r"(?im)^certifications?|^licen[cs]es?(?:\s+and\s+certifications?)?|"
        r"^professional\s+certifications?"
    ),
    "summary": re.compile(
        r"(?im)^(?:professional\s+)?summary|^profile$|^objective|"
        r"^about\s+me|^career\s+objective|^professional\s+profile"
    ),
    "projects": re.compile(
        r"(?im)^projects?$|^key\s+projects?|^personal\s+projects?"
    ),
}

DEGREE_PATTERNS = [
    (re.compile(r"\b(?:B\.?Tech|BTech|Bachelor of Technology)\b", re.I), "B.Tech"),
    (re.compile(r"\b(?:B\.?E\.?|Bachelor of Engineering)\b", re.I), "B.E."),
    (re.compile(r"\b(?:B\.?S\.?|B\.?Sc\.?|Bachelor of Science)\b", re.I), "B.S."),
    (re.compile(r"\b(?:B\.?A\.?|Bachelor of Arts)\b", re.I), "B.A."),
    (re.compile(r"\b(?:M\.?Tech|MTech|Master of Technology)\b", re.I), "M.Tech"),
    (re.compile(r"\b(?:M\.?S\.?|M\.?Sc\.?|Master of Science)\b", re.I), "M.S."),
    (re.compile(r"\b(?:M\.?B\.?A\.?|Master of Business Administration)\b", re.I), "MBA"),
    (re.compile(r"\b(?:M\.?C\.?A\.?)\b", re.I), "MCA"),
    (re.compile(r"\b(?:Ph\.?D\.?|Doctor of Philosophy)\b", re.I), "Ph.D."),
    (re.compile(r"\b(?:B\.?Com|Bachelor of Commerce)\b", re.I), "B.Com"),
]

CERT_KEYWORDS = [
    "AWS Solutions Architect",
    "AWS Developer",
    "AWS SysOps",
    "AWS Certified",
    "Azure Administrator",
    "Azure Developer",
    "Azure Solutions Architect",
    "Google Cloud Professional",
    "GCP Professional",
    "CKA",
    "CKAD",
    "Certified Kubernetes Administrator",
    "Certified Kubernetes Application Developer",
    "PMP",
    "Project Management Professional",
    "CSM",
    "Certified ScrumMaster",
    "PSM",
    "ITIL",
    "CISSP",
    "CompTIA Security+",
    "Oracle Certified",
    "Terraform Associate",
    "HashiCorp Certified",
    "Databricks",
    "Snowflake",
]

# Skills lexicon — matched case-insensitively as whole tokens / phrases
SKILL_LEXICON = [
    # Languages
    "Python", "Java", "JavaScript", "TypeScript", "Go", "Golang", "Rust", "C++",
    "C#", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB", "SQL",
    "Bash", "Shell",
    # Web / backend
    "React", "Angular", "Vue", "Node.js", "NodeJS", "Express", "Django", "Flask",
    "FastAPI", "Spring", "Spring Boot", "ASP.NET", ".NET", "Next.js", "GraphQL",
    "REST", "gRPC", "HTML", "CSS", "Svelte",
    # Data / ML
    "pandas", "NumPy", "scikit-learn", "TensorFlow", "PyTorch", "Keras",
    "Spark", "Hadoop", "Airflow", "dbt", "Tableau", "Power BI", "Kafka",
    "ETL", "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes", "K8s",
    "Terraform", "Ansible", "Jenkins", "CI/CD", "GitHub Actions", "GitLab CI",
    "Linux", "Prometheus", "Grafana", "Helm", "Lambda", "EC2", "S3",
    "CloudFormation", "Pulumi",
    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "DynamoDB",
    "Cassandra", "Oracle", "SQL Server", "SQLite", "Neo4j",
    # Soft / process
    "Agile", "Scrum", "leadership", "mentoring", "system design",
    "microservices", "TDD", "Jira",
]

# Institutions (lightweight signal for education lines)
INSTITUTION_HINTS = re.compile(
    r"(?i)\b(?:University|College|Institute|IIT|NIT|IIIT|MIT|Stanford|"
    r"Berkeley|Harvard|Oxford|Cambridge|BITS|VIT|SRM|Amity)\b"
)

def _current_year() -> int:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).year


# Kept as callable-friendly module default (tests may patch _current_year)
CURRENT_YEAR = _current_year()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_sections(text: str) -> dict[str, str]:
    """
    Split resume text into named sections by common headers.
    Always includes 'header' (content before first known section) and 'full'.
    Preserves inline content after headers (e.g. "Skills: Python, React").
    """
    text = text or ""
    lines = text.split("\n")
    # markers: (line_index, section_name, inline_remainder_after_header)
    markers: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 100:
            continue
        for name, pat in SECTION_HEADERS.items():
            m = pat.search(stripped)
            if not m:
                continue
            if not _looks_like_header(stripped, m):
                continue
            # Remainder on same line after "Skills:" / "Experience -" etc.
            rest = stripped[m.end() :].lstrip(" \t:-–—|")
            markers.append((i, name, rest))
            break

    sections: dict[str, str] = {"full": text}
    if not markers:
        sections["header"] = "\n".join(lines[:25])
        sections["body"] = text
        return sections

    first_idx = markers[0][0]
    sections["header"] = "\n".join(lines[:first_idx]).strip()

    for j, (start, name, inline) in enumerate(markers):
        end = markers[j + 1][0] if j + 1 < len(markers) else len(lines)
        body_lines = lines[start + 1 : end]
        chunk_parts = []
        if inline:
            chunk_parts.append(inline)
        chunk_parts.extend(body_lines)
        chunk = "\n".join(chunk_parts).strip()
        if name in sections and sections[name]:
            sections[name] = sections[name] + "\n" + chunk
        else:
            sections[name] = chunk

    return sections


def _looks_like_header(line: str, match: re.Match | None = None) -> bool:
    """True if line is a section title, or a title with short inline content."""
    # "Skills: Python, React, HTML" — header prefix + list OK
    if match is not None:
        prefix = line[: match.end()].strip()
        rest = line[match.end() :].lstrip(" \t:-–—|")
        # Prefix should be short (the heading itself)
        if len(prefix.split()) > 6:
            return False
        if rest and len(rest) > 120:
            # Long narrative starting with the word "experience" — not a header
            if not re.match(r"(?i)^(skills?|technologies|certifications?)\b", prefix):
                return False
        return True

    words = line.split()
    if len(words) > 8:
        return False
    if line.endswith(".") and len(words) > 3:
        return False
    return True


def hybrid_extract(
    text: str,
    sections: Optional[dict[str, str]] = None,
    *,
    filename: Optional[str] = None,
) -> dict[str, Any]:
    """
    Industry-light deterministic extract (no tech skill lexicon).

    Hybrid owns only high-trust / structural fields:
      name, email, phone, location, years, education, notice.
    Skills / licenses / rich certs come from the LLM + hiring brief.
    """
    sections = sections or split_sections(text)
    header = sections.get("header") or "\n".join((text or "").split("\n")[:20])
    exp_sec = sections.get("experience") or ""
    edu_sec = sections.get("education") or ""
    full = text or ""

    conf: dict[str, str] = {}
    sources: dict[str, str] = {}
    trust_warnings: list[str] = []

    email = _first_email(header) or _first_email(full)
    phone = _first_phone(header) or _first_phone(full)
    linkedin = _first_match(LINKEDIN_RE, full)
    github = _first_match(GITHUB_RE, full)
    resolved = resolve_candidate_name(
        llm_name=None,
        hybrid_name=_guess_name(header, email, full_text=full),
        email=email,
        filename=filename,
        resume_text=full,
    )
    name = resolved.get("name")
    if resolved.get("warning"):
        trust_warnings.append(resolved["warning"])
    if name:
        conf["name"] = resolved.get("confidence") or "medium"
        sources["name"] = resolved.get("source") or "heuristic"
    location = _guess_location(header, full)
    if location and not _is_plausible_location(location):
        trust_warnings.append(f"Rejected non-location text as location: {location!r}")
        location = None
    notice = _parse_notice_period(full)
    # Education only (no tech skill/cert lexicons — domain-agnostic)
    education = _parse_education(edu_sec if edu_sec else full)
    years_explicit = _explicit_years(full)
    work_years_est, date_ranges = _estimate_years_from_dates(exp_sec or "")

    total_years: Optional[float] = None
    if years_explicit is not None:
        total_years = years_explicit
        conf["total_years_experience"] = "high"
        sources["total_years_experience"] = "regex"
    elif work_years_est is not None and exp_sec:
        total_years = work_years_est
        conf["total_years_experience"] = "medium"
        sources["total_years_experience"] = "date_range"

    result: dict[str, Any] = {
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
        "linkedin": linkedin,
        "github": github,
        "total_years_experience": total_years,
        "education": education,
        # Skills intentionally empty here — LLM fills from resume text
        "skills": [],
        "certifications": [],
        "licenses": [],
        "work_history_hints": date_ranges[:12],
        "notice_period_days": notice,
        "summary": None,
        "_confidence": conf,
        "_sources": sources,
        "_trust_warnings": trust_warnings,
        "_filename": filename,
        "_resume_text": (full or "")[:2000],
        "_section_hits": [
            k for k in sections if k not in ("full", "body") and sections.get(k)
        ],
    }

    if email:
        conf["email"] = "high"
        sources["email"] = "regex"
    if phone:
        conf["phone"] = "high"
        sources["phone"] = "regex"
    if not name:
        conf["name"] = "low"
        sources["name"] = "missing"
        trust_warnings.append(
            "Name not found yet — will try LLM + email + filename after parse."
        )
    if location:
        conf["location"] = "medium"
        sources["location"] = "heuristic"
    if education:
        conf["education"] = "medium"
        sources["education"] = "heuristic"
    if notice is not None:
        conf["notice_period_days"] = "high"
        sources["notice_period_days"] = "regex"

    result["_confidence"] = conf
    result["_sources"] = sources
    result["_trust_warnings"] = trust_warnings
    return result


def structured_from_hybrid(
    hybrid: dict[str, Any],
    *,
    resume_text: str = "",
    doc_meta: Optional[dict[str, Any]] = None,
    filename: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build a scorable structured profile from hybrid-only extract (no LLM).

    Used in fast bulk phase-1: ATS + embeddings still work via raw resume text.
    Skills/work_history stay thin until a later LLM extract on top-K candidates.
    """
    doc_meta = doc_meta or {}
    hybrid = dict(hybrid or {})
    if filename:
        hybrid["_filename"] = filename

    data = sanitize_structured(
        {
            "name": hybrid.get("name"),
            "email": hybrid.get("email"),
            "phone": hybrid.get("phone"),
            "location": hybrid.get("location"),
            "total_years_experience": hybrid.get("total_years_experience") or 0,
            "education": hybrid.get("education") or [],
            "licenses": hybrid.get("licenses") or [],
            "certifications": hybrid.get("certifications") or [],
            "skills": hybrid.get("skills") or [],
            "work_history": [],
            "notice_period_days": hybrid.get("notice_period_days"),
            "summary": "",
        }
    )
    data = apply_two_pass_years(data, hybrid=hybrid, resume_text=resume_text)
    data = sanitize_structured(data)

    conf = dict(hybrid.get("_confidence") or {})
    sources = dict(hybrid.get("_sources") or {})
    trust = list(hybrid.get("_trust_warnings") or [])
    section_hits = list(hybrid.get("_section_hits") or [])
    quality = doc_meta.get("quality") or "medium"
    overall = "medium" if data.get("name") or data.get("email") else "low"
    if quality == "low":
        overall = "low"

    data["_confidence"] = conf
    data["_sources"] = sources
    data["_extraction"] = {
        "overall": overall,
        "llm_used": False,
        "hybrid_only": True,
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
        "validation_issues": [],
        "trust_warnings": trust,
    }
    return data


def merge_hybrid_and_llm(
    hybrid: dict[str, Any],
    llm: dict[str, Any],
) -> dict[str, Any]:
    """
    Prefer hybrid for high-confidence contact/skills; use LLM for narrative fields.
    Always keep internal _meta keys for UI/debugging.
    """
    conf = dict(hybrid.get("_confidence") or {})
    sources = dict(hybrid.get("_sources") or {})
    out: dict[str, Any] = dict(llm or {})

    def prefer(field: str, min_conf: str = "medium") -> None:
        hval = hybrid.get(field)
        if hval in (None, "", [], {}):
            return
        rank = {"high": 3, "medium": 2, "low": 1}
        if rank.get(conf.get(field, ""), 0) >= rank.get(min_conf, 2):
            out[field] = hval
            sources[field] = sources.get(field, "hybrid")

    # Contact — hybrid wins when present (location validated below)
    prefer("email", "medium")
    prefer("phone", "medium")

    # Location: reject skill blobs like "JavaScript, HTML"
    h_loc = hybrid.get("location")
    l_loc = out.get("location")
    if _is_plausible_location(l_loc):
        out["location"] = str(l_loc).strip()
        sources["location"] = "llm"
        conf["location"] = "medium"
    elif _is_plausible_location(h_loc):
        out["location"] = str(h_loc).strip()
        sources["location"] = "heuristic"
        conf["location"] = "medium"
    else:
        out["location"] = None
        sources["location"] = "rejected_invalid"
        conf.pop("location", None)

    # Name: multi-source resolution (filename + email + LLM + heuristic)
    # 100% on messy PDFs is impossible; consensus + fail-closed wrong names ≈ best practical accuracy.
    resolved = resolve_candidate_name(
        llm_name=out.get("name"),
        hybrid_name=hybrid.get("name"),
        email=out.get("email") or hybrid.get("email"),
        filename=hybrid.get("_filename") or out.get("_filename"),
        resume_text=hybrid.get("_resume_text") or "",
    )
    out["name"] = resolved.get("name")
    sources["name"] = resolved.get("source") or "none"
    conf["name"] = resolved.get("confidence") or "low"
    if resolved.get("warning"):
        tw = list(hybrid.get("_trust_warnings") or out.get("_trust_warnings") or [])
        tw.append(resolved["warning"])
        out["_trust_warnings"] = tw
    if resolved.get("name") is None:
        conf["name"] = "low"

    # Skills / licenses / certs: LLM only (no tech lexicon merge)
    l_skills = out.get("skills") or []
    if isinstance(l_skills, str):
        l_skills = [s.strip() for s in l_skills.split(",") if s.strip()]
    if isinstance(l_skills, list):
        out["skills"] = [str(s).strip() for s in l_skills if str(s).strip()][:60]
        sources["skills"] = "llm"
    else:
        out["skills"] = []

    for list_key in ("certifications", "licenses"):
        val = out.get(list_key) or []
        if isinstance(val, str):
            val = [x.strip() for x in val.split(",") if x.strip()]
        if not isinstance(val, list):
            val = []
        out[list_key] = [str(x).strip() for x in val if str(x).strip()][:40]
        sources[list_key] = "llm"

    # Notice period: hybrid wins
    if hybrid.get("notice_period_days") is not None:
        out["notice_period_days"] = hybrid["notice_period_days"]

    # Education: if hybrid found degrees and LLM empty, use hybrid
    if hybrid.get("education") and not out.get("education"):
        out["education"] = hybrid["education"]

    # Optional social links
    if hybrid.get("linkedin"):
        out["linkedin"] = hybrid["linkedin"]
    if hybrid.get("github"):
        out["github"] = hybrid["github"]

    out["_confidence"] = conf
    out["_sources"] = sources
    out["_section_hits"] = hybrid.get("_section_hits") or []

    # Two-pass years: fill job durations then compute from work_history (code, not LLM math)
    out = apply_two_pass_years(out, hybrid=hybrid)
    conf = dict(out.get("_confidence") or conf)
    sources = dict(out.get("_sources") or sources)
    out["_confidence"] = conf
    out["_sources"] = sources
    return out


# ---------------------------------------------------------------------------
# Two-pass years (code-side, after work_history is known)
# ---------------------------------------------------------------------------


_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _month_num(token: str) -> Optional[int]:
    return _MONTH_MAP.get(str(token).lower().strip("."))


def _parse_year_bounds(duration: str) -> Optional[tuple[int, int]]:
    """
    Return (start_year, end_year_clamped_to_now) or None.
    Future end years (e.g. graduation 2027) are clamped to the current year.
    """
    if not duration:
        return None
    now = _current_year()
    m = DATE_RANGE_RE.search(str(duration))
    if not m:
        return None
    try:
        start = int(m.group(1))
    except ValueError:
        return None
    end_s = m.group(2)
    end_l = end_s.lower()
    if any(x in end_l for x in ("present", "current", "now", "ongoing", "till")):
        end = now
    else:
        ym = re.search(r"(?:19|20)\d{2}", end_s)
        if not ym:
            return None
        end = int(ym.group())
    if start < 1970 or start > now:
        return None
    end = min(end, now)
    if end < start:
        return None
    return start, end


def _parse_month_range_years(duration: str) -> Optional[float]:
    """
    Parse ranges like 'Jul 2025 – Nov 2025', '01/2020 - 03/2022', 'Mar 2019 – Present'.
    Returns fractional years.
    """
    if not duration:
        return None
    now = _current_year()
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc)
    cur_y, cur_m = today.year, today.month

    # Mon YYYY … Mon YYYY | Present
    m = re.search(
        r"(?i)\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s*((?:19|20)\d{2})\s*[-–—to]+\s*"
        r"(?:(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s*)?((?:19|20)\d{2}|present|current|now|ongoing)\b",
        str(duration),
    )
    if m:
        sm = _month_num(m.group(1)) or 1
        sy = int(m.group(2))
        end_raw = m.group(4).lower()
        if end_raw in ("present", "current", "now", "ongoing"):
            em, ey = cur_m, cur_y
        else:
            em = _month_num(m.group(3) or "jan") or 12
            ey = int(end_raw)
            if ey > cur_y or (ey == cur_y and em > cur_m):
                em, ey = cur_m, cur_y
        if sy > cur_y:
            return None
        months = (ey - sy) * 12 + (em - sm)
        if months < 0:
            return None
        return round(min(months / 12.0, 40.0), 2)

    # MM/YYYY - MM/YYYY
    m2 = re.search(
        r"\b(0?[1-9]|1[0-2])[/\-]((?:19|20)\d{2})\s*[-–—to]+\s*"
        r"(?:(0?[1-9]|1[0-2])[/\-])?((?:19|20)\d{2}|present|current)\b",
        str(duration),
        re.I,
    )
    if m2:
        sm, sy = int(m2.group(1)), int(m2.group(2))
        end_raw = m2.group(4).lower()
        if end_raw in ("present", "current"):
            em, ey = cur_m, cur_y
        else:
            em = int(m2.group(3) or 12)
            ey = int(end_raw)
            if ey > cur_y or (ey == cur_y and em > cur_m):
                em, ey = cur_m, cur_y
        if sy > cur_y:
            return None
        months = (ey - sy) * 12 + (em - sm)
        if months < 0:
            return None
        return round(min(months / 12.0, 40.0), 2)
    return None


def years_from_duration_string(duration: str) -> Optional[float]:
    """Parse duration strings into years (supports month-level ranges)."""
    if not duration:
        return None
    # Prefer month-accurate parse when months present
    if re.search(
        r"(?i)jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|/",
        str(duration),
    ):
        m_yrs = _parse_month_range_years(duration)
        if m_yrs is not None:
            return m_yrs

    bounds = _parse_year_bounds(duration)
    if bounds is not None:
        start, end = bounds
        yrs = float(end - start)
        if yrs == 0:
            # same year without month info — treat as ~half year if "Present" or range dash
            if re.search(r"(?i)present|current|–|—|-", str(duration)):
                return 0.5
        return round(min(yrs, 40.0), 1)

    m2 = re.search(
        r"(?i)(\d+(?:\.\d+)?)\s*(years?|yrs?|months?|mos?)", str(duration)
    )
    if m2:
        n = float(m2.group(1))
        unit = m2.group(2).lower()
        if unit.startswith("month") or unit.startswith("mo"):
            return round(n / 12.0, 1)
        return n
    return None


def _is_real_job_entry(w: dict[str, Any]) -> bool:
    """
    True only for entries that look like employment (not bare date ranges).
    Empty title+company rows are often degree / project dates mis-tagged as jobs.
    """
    title = str(w.get("title") or "").strip()
    company = str(w.get("company") or "").strip()
    if not title and not company:
        return False
    junk = {"", "-", "n/a", "na", "none", "null", ".", "—", "–"}
    if title.lower() in junk and company.lower() in junk:
        return False
    # Titles that are really education
    blob = f"{title} {company}".lower()
    if any(
        k in blob
        for k in (
            "bachelor",
            "master",
            "b.tech",
            "btech",
            "m.tech",
            "university",
            "institute of",
            "high school",
            "class xii",
            "class 12",
        )
    ):
        return False
    return True


def _looks_like_student(data: dict[str, Any], resume_text: str = "") -> bool:
    """Detect student / fresher profiles with little or no full-time work."""
    text = f"{resume_text or ''} {data.get('summary') or ''}".lower()
    student_cues = (
        "student",
        "undergrad",
        "undergraduate",
        "graduating",
        "pursuing",
        "cgpa",
        "gpa",
        "batch of",
        "expected graduation",
        "fresher",
        "final year",
        "pre-final",
    )
    exp_cues = (
        "years of experience",
        "years experience",
        "yrs of experience",
        "professional experience of",
    )
    has_student = any(c in text for c in student_cues)
    has_exp_claim = any(c in text for c in exp_cues)
    return has_student and not has_exp_claim


def enrich_work_history_years(work: list[Any]) -> list[dict[str, Any]]:
    """Pass A: fill years from duration; zero out non-jobs and invalid ranges."""
    out: list[dict[str, Any]] = []
    for w in work or []:
        if not isinstance(w, dict):
            continue
        item = dict(w)
        if not _is_real_job_entry(item):
            # Keep row for display but do not count as experience
            item["years"] = 0.0
            item["_count_as_experience"] = False
            out.append(item)
            continue
        item["_count_as_experience"] = True
        y = years_from_duration_string(str(item.get("duration") or ""))
        if y is None:
            try:
                if item.get("years") not in (None, ""):
                    y = float(item["years"])
            except (TypeError, ValueError):
                y = None
        # Recompute from dates even if LLM put a large years value
        if y is not None:
            item["years"] = max(0.0, float(y))
        else:
            item["years"] = 0.0
        out.append(item)
    return out


def compute_years_from_work_history(
    work: list[Any],
) -> tuple[Optional[float], str]:
    """
    Pass B: compute career years from *real* job entries only.
    Ignores empty title/company date rows (common education mis-tags).
    """
    work = enrich_work_history_years(work)
    jobs = [w for w in work if w.get("_count_as_experience")]
    if not jobs:
        return None, "none"

    sum_years = 0.0
    intervals: list[tuple[int, int]] = []
    for w in jobs:
        try:
            fy = float(w.get("years") or 0)
        except (TypeError, ValueError):
            fy = 0.0
        if fy > 0:
            sum_years += fy
        bounds = _parse_year_bounds(str(w.get("duration") or ""))
        if bounds:
            intervals.append(bounds)

    if sum_years <= 0 and not intervals:
        return 0.0, "no_countable_jobs"

    if intervals:
        earliest = min(s for s, _ in intervals)
        latest = max(e for _, e in intervals)
        span = float(max(0, latest - earliest))
        if sum_years > 0 and span > 0 and sum_years > span * 1.25:
            return round(min(span, 50.0), 1), "work_history_span"
        if sum_years > 0:
            cap = span if span > 0 else sum_years
            return round(min(sum_years, cap, 50.0), 1), "work_history_sum"
        if span > 0:
            return round(min(span, 50.0), 1), "work_history_span"

    if sum_years > 0:
        return round(min(sum_years, 50.0), 1), "work_history_sum"
    return 0.0, "no_countable_jobs"


def apply_two_pass_years(
    data: dict[str, Any],
    *,
    hybrid: Optional[dict[str, Any]] = None,
    resume_text: str = "",
) -> dict[str, Any]:
    """
    Two-pass years of experience:
      1) Keep only real jobs (title or company present)
      2) Compute years from those jobs; clamp future dates
      3) Students / freshers without real jobs → 0 (not degree length)

    Recruiter overrides always win.
    """
    out = dict(data or {})
    conf = dict(out.get("_confidence") or {})
    sources = dict(out.get("_sources") or {})
    hybrid = hybrid or {}

    overrides = out.get("_overrides") or {}
    if "total_years_experience" in overrides and overrides["total_years_experience"] is not None:
        try:
            out["total_years_experience"] = max(
                0.0, min(50.0, float(overrides["total_years_experience"]))
            )
            conf["total_years_experience"] = "high"
            sources["total_years_experience"] = "recruiter_override"
            out["_years_method"] = "recruiter_override"
            out["_confidence"] = conf
            out["_sources"] = sources
            return out
        except (TypeError, ValueError):
            pass

    work = out.get("work_history") or []
    if isinstance(work, list) and work:
        work = enrich_work_history_years(work)
        # Strip internal flags before persistence? keep for debug then clean
        cleaned = []
        for w in work:
            row = {k: v for k, v in w.items() if not str(k).startswith("_")}
            cleaned.append(row)
        out["work_history"] = cleaned
        # keep enriched with flags only for compute
        work_for_compute = work
    else:
        work_for_compute = []

    computed, method = compute_years_from_work_history(work_for_compute)

    # Explicit "X years experience" from resume text (not degree length)
    explicit = None
    if resume_text:
        explicit = _explicit_years(resume_text)
    if explicit is None:
        # hybrid explicit only if source was regex phrase, not date_range guess
        h_src = (hybrid.get("_sources") or {}).get("total_years_experience")
        if h_src == "regex" and hybrid.get("total_years_experience") is not None:
            try:
                explicit = float(hybrid["total_years_experience"])
            except (TypeError, ValueError):
                explicit = None

    student = _looks_like_student(out, resume_text)
    real_jobs = [
        w
        for w in (work_for_compute or [])
        if isinstance(w, dict) and w.get("_count_as_experience")
    ]

    final: Optional[float] = None
    final_method = "none"

    if student and not real_jobs:
        # Fresher / student: do NOT treat degree dates as experience
        final = 0.0
        final_method = "student_no_jobs"
    elif computed is not None and method not in ("none",):
        final = float(computed)
        final_method = method
        if explicit is not None and explicit > 0:
            # Prefer explicit professional experience phrase when present
            if abs(explicit - final) <= 2.0:
                final = round((final * 0.5 + explicit * 0.5), 1)
                final_method = f"{method}+explicit"
            elif explicit < final:
                # Don't inflate past a lower explicit claim
                final = explicit
                final_method = "explicit_phrase"
    elif explicit is not None:
        final = float(explicit)
        final_method = "explicit_phrase"
    elif student:
        final = 0.0
        final_method = "student_default_zero"
    else:
        # LLM fallback only if we have real jobs; otherwise 0
        if real_jobs:
            try:
                final = float(out.get("total_years_experience") or 0)
                final_method = "llm_fallback"
            except (TypeError, ValueError):
                final = 0.0
                final_method = "default_zero"
        else:
            final = 0.0
            final_method = "no_real_jobs"

    if final is None:
        final = 0.0
        final_method = "default_zero"

    # Sanity: student with only internships can have small years; cap wild LLM values
    if student and final > 2.0 and not (explicit and explicit > 2):
        # Degree length often leaks as 3–4; treat as fresher unless real multi-year jobs
        if not real_jobs or sum(float(j.get("years") or 0) for j in real_jobs) <= 1.5:
            final = min(final, sum(float(j.get("years") or 0) for j in real_jobs) if real_jobs else 0.0)
            final_method = "student_capped"

    out["total_years_experience"] = max(0.0, min(50.0, round(float(final), 1)))
    conf["total_years_experience"] = (
        "high"
        if final_method
        in (
            "work_history_sum",
            "work_history_span",
            "explicit_phrase",
            "recruiter_override",
            "student_no_jobs",
            "no_real_jobs",
        )
        or final_method.endswith("+explicit")
        else "medium"
    )
    sources["total_years_experience"] = final_method
    out["_years_method"] = final_method
    out["_confidence"] = conf
    out["_sources"] = sources
    return out


def apply_recruiter_overrides(
    structured: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply recruiter field corrections. Stores audit trail under _overrides.
    List fields accept comma-separated strings or lists.
    """
    out = dict(structured or {})
    applied: dict[str, Any] = dict(out.get("_overrides") or {})
    conf = dict(out.get("_confidence") or {})
    sources = dict(out.get("_sources") or {})

    scalar_fields = (
        "name",
        "email",
        "phone",
        "location",
        "total_years_experience",
        "notice_period_days",
        "summary",
    )
    list_fields = ("skills", "certifications", "licenses")

    for field in scalar_fields:
        if field not in overrides:
            continue
        val = overrides[field]
        if val is None or val == "":
            continue
        if field in ("total_years_experience",):
            try:
                val = max(0.0, min(50.0, float(val)))
            except (TypeError, ValueError):
                continue
        if field == "notice_period_days":
            try:
                val = max(0, min(365, int(float(val))))
            except (TypeError, ValueError):
                continue
        out[field] = val
        applied[field] = val
        conf[field] = "high"
        sources[field] = "recruiter_override"

    for field in list_fields:
        if field not in overrides:
            continue
        val = overrides[field]
        if val is None:
            continue
        if isinstance(val, str):
            items = [x.strip() for x in val.split(",") if x.strip()]
        elif isinstance(val, list):
            items = [str(x).strip() for x in val if str(x).strip()]
        else:
            continue
        # dedupe
        seen: set[str] = set()
        clean: list[str] = []
        for s in items:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                clean.append(s)
        out[field] = clean
        applied[field] = clean
        conf[field] = "high"
        sources[field] = "recruiter_override"

    out["_overrides"] = applied
    out["_confidence"] = conf
    out["_sources"] = sources
    # Re-run two-pass years if years overridden or work history present
    out = apply_two_pass_years(out, hybrid={})
    return sanitize_structured(out)


def validate_structured(data: dict[str, Any]) -> list[str]:
    """Return list of validation issues (empty = OK enough to use)."""
    issues: list[str] = []
    if not isinstance(data, dict):
        return ["root is not an object"]

    email = data.get("email")
    if email is not None and email != "" and not EMAIL_RE.fullmatch(str(email).strip()):
        # allow if contains valid email
        if not EMAIL_RE.search(str(email)):
            issues.append("email format invalid")

    years = data.get("total_years_experience")
    if years is not None:
        try:
            y = float(years)
            if y < 0 or y > 50:
                issues.append("total_years_experience out of range 0-50")
        except (TypeError, ValueError):
            issues.append("total_years_experience not numeric")

    notice = data.get("notice_period_days")
    if notice is not None:
        try:
            n = int(float(notice))
            if n < 0 or n > 365:
                issues.append("notice_period_days out of range")
        except (TypeError, ValueError):
            issues.append("notice_period_days not numeric")

    for list_field in (
        "skills",
        "certifications",
        "licenses",
        "education",
        "work_history",
    ):
        val = data.get(list_field)
        if val is not None and not isinstance(val, list):
            issues.append(f"{list_field} must be a list")

    # Soft requirement: need either name or email for identity
    if not data.get("name") and not data.get("email"):
        issues.append("missing both name and email")

    return issues


def sanitize_structured(data: dict[str, Any]) -> dict[str, Any]:
    """Clamp types/ranges after validation or merge."""
    out = dict(data or {})

    def _num(v: Any, default: float = 0.0) -> float:
        try:
            if v is None or v == "":
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    years = _num(out.get("total_years_experience"), 0.0)
    out["total_years_experience"] = max(0.0, min(50.0, round(years, 1)))

    notice = out.get("notice_period_days")
    if notice is not None and notice != "":
        try:
            out["notice_period_days"] = max(0, min(365, int(float(notice))))
        except (TypeError, ValueError):
            out["notice_period_days"] = None
    else:
        out["notice_period_days"] = None

    email = out.get("email")
    if email:
        m = EMAIL_RE.search(str(email))
        out["email"] = m.group(0) if m else None

    phone = out.get("phone")
    if phone:
        digits = re.sub(r"\D", "", str(phone))
        if len(digits) < PHONE_MIN_DIGITS:
            out["phone"] = None
        else:
            out["phone"] = str(phone).strip()

    for lf in ("skills", "certifications", "licenses"):
        val = out.get(lf) or []
        if isinstance(val, str):
            val = [x.strip() for x in val.split(",") if x.strip()]
        if not isinstance(val, list):
            val = []
        seen: set[str] = set()
        clean: list[str] = []
        for item in val:
            s = str(item).strip()
            if not s:
                continue
            k = s.lower()
            if k not in seen:
                seen.add(k)
                clean.append(s)
        out[lf] = clean

    edu = out.get("education") or []
    if not isinstance(edu, list):
        edu = []
    clean_edu = []
    for e in edu:
        if isinstance(e, dict):
            clean_edu.append(
                {
                    "degree": e.get("degree") or "",
                    "field": e.get("field") or "",
                    "institution": e.get("institution") or "",
                    "year": e.get("year"),
                }
            )
        elif e:
            clean_edu.append(
                {"degree": str(e), "field": "", "institution": "", "year": None}
            )
    out["education"] = clean_edu

    work = out.get("work_history") or []
    if not isinstance(work, list):
        work = []
    clean_work = []
    for w in work:
        if isinstance(w, dict):
            yrs = w.get("years")
            try:
                yrs_f = float(yrs) if yrs is not None and yrs != "" else None
            except (TypeError, ValueError):
                yrs_f = None
            clean_work.append(
                {
                    "title": w.get("title") or "",
                    "company": w.get("company") or "",
                    "duration": w.get("duration") or "",
                    "years": yrs_f,
                }
            )
    out["work_history"] = clean_work

    if out.get("name"):
        out["name"] = str(out["name"]).strip()[:120] or None
    if out.get("location"):
        out["location"] = str(out["location"]).strip()[:120] or None
    if out.get("summary"):
        out["summary"] = str(out["summary"]).strip()[:600]
    else:
        out["summary"] = out.get("summary") or ""

    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _first_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None


def _first_phone(text: str) -> Optional[str]:
    best = None
    for m in PHONE_RE.finditer(text or ""):
        raw = m.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        # Filter years like 2019-2021 and short numbers
        if len(digits) < PHONE_MIN_DIGITS or len(digits) > 15:
            continue
        # Avoid pure year ranges captured poorly
        if re.fullmatch(r"20\d{2}\s*[-–]\s*20\d{2}", raw):
            continue
        best = raw
        break
    return best


def _first_match(pat: re.Pattern, text: str) -> Optional[str]:
    m = pat.search(text or "")
    return m.group(0).rstrip("/") if m else None


# Lines / phrases that look like address/location, not a person name
_NAME_REJECT_LINE = re.compile(
    r"(?i)\b("
    r"based\s+in|located\s+in|residing\s+in|lives?\s+in|live\s+in|"
    r"address|current\s+location|location\s*:|relocat|"
    r"open\s+to\s+work|seeking|looking\s+for|available\s+for|"
    r"phone|mobile|email|linkedin|github|http|www\.|"
    r"resume|curriculum|vitae|\bcv\b|objective|summary|profile|contact|"
    r"experience|education|skills|certification|project|technologies|"
    r"engineer|developer|student|intern|bachelor|master|b\.?tech|"
    r"database|mongodb|nosql|mysql|postgres|redis|sql|html|css|"
    r"javascript|typescript|python|java|react|node|express|docker|"
    r"kubernetes|aws|azure|gcp|fullstack|full\s*stack|backend|frontend|"
    r"novelties|include|hands[\s\-]?on|proficient|responsible|"
    r"interested|passionate|motivated|dedicated|results?\s*driven"
    r")\b"
)

# Filename noise tokens (not part of a person's name)
_FILENAME_NOISE_TOKENS = {
    "resume", "resumes", "cv", "cvs", "biodata", "bio", "data", "doc", "docs",
    "document", "documents", "file", "files", "copy", "final", "updated",
    "latest", "new", "old", "draft", "scan", "scanned", "pdf", "docx", "doc",
    "naukri", "linkedin", "indeed", "monster", "shine", "foundit", "portal",
    "download", "attachment", "untitled", "unknown", "candidate", "profile",
    "my", "the", "and", "of", "for", "to", "v1", "v2", "v3", "ver", "version",
    "info", "details", "application", "appl", "soft", "copy", "hard",
}

# Words that almost never appear in real person names (prose / resume body)
_NAME_PROSE_WORDS = {
    "my", "our", "the", "a", "an", "and", "or", "but", "with", "from", "for",
    "to", "of", "in", "on", "at", "by", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can",
    "include", "includes", "including", "have", "having", "using", "used",
    "working", "worked", "work", "experience", "experienced", "years",
    "year", "skills", "skill", "knowledge", "proficient", "strong",
    "hands", "hand", "on", "off", "novelties", "novelty", "interested",
    "looking", "seeking", "available", "immediate", "currently", "current",
    "responsible", "responsibilities", "developed", "designed", "created",
    "built", "managed", "led", "team", "project", "projects", "about",
    "me", "objective", "summary", "profile", "contact", "phone", "email",
    "mobile", "address", "location", "info", "information", "details",
    "this", "that", "these", "those", "their", "his", "her", "its",
    "also", "etc", "via", "per", "into", "over", "under", "above",
    "some", "any", "all", "body", "text", "page", "pages", "section",
    "software", "hardware", "system", "systems", "role", "roles",
    "position", "company", "organization", "university", "college",
    "school", "fresher", "graduate", "professional", "personal",
}

# Common place / filler words that should not appear in a personal name
_NAME_PLACE_WORDS = {
    "based", "in", "at", "near", "from", "of", "the", "and", "ncr", "delhi",
    "noida", "gurgaon", "gurugram", "ghaziabad", "indirapuram", "bangalore",
    "bengaluru", "hyderabad", "chennai", "mumbai", "pune", "kolkata", "india",
    "remote", "onsite", "location", "address", "area", "sector", "street",
    "road", "city", "state", "country", "pin", "zip",
}

# Tech / skill tokens that must never be used as a person name or city
_TECH_WORDS = {
    "database", "mongodb", "nosql", "mysql", "postgres", "postgresql", "redis",
    "sql", "sqlite", "oracle", "dynamodb", "cassandra",
    "javascript", "typescript", "python", "java", "golang", "rust", "kotlin",
    "html", "css", "react", "angular", "vue", "node", "nodejs", "express",
    "nextjs", "next", "django", "flask", "fastapi", "spring",
    "docker", "kubernetes", "k8s", "aws", "azure", "gcp", "linux", "git",
    "ci", "cd", "api", "rest", "graphql", "microservices", "backend",
    "frontend", "fullstack", "devops", "ml", "ai", "nlp", "tensorflow",
    "pytorch", "pandas", "numpy", "bootstrap", "tailwind", "sass",
    "jquery", "redux", "mongodb", "mongoose", "prisma", "firebase",
    "hadoop", "spark", "kafka", "rabbitmq", "nginx", "apache",
    "c", "cpp", "csharp", "php", "ruby", "swift", "scala", "r",
    "skills", "technologies", "tools", "frameworks", "libraries",
    "programming", "languages", "technical", "proficient", "expertise",
}


def _norm_token(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(w).lower())


def _token_is_tech(w: str) -> bool:
    t = _norm_token(w)
    if not t:
        return False
    if t in _TECH_WORDS:
        return True
    # skill lexicon match
    for skill in SKILL_LEXICON:
        if _norm_token(skill) == t:
            return True
    return False


def _looks_like_tech_or_skills_blob(text: str) -> bool:
    """True if string is mostly technologies (e.g. 'Database MongoDB NoSQL', 'JavaScript, HTML')."""
    if not text:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9+.#\-']*", str(text))
    if not words:
        return False
    tech_hits = sum(1 for w in words if _token_is_tech(w))
    # 1+ tech word in a short phrase, or majority tech
    if tech_hits >= 1 and len(words) <= 4 and tech_hits >= max(1, len(words) - 1):
        return True
    if tech_hits >= 2:
        return True
    if tech_hits / max(len(words), 1) >= 0.5 and tech_hits >= 1:
        return True
    return False


def _looks_like_prose_name(name: str) -> bool:
    """True if string is resume body / sentence, not a person name."""
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", str(name or ""))
    if not words:
        return True
    lower = [w.lower().strip("-'") for w in words]
    prose_hits = sum(1 for w in lower if w in _NAME_PROSE_WORDS)
    # Any prose glue word in a short "name" is almost always wrong
    if prose_hits >= 1 and len(words) >= 3:
        return True
    if prose_hits >= 2:
        return True
    # "My Something" / "The Something"
    if lower and lower[0] in ("my", "our", "the", "a", "an", "this", "his", "her"):
        return True
    # Verb-like endings in multi-word phrases
    if any(w.endswith(("ing", "ed", "ly")) and w not in ("king", "young") for w in lower):
        if len(words) >= 3 or prose_hits:
            return True
    return False


def _is_plausible_person_name(name: Any, *, allow_single: bool = False) -> bool:
    """Reject location/skill/prose lines mistaken for a candidate name."""
    if name is None:
        return False
    raw = str(name).strip()
    if not raw or len(raw) < 3 or len(raw) > 80:
        return False
    # Strip trailing noise tokens before validating (e.g. "Amit Singh Resume")
    cleaned = _strip_name_noise_tokens(raw)
    if cleaned:
        raw = cleaned
    if _NAME_REJECT_LINE.search(raw):
        return False
    if _looks_like_tech_or_skills_blob(raw):
        return False
    if _looks_like_prose_name(raw):
        return False
    if EMAIL_RE.search(raw) or (PHONE_RE.search(raw) and len(re.sub(r"\D", "", raw)) >= 10):
        return False
    # Reject if line has digits (page numbers, experience years)
    if re.search(r"\d", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", raw)
    if len(words) == 1:
        w = words[0]
        if not allow_single:
            return False
        if len(w) < 5 or _token_is_tech(w) or w.lower() in _NAME_PLACE_WORDS:
            return False
        if w.lower() in _NAME_PROSE_WORDS or w.lower() in _FILENAME_NOISE_TOKENS:
            return False
        # Avoid all-caps acronyms (HTML, CSS, AWS)
        if w.isupper() and len(w) <= 5:
            return False
        return w[0].isalpha()
    if not (2 <= len(words) <= 4):
        return False
    lower_words = [w.lower() for w in words]
    if any(_token_is_tech(w) for w in words):
        return False
    if any(w in _FILENAME_NOISE_TOKENS for w in lower_words):
        return False
    if any(w in _NAME_PROSE_WORDS for w in lower_words):
        return False
    place_hits = sum(1 for w in lower_words if w in _NAME_PLACE_WORDS)
    if place_hits >= 1 and place_hits >= len(words) - 1:
        return False
    if place_hits >= 2:
        return False
    content = [w for w in lower_words if w not in _NAME_PLACE_WORDS and len(w) > 1]
    if len(content) < 2:
        return False
    # Prefer name-like casing: Title Case or ALL CAPS or First Last
    titled = sum(1 for w in words if w[0].isupper())
    if titled == 0 and len(words) >= 2:
        # all-lowercase multi-word is usually prose
        return False
    return True


def _strip_name_noise_tokens(text: str) -> str:
    """Remove resume/cv/doc/info and similar tokens from a name-like string."""
    if not text:
        return ""
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", text)
    kept = [w for w in words if w.lower().strip("-'") not in _FILENAME_NOISE_TOKENS]
    # Also drop pure numeric leftovers already removed by findall
    if not kept:
        return ""
    return " ".join(kept)


def _format_name_words(words: list[str]) -> str:
    out = []
    for w in words:
        if w.isupper() or w.islower():
            out.append(w.title())
        else:
            out.append(w)
    return " ".join(out)


def _is_plausible_location(loc: Any) -> bool:
    """Reject skill lists mis-tagged as location (e.g. 'JavaScript, HTML')."""
    if loc is None:
        return False
    raw = str(loc).strip()
    if not raw or len(raw) < 2 or len(raw) > 120:
        return False
    if _looks_like_tech_or_skills_blob(raw):
        return False
    if EMAIL_RE.search(raw) or (PHONE_RE.search(raw) and len(re.sub(r"\D", "", raw)) >= 10):
        return False
    # Pure skill dump with commas
    parts = [p.strip() for p in re.split(r"[,|/•·;]", raw) if p.strip()]
    if len(parts) >= 2 and sum(1 for p in parts if _looks_like_tech_or_skills_blob(p)) >= 2:
        return False
    words = re.findall(r"[A-Za-z]+", raw)
    if words and all(_token_is_tech(w) for w in words):
        return False
    # Must not be only tech
    if words and sum(1 for w in words if _token_is_tech(w)) >= max(1, len(words) - 0):
        if sum(1 for w in words if _token_is_tech(w)) == len(words):
            return False
    return True


def _name_from_email(email: Optional[str]) -> Optional[str]:
    if not email or "@" not in email:
        return None
    local = email.split("@", 1)[0]
    local = re.sub(r"[0-9]+", " ", local)
    local = local.replace(".", " ").replace("_", " ").replace("-", " ").replace("+", " ")
    parts = [p for p in local.split() if len(p) > 1 and p.isalpha()]
    if len(parts) >= 2:
        name = " ".join(p.title() for p in parts[:4])
        if _is_plausible_person_name(name) or not _looks_like_tech_or_skills_blob(name):
            return name
    if len(parts) == 1 and len(parts[0]) >= 5:
        token = parts[0]
        if _token_is_tech(token):
            return None
        # Try to split camelCase / FirstLast
        split = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
        if " " in split:
            cand = " ".join(p.title() for p in split.split())
            if _is_plausible_person_name(cand):
                return cand
        return token.title()
    return None


def _name_from_filename(filename: Optional[str]) -> Optional[str]:
    """
    Parse portal-style filenames which are often more reliable than PDF text:
      Naukri_PALLAVIUTEKAR[3y_1m].pdf
      Resume_John_Doe.pdf
      GARV RANDHAR_Doc-6.pdf
      Amit_Singh Resume.pdf
      AYAN NASHINE RESUME (1) (1).pdf
    """
    if not filename:
        return None
    stem = str(filename)
    # strip path
    stem = stem.replace("\\", "/").split("/")[-1]
    # strip extension
    stem = re.sub(r"\.(pdf|docx?|txt|rtf)$", "", stem, flags=re.I)
    # strip experience brackets: [3y_1m], (3 years), (1)
    stem = re.sub(r"[\[\(].*?[\]\)]", " ", stem)
    # strip common portal / document prefixes (repeat for Resume_CV_Name)
    for _ in range(3):
        next_stem = re.sub(
            r"(?i)^(naukri|linkedin|indeed|monster|shine|foundit|resume|cv|"
            r"biodata|doc|document|file|copy|my|the)[\s_\-\.]+",
            "",
            stem,
        )
        if next_stem == stem:
            break
        stem = next_stem
    # strip experience / version tails
    stem = re.sub(r"(?i)[_\-\s]+\d+\s*[ym](?:ear|r|onth|o)?s?\b.*$", " ", stem)
    stem = re.sub(r"(?i)[_\-\s]+exp(?:erience)?.*$", " ", stem)
    stem = re.sub(r"(?i)[_\-\s]+v\d+\b.*$", " ", stem)
    stem = re.sub(r"\d+", " ", stem)  # page/copy numbers
    # separators → spaces
    stem = stem.replace("_", " ").replace("-", " ").replace(".", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem or len(stem) < 3:
        return None

    # Drop noise tokens (Resume, Doc, Info, CV, …) then reassemble
    words = re.findall(r"[A-Za-z][A-Za-z']+", stem)
    words = [w for w in words if w.lower() not in _FILENAME_NOISE_TOKENS]
    if not words:
        return None

    # ALLCAPS FirstLast glued: PALLAVIUTEKAR / Amitsingh
    if len(words) == 1 and len(words[0]) >= 6:
        w = words[0]
        split = re.sub(r"([a-z])([A-Z])", r"\1 \2", w)
        if " " in split:
            parts = split.split()
            cand = _format_name_words(parts)
            if _is_plausible_person_name(cand):
                return cand
        # Heuristic: split long ALLCAPS Indian names is hard — keep title case
        cand = w.title()
        if not _token_is_tech(cand) and cand.lower() not in _NAME_PROSE_WORDS:
            # Single token from filename is acceptable with allow_single
            if _is_plausible_person_name(cand, allow_single=True):
                return cand
        return None

    if 2 <= len(words) <= 4:
        cand = _format_name_words(words)
        if _is_plausible_person_name(cand):
            return cand
        # Filename after noise strip: trust 2–3 alpha tokens even if casing odd
        if (
            not _looks_like_tech_or_skills_blob(cand)
            and not _looks_like_prose_name(cand)
            and not any(_token_is_tech(w) for w in words)
            and not any(w.lower() in _NAME_PROSE_WORDS for w in words)
        ):
            return cand
    return None


def _normalize_name_key(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def _name_quality_score(name: str, source: str) -> int:
    """Higher = more trustworthy display name."""
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", name or "")
    score = 0
    base = {
        "filename": 78,
        "email": 72,
        "llm": 68,
        "heuristic": 48,
    }
    # Multi-source tags like "email+filename"
    parts = source.split("+")
    score = max(base.get(p, 40) for p in parts)
    if len(parts) >= 2:
        score += 18
    if len(words) == 2:
        score += 8  # classic First Last
    elif len(words) == 3:
        score += 4
    elif len(words) == 1:
        score -= 12
    elif len(words) >= 4:
        score -= 10
    # Title-case bonus
    if words and all(w[0].isupper() for w in words):
        score += 4
    if _looks_like_prose_name(name):
        score -= 40
    if any(w.lower() in _FILENAME_NOISE_TOKENS for w in words):
        score -= 30
    return score


def resolve_candidate_name(
    *,
    llm_name: Any = None,
    hybrid_name: Any = None,
    email: Any = None,
    filename: Any = None,
    resume_text: str = "",
) -> dict[str, Any]:
    """
    Multi-source name resolution for production HR use.

    Priority: consensus > clean filename > email > LLM > header heuristic.
    Never return prose / skill / Doc-suffix junk as a name.
    """
    email_s = str(email).strip() if email else None
    file_s = str(filename).strip() if filename else None

    candidates: list[tuple[str, str, int, str]] = []
    # (name, source, score, confidence)

    fn = _name_from_filename(file_s)
    if fn:
        # Portal / structured filenames are strong signals
        score = 92 if re.search(r"(?i)naukri|linkedin|indeed", file_s or "") else 82
        # 2-word cleaned names from file are high confidence
        conf = "high" if len(fn.split()) >= 2 else "medium"
        candidates.append((fn, "filename", score, conf))

    en = _name_from_email(email_s)
    if en and _is_plausible_person_name(en, allow_single=True):
        score = 80 if " " in en else 52
        candidates.append((en, "email", score, "medium" if " " in en else "low"))

    llm_clean = _strip_name_noise_tokens(str(llm_name).strip()) if llm_name else None
    if llm_clean and _is_plausible_person_name(llm_clean):
        candidates.append((llm_clean, "llm", 70, "medium"))
    elif llm_clean and _is_plausible_person_name(llm_clean, allow_single=True):
        candidates.append((llm_clean, "llm", 48, "low"))

    hyb_clean = _strip_name_noise_tokens(str(hybrid_name).strip()) if hybrid_name else None
    if hyb_clean and _is_plausible_person_name(hyb_clean):
        # Heuristic alone is weaker — lose to filename/email
        candidates.append((hyb_clean, "heuristic", 45, "medium"))

    # Bonus when sources agree (normalized)
    if len(candidates) >= 2:
        keys: dict[str, list[tuple[str, str, int, str]]] = {}
        for name, src, score, conf in candidates:
            k = _normalize_name_key(name)
            keys.setdefault(k, []).append((name, src, score, conf))
        boosted: list[tuple[str, str, int, str]] = []
        for k, group in keys.items():
            if len(group) >= 2:
                best = max(group, key=lambda x: (x[2], len(x[0])))
                srcs = "+".join(sorted({g[1] for g in group}))
                boosted.append(
                    (best[0], srcs, min(100, best[2] + 22), "high")
                )
            else:
                boosted.append(group[0])
        candidates = boosted

    if not candidates:
        return {
            "name": None,
            "source": "none",
            "confidence": "low",
            "warning": "Could not extract a reliable name — verify from the resume file.",
        }

    # Drop remaining junk
    clean = []
    for name, src, score, conf in candidates:
        name = _strip_name_noise_tokens(name) or name
        if not name:
            continue
        if _looks_like_tech_or_skills_blob(name):
            continue
        if _looks_like_prose_name(name):
            continue
        if _NAME_REJECT_LINE.search(name) and "filename" not in src:
            # Filename path already stripped noise; reject line is extra safety for others
            continue
        if not _is_plausible_person_name(name, allow_single=("email" in src or "filename" in src)):
            # Allow single-token only from email/filename
            if not (
                " " not in name
                and ("email" in src or "filename" in src)
                and _is_plausible_person_name(name, allow_single=True)
            ):
                continue
        q = _name_quality_score(name, src)
        clean.append((name, src, score + q // 5, conf, q))

    if not clean:
        return {
            "name": None,
            "source": "rejected",
            "confidence": "low",
            "warning": "Extracted name candidates looked invalid (skills/headers/prose).",
        }

    # Prefer quality, then score, then multi-word completeness
    clean.sort(key=lambda x: (-x[4], -x[2], -len(x[0].split()), -len(x[0])))
    name, src, score, conf, q = clean[0]

    # Prefer filename/email over weak heuristic when scores are close
    if "heuristic" in src and len(clean) > 1:
        for alt in clean[1:]:
            if alt[1] in ("filename", "email") or "filename" in alt[1] or "email" in alt[1]:
                if alt[4] + 5 >= q:
                    name, src, score, conf, q = alt
                    break

    warning = None
    if conf == "low" or score < 55 or "heuristic" in src and "+" not in src:
        warning = (
            f"Name “{name}” is low-confidence (source: {src}). "
            "Please confirm before using for outreach."
        )
    return {
        "name": name,
        "source": src,
        "confidence": conf if q >= 50 else "low",
        "warning": warning,
        "score": score,
    }


def _line_name_score(line: str, name: str) -> int:
    """Rank how name-like a header line is (higher = better)."""
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", name)
    score = 10
    if len(words) == 2:
        score += 25
    elif len(words) == 3:
        score += 15
    elif len(words) == 1:
        score += 0
    else:
        score -= 5
    # Short lines are better (names aren't long sentences)
    if len(line) < 35:
        score += 12
    elif len(line) < 50:
        score += 5
    # Title / ALL CAPS name lines
    if words and all(w[0].isupper() for w in words):
        score += 10
    if line.isupper() and 2 <= len(words) <= 3:
        score += 8
    # Near contact punctuation
    if re.search(r"[|•·]", line):
        score += 6
    if _looks_like_prose_name(name):
        score -= 50
    return score


def _guess_name(header: str, email: Optional[str], full_text: str = "") -> Optional[str]:
    """
    Header heuristic: prefer short Title-Case lines near email/phone.
    Never return the first random 2–4 word phrase from the body.
    """
    blocks: list[str] = []
    if email and full_text:
        lines = full_text.split("\n")
        for i, ln in enumerate(lines):
            if email.lower() in ln.lower():
                lo, hi = max(0, i - 5), min(len(lines), i + 2)
                blocks.append("\n".join(lines[lo:hi]))
                break
    if header:
        blocks.append(header)
    if full_text and full_text != header:
        # Only first 8 lines — true contact header zone
        blocks.append("\n".join(full_text.split("\n")[:8]))

    scored: list[tuple[int, str]] = []
    seen: set[str] = set()

    for block in blocks:
        lines = [ln.strip() for ln in (block or "").split("\n") if ln.strip()]
        for ln in lines[:12]:
            ln_clean = re.sub(
                r"(?i)^(name|candidate|full\s*name)\s*[:\-]\s*", "", ln
            ).strip()
            # Drop lines that are clearly not names early
            if len(ln_clean) > 55:
                continue
            if _NAME_REJECT_LINE.search(ln_clean) or _looks_like_tech_or_skills_blob(ln_clean):
                continue
            if _looks_like_prose_name(ln_clean):
                continue

            candidate: Optional[str] = None
            if EMAIL_RE.search(ln_clean):
                left = re.split(r"[|•·,]", ln_clean)[0].strip()
                left = _strip_name_noise_tokens(left) or left
                if left and not EMAIL_RE.search(left) and _is_plausible_person_name(left):
                    candidate = left
            elif PHONE_RE.search(ln_clean) and len(re.sub(r"\D", "", ln_clean)) >= 10:
                left = re.split(r"[|•·,]", ln_clean)[0].strip()
                left = _strip_name_noise_tokens(left) or left
                if left and _is_plausible_person_name(left):
                    candidate = left
            else:
                words = re.findall(r"[A-Za-z][A-Za-z\-']+", ln_clean)
                # Only 2–3 tokens for free-standing name lines
                if 2 <= len(words) <= 3 and len(ln_clean) < 48:
                    # Reject if line has many non-name characters
                    if re.search(r"[:;/\\@]{2,}|\d{3,}", ln_clean):
                        continue
                    name = _format_name_words(words)
                    if _is_plausible_person_name(name):
                        candidate = name

            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            scored.append((_line_name_score(ln_clean, candidate), candidate))

    if scored:
        scored.sort(key=lambda x: (-x[0], -len(x[1].split())))
        best_score, best = scored[0]
        if best_score >= 20:
            return best

    email_name = _name_from_email(email)
    if email_name and _is_plausible_person_name(email_name, allow_single=True):
        return email_name
    return None


def _guess_location(header: str, full: str) -> Optional[str]:
    m = LOCATION_LINE_RE.search(header) or LOCATION_LINE_RE.search(full)
    if m:
        cand = m.group(1).strip(" .,-")
        if _is_plausible_location(cand):
            return cand
    for ln in (header or "").split("\n")[:12]:
        if _looks_like_tech_or_skills_blob(ln):
            continue
        if _NAME_REJECT_LINE.search(ln) and not re.search(
            r"(?i)based\s+in|located\s+in|location", ln
        ):
            # skip pure skill/role lines; allow "based in X"
            if not re.search(r"(?i)based\s+in|located|location\s*:", ln):
                continue
        cm = CITY_STATE_RE.search(ln)
        if cm:
            cand = f"{cm.group(1)}, {cm.group(2)}"
            if _is_plausible_location(cand):
                return cand
        # "City, Country" or single known-style place line
        if re.search(r"(?i)\b(based\s+in|location\s*:)\s*(.+)$", ln):
            m2 = re.search(r"(?i)(?:based\s+in|location\s*:)\s*(.+)$", ln)
            if m2:
                cand = m2.group(1).strip(" .,-|")
                if _is_plausible_location(cand):
                    return cand
    return None


def _explicit_years(text: str) -> Optional[float]:
    candidates: list[float] = []
    for pat in (YEARS_EXP_PHRASE_RE, YEARS_EXPLICIT_RE):
        for m in pat.finditer(text or ""):
            try:
                v = float(m.group(1))
                if 0 < v <= 50:
                    candidates.append(v)
            except (TypeError, ValueError):
                continue
    if not candidates:
        return None
    # Prefer values near phrases about total experience — take max reasonable
    return max(candidates)


def _estimate_years_from_dates(text: str) -> tuple[Optional[float], list[dict[str, Any]]]:
    ranges: list[dict[str, Any]] = []
    total = 0.0
    for m in DATE_RANGE_RE.finditer(text or ""):
        start_s, end_s = m.group(1), m.group(2)
        try:
            start = int(start_s)
        except ValueError:
            continue
        end_l = end_s.lower()
        if any(x in end_l for x in ("present", "current", "now", "ongoing", "till")):
            end = CURRENT_YEAR
        else:
            try:
                end = int(re.search(r"(?:19|20)\d{2}", end_s).group())  # type: ignore[union-attr]
            except Exception:
                continue
        if end < start or start < 1970 or end > CURRENT_YEAR + 1:
            continue
        yrs = float(end - start)
        if yrs > 40:
            continue
        ranges.append(
            {
                "duration": f"{start_s}-{end_s}",
                "years": yrs,
                "span": m.group(0),
            }
        )
        total += yrs
    # Cap: overlapping jobs double-count; use min(total, max_span) heuristic
    if not ranges:
        return None, []
    earliest = min(int(r["duration"].split("-")[0]) for r in ranges if r["duration"][:4].isdigit())
    # crude career span
    span = float(CURRENT_YEAR - earliest)
    est = min(total, span, 50.0)
    # If many overlapping, span is better
    if total > span * 1.3:
        est = span
    return round(est, 1), ranges


def _parse_notice_period(text: str) -> Optional[int]:
    if NOTICE_IMMEDIATE_RE.search(text or ""):
        return 0
    m = NOTICE_RE.search(text or "")
    if not m:
        # bare "Notice period: 30 days"
        m2 = re.search(
            r"(?i)notice[^.\n]{0,40}?(\d+)\s*(day|days|week|weeks|month|months)",
            text or "",
        )
        if not m2:
            if re.search(r"(?i)\bimmediate\b", text or "") and re.search(
                r"(?i)notice", text or ""
            ):
                return 0
            return None
        num = int(m2.group(1))
        unit = m2.group(2).lower()
        return _to_days(num, unit)

    raw = m.group(1).lower().strip()
    if "immediate" in raw:
        return 0
    if "serving" in raw:
        return 0
    m3 = re.search(r"(\d+)\s*(day|days|week|weeks|month|months)", raw)
    if not m3:
        return None
    return _to_days(int(m3.group(1)), m3.group(2))


def _to_days(num: int, unit: str) -> int:
    unit = unit.lower()
    if unit.startswith("day"):
        return num
    if unit.startswith("week"):
        return num * 7
    if unit.startswith("month"):
        return num * 30
    return num


def _match_skills(text: str) -> list[str]:
    found: list[str] = []
    lower = text or ""
    # Sort longer phrases first to prefer "Spring Boot" over "Spring"
    for skill in sorted(SKILL_LEXICON, key=len, reverse=True):
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])", re.I)
        if pat.search(lower):
            # Normalize display to lexicon canonical casing
            if skill.lower() not in {f.lower() for f in found}:
                found.append(skill)
    return found


def _match_certs(text: str) -> list[str]:
    found: list[str] = []
    for cert in sorted(CERT_KEYWORDS, key=len, reverse=True):
        if re.search(re.escape(cert), text or "", re.I):
            if cert.lower() not in {f.lower() for f in found}:
                found.append(cert)
    return found


def _parse_education(text: str) -> list[dict[str, Any]]:
    edu: list[dict[str, Any]] = []
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    for ln in lines[:40]:
        degree = None
        for pat, label in DEGREE_PATTERNS:
            if pat.search(ln):
                degree = label
                break
        if not degree and not INSTITUTION_HINTS.search(ln):
            continue
        year_m = re.search(r"\b((?:19|20)\d{2})\b", ln)
        year = year_m.group(1) if year_m else None
        field = ""
        # crude field after "in" or degree
        fm = re.search(
            r"(?i)(?:in|of)\s+([A-Za-z &/]{3,40})(?:\s+from|\s+at|,|$)", ln
        )
        if fm:
            field = fm.group(1).strip()
        inst = ""
        im = INSTITUTION_HINTS.search(ln)
        if im:
            # take a window around the match
            inst = ln[max(0, im.start() - 20) : im.end() + 40].strip(" ,-|")
        if degree or inst:
            edu.append(
                {
                    "degree": degree or "",
                    "field": field,
                    "institution": inst or ln[:80],
                    "year": year,
                }
            )
    # Dedupe by degree+year
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for e in edu:
        k = f"{e.get('degree')}|{e.get('year')}|{e.get('institution')}"
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq[:8]


def _recompute_years_from_work_history(data: dict[str, Any]) -> dict[str, Any]:
    work = data.get("work_history") or []
    if not isinstance(work, list) or not work:
        return data
    total = 0.0
    any_years = False
    for w in work:
        if not isinstance(w, dict):
            continue
        y = w.get("years")
        if y is not None:
            try:
                total += float(y)
                any_years = True
            except (TypeError, ValueError):
                pass
    current = data.get("total_years_experience")
    try:
        current_f = float(current) if current is not None else 0.0
    except (TypeError, ValueError):
        current_f = 0.0
    if any_years and current_f <= 0:
        data["total_years_experience"] = round(min(total, 50.0), 1)
    return data
