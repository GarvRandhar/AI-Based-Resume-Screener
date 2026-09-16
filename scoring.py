"""ATS keyword overlap, embedding cosine similarity, brief match, and final score blend."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

# Default blend weights (must sum to 1.0). LLM weighted highest.
WEIGHT_ATS = 0.20
WEIGHT_EMBED = 0.30
WEIGHT_LLM = 0.50

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.#/-]{1,}")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can",
    "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "them", "their", "our", "your", "my", "me", "him",
    "her", "his", "its", "who", "whom", "which", "what", "where", "when",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "also", "into", "over", "after",
    "before", "between", "under", "again", "further", "then", "once",
    "here", "there", "any", "if", "because", "while", "during", "about",
    "against", "above", "below", "up", "down", "out", "off", "through",
    "resume", "curriculum", "vitae", "cv", "page", "email", "phone",
    "address", "www", "http", "https", "com", "need", "looking", "hire",
    "hiring", "role", "position", "candidate", "candidates", "team",
    "strong", "good", "preferred", "prefer", "required", "must", "nice",
}


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def criteria_text_blob(criteria: dict[str, Any]) -> str:
    """Ranking text = hiring brief only (prompt-first)."""
    prompt = (
        (criteria.get("hiring_prompt") or "").strip()
        or (criteria.get("job_description") or "").strip()
    )
    return prompt


def get_score_weights(criteria: dict[str, Any] | None) -> tuple[float, float, float]:
    """Return (w_ats, w_embed, w_llm) from criteria or defaults."""
    criteria = criteria or {}
    w = criteria.get("score_weights") or {}
    try:
        wa = float(w.get("ats", WEIGHT_ATS))
        we = float(w.get("embed", WEIGHT_EMBED))
        wl = float(w.get("llm", WEIGHT_LLM))
    except (TypeError, ValueError):
        return WEIGHT_ATS, WEIGHT_EMBED, WEIGHT_LLM
    total = wa + we + wl
    if total <= 0:
        return WEIGHT_ATS, WEIGHT_EMBED, WEIGHT_LLM
    return wa / total, we / total, wl / total


def get_min_final_score(criteria: dict[str, Any] | None) -> float:
    """Minimum final score (0–100) required to enter shortlist."""
    criteria = criteria or {}
    try:
        v = float(criteria.get("min_final_score") if criteria.get("min_final_score") is not None else 0)
    except (TypeError, ValueError):
        v = 0.0
    return max(0.0, min(100.0, v))


def resume_text_for_matching(
    structured: dict[str, Any] | None, raw_text: str | None
) -> str:
    """Domain-agnostic resume text for ATS + embeddings."""
    parts: list[str] = []
    if structured:
        if structured.get("summary"):
            parts.append(str(structured["summary"]))
        skills = structured.get("skills") or []
        if skills:
            parts.append("Skills: " + ", ".join(str(s) for s in skills))
        for key, label in (
            ("licenses", "Licenses"),
            ("certifications", "Certifications"),
        ):
            vals = structured.get(key) or []
            if vals:
                parts.append(f"{label}: " + ", ".join(str(c) for c in vals))
        for wh in structured.get("work_history") or []:
            if isinstance(wh, dict):
                parts.append(
                    f"{wh.get('title', '')} {wh.get('company', '')} {wh.get('duration', '')}"
                )
        for edu in structured.get("education") or []:
            if isinstance(edu, dict):
                parts.append(
                    f"{edu.get('degree', '')} {edu.get('field', '')} {edu.get('institution', '')}"
                )
        if structured.get("location"):
            parts.append(f"Location: {structured['location']}")
    if raw_text:
        parts.append(raw_text[:6000])
    return "\n".join(parts)


def ats_keyword_score(
    resume_text: str,
    criteria: dict[str, Any],
) -> float:
    """Keyword-overlap percentage (0–100) of brief tokens found in resume."""
    crit_tokens = tokenize(criteria_text_blob(criteria))
    if not crit_tokens:
        return 0.0
    resume_tokens = set(tokenize(resume_text))
    if not resume_tokens:
        return 0.0

    crit_set = set(crit_tokens)
    hits = sum(1 for t in crit_set if t in resume_tokens)
    pct = 100.0 * hits / max(len(crit_set), 1)

    crit_counts = Counter(crit_tokens)
    weighted_hits = 0.0
    weighted_total = 0.0
    for term, w in crit_counts.items():
        weight = 1.0 + math.log1p(w)
        weighted_total += weight
        if term in resume_tokens:
            weighted_hits += weight
    if weighted_total > 0:
        tf_pct = 100.0 * weighted_hits / weighted_total
        pct = 0.6 * pct + 0.4 * tf_pct

    return round(min(100.0, max(0.0, pct)), 2)


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity mapped to ~[0, 1]."""
    va = list(a) if a is not None else []
    vb = list(b) if b is not None else []
    if not va or not vb or len(va) != len(vb):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(va, vb):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    sim = dot / (math.sqrt(na) * math.sqrt(nb))
    sim = max(-1.0, min(1.0, sim))
    return round((sim + 1.0) / 2.0 if sim < 0 else sim, 4)


def combine_scores(
    ats_pct: float | None,
    embed_sim: float | None,
    llm_score: float | None,
    *,
    w_ats: float = WEIGHT_ATS,
    w_embed: float = WEIGHT_EMBED,
    w_llm: float = WEIGHT_LLM,
    criteria: dict[str, Any] | None = None,
) -> float:
    """Blend component scores into final 0–100. Criteria may override weights."""
    if criteria is not None:
        w_ats, w_embed, w_llm = get_score_weights(criteria)
    components: list[tuple[float, float]] = []
    if ats_pct is not None:
        components.append((w_ats, float(ats_pct)))
    if embed_sim is not None:
        components.append((w_embed, float(embed_sim) * 100.0))
    if llm_score is not None:
        components.append((w_llm, float(llm_score) * 10.0))

    if not components:
        return 0.0
    total_w = sum(w for w, _ in components)
    if total_w <= 0:
        return 0.0
    final = sum((w / total_w) * val for w, val in components)
    return round(min(100.0, max(0.0, final)), 2)


# ---------------------------------------------------------------------------
# Hiring brief quality checklist
# ---------------------------------------------------------------------------

_BRIEF_CHECKS = [
    (
        "years_or_seniority",
        re.compile(
            r"(?i)\b(\d+\+?\s*years?|senior|junior|mid[- ]?level|entry[- ]?level|"
            r"fresher|intern|lead|principal|manager)\b"
        ),
        "Years of experience or seniority (e.g. 3+ years, Senior, Fresher)",
    ),
    (
        "location",
        re.compile(
            r"(?i)\b(location|based in|remote|hybrid|onsite|on-site|relocat|"
            r"bangalore|bengaluru|mumbai|delhi|hyderabad|chennai|pune|noida|"
            r"gurgaon|kolkata|india|usa|uk|city)\b"
        ),
        "Location or work mode (city, remote, hybrid)",
    ),
    (
        "must_haves",
        re.compile(
            r"(?i)\b(must|required|mandatory|need|essential|critical|"
            r"should have|looking for)\b"
        ),
        "Must-have requirements language (must / required / need)",
    ),
    (
        "skills_or_tools",
        re.compile(
            r"(?i)\b(skill|experience with|proficient|knowledge of|tools?|"
            r"software|platform|crm|erp|excel|sap|salesforce|python|java|"
            r"nursing|sales|marketing|accounting|warehouse|license)\b"
        ),
        "Skills, tools, or domain capabilities",
    ),
    (
        "education_or_license",
        re.compile(
            r"(?i)\b(degree|b\.?tech|mba|b\.?sc|m\.?sc|diploma|graduate|"
            r"license|licence|certified|certification|rn|ca|cpa)\b"
        ),
        "Education or license/certification (if relevant)",
    ),
]


def analyze_hiring_brief(criteria: dict[str, Any] | str) -> dict[str, Any]:
    """
    Checklist so HR writes a stronger brief before ranking.
    Returns {score 0-100, checks: [{id, label, present}], suggestions: [...]}.
    """
    if isinstance(criteria, dict):
        text = criteria_text_blob(criteria)
    else:
        text = str(criteria or "")
    checks = []
    present_n = 0
    for cid, pat, label in _BRIEF_CHECKS:
        ok = bool(pat.search(text))
        if ok:
            present_n += 1
        checks.append({"id": cid, "label": label, "present": ok})
    score = round(100.0 * present_n / max(len(_BRIEF_CHECKS), 1), 0)
    suggestions = [
        c["label"] for c in checks if not c["present"]
    ]
    length_ok = len(text.strip()) >= 80
    if not length_ok:
        suggestions.insert(0, "Write a longer brief (at least a few sentences)")
        score = min(score, 40)
    return {
        "score": int(score),
        "length_ok": length_ok,
        "char_count": len(text.strip()),
        "checks": checks,
        "suggestions": suggestions,
        "ready": length_ok and present_n >= 3,
    }


# ---------------------------------------------------------------------------
# Matched brief points (explainability)
# ---------------------------------------------------------------------------

def extract_brief_points(criteria: dict[str, Any], *, max_points: int = 12) -> list[str]:
    """
    Pull concrete requirement-like phrases from the hiring brief for match bullets.
    """
    text = criteria_text_blob(criteria)
    if not text:
        return []
    points: list[str] = []
    # Bullet / numbered lines
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", ln)
        if m:
            p = m.group(1).strip()
            if 4 <= len(p) <= 120:
                points.append(p)
    # "Must: X, Y" / "Required: ..."
    for m in re.finditer(
        r"(?i)(?:must(?:-have)?s?|required|need(?:s)?|looking for)\s*[:\-]\s*(.+)",
        text,
    ):
        chunk = m.group(1).strip()
        for part in re.split(r"[,;]| and ", chunk):
            p = part.strip(" .")
            if 3 <= len(p) <= 80:
                points.append(p)
    # Significant multi-word phrases from tokens (fallback)
    if len(points) < 4:
        tokens = tokenize(text)
        # keep distinctive tokens as soft points
        for t in list(dict.fromkeys(tokens))[:max_points]:
            if len(t) >= 4:
                points.append(t)
    # dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for p in points:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
        if len(out) >= max_points:
            break
    return out


def match_brief_points(
    criteria: dict[str, Any],
    structured: dict[str, Any] | None,
    raw_text: str | None,
    *,
    max_points: int = 10,
) -> dict[str, Any]:
    """
    For each brief point, mark matched / missing against resume text + profile.
    """
    points = extract_brief_points(criteria, max_points=max_points)
    hay = resume_text_for_matching(structured, raw_text).lower()
    matched: list[str] = []
    missing: list[str] = []
    details: list[dict[str, Any]] = []
    for p in points:
        tokens = tokenize(p)
        if not tokens:
            # single short phrase
            ok = p.lower() in hay
        elif len(tokens) == 1:
            ok = tokens[0] in hay or tokens[0] in set(tokenize(hay))
        else:
            # match if most distinctive tokens appear
            hits = sum(1 for t in tokens if t in hay)
            ok = hits >= max(1, int(math.ceil(len(tokens) * 0.6)))
        details.append({"point": p, "matched": ok})
        (matched if ok else missing).append(p)
    total = len(points) or 1
    return {
        "matched": matched,
        "missing": missing,
        "details": details,
        "match_ratio": round(len(matched) / total, 3),
        "summary": (
            f"{len(matched)}/{len(points)} brief points evidenced in resume"
            if points
            else "No discrete brief points extracted"
        ),
    }


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def _norm_email(e: Any) -> str:
    return str(e or "").strip().lower()


def _norm_phone(p: Any) -> str:
    digits = re.sub(r"\D", "", str(p or ""))
    # last 10 digits for IN/US-style
    return digits[-10:] if len(digits) >= 10 else digits


def find_duplicates(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """
    Map resume_id → {is_duplicate, duplicate_of: [ids], reason}.
    Groups by email or phone when present.
    """
    by_email: dict[str, list[int]] = defaultdict(list)
    by_phone: dict[str, list[int]] = defaultdict(list)
    id_to_row: dict[int, dict] = {}

    for r in results:
        rid = r.get("resume_id")
        if rid is None:
            continue
        rid = int(rid)
        id_to_row[rid] = r
        structured = r.get("structured") if isinstance(r.get("structured"), dict) else {}
        email = _norm_email(structured.get("email") or r.get("email"))
        phone = _norm_phone(structured.get("phone") or r.get("phone"))
        if email and "@" in email:
            by_email[email].append(rid)
        if phone and len(phone) >= 10:
            by_phone[phone].append(rid)

    out: dict[int, dict[str, Any]] = {}
    for group_map, reason in ((by_email, "same email"), (by_phone, "same phone")):
        for key, ids in group_map.items():
            if len(ids) < 2:
                continue
            # Prefer highest final_score as primary
            ids_sorted = sorted(
                ids,
                key=lambda i: (
                    float(id_to_row[i].get("final_score") or 0),
                    -i,
                ),
                reverse=True,
            )
            primary = ids_sorted[0]
            for i in ids_sorted:
                prev = out.get(i, {"is_duplicate": False, "duplicate_of": [], "reasons": []})
                if i == primary:
                    others = [x for x in ids_sorted if x != i]
                    prev["duplicate_group"] = ids_sorted
                    prev["is_primary"] = True
                    prev["reasons"] = list(set(prev.get("reasons") or []) | {reason})
                    prev["duplicate_peers"] = others
                else:
                    prev["is_duplicate"] = True
                    prev["is_primary"] = False
                    dups = set(prev.get("duplicate_of") or [])
                    dups.add(primary)
                    prev["duplicate_of"] = list(dups)
                    prev["reasons"] = list(set(prev.get("reasons") or []) | {reason})
                out[i] = prev
    return out


def is_shortlist_eligible(
    *,
    rank: Optional[int],
    final_score: Optional[float],
    hard_filter_pass: Any,
    shortlist_size: int,
    min_final_score: float,
    is_duplicate: bool = False,
    exclude_duplicates: bool = True,
) -> bool:
    """Whether a candidate belongs in the shortlist."""
    try:
        passed = int(hard_filter_pass or 0) == 1
    except (TypeError, ValueError):
        passed = bool(hard_filter_pass)
    if not passed:
        return False
    if exclude_duplicates and is_duplicate:
        return False
    if rank is None:
        return False
    try:
        rank_i = int(rank)
    except (TypeError, ValueError):
        return False
    if rank_i > shortlist_size:
        return False
    try:
        score = float(final_score) if final_score is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0
    if score < min_final_score:
        return False
    return True
