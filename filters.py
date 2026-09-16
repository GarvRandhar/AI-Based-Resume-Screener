"""Hard filters against structured resume JSON — fuzzy, HR-friendly matching."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


# Example rule:
#   {"field": "degree", "operator": "contains", "value": "BTech", "label": "B.Tech required"}


# ---------------------------------------------------------------------------
# Normalization & synonym tables (degrees, common skills)
# ---------------------------------------------------------------------------

# Each group is a set of equivalent tokens after normalization (alnum-only lower).
DEGREE_SYNONYM_GROUPS: list[set[str]] = [
    {
        "btech", "btechnology", "bacheloroftechnology", "bachelorintech",
        "bacheloroftech", "btechcse", "btechcs", "btechece", "btechit",
        "be", "beng", "bachelorofengineering", "bachelorinengineering",
        "engineering", "engineer",  # broad HR intent: engineering degree
    },
    {
        "mtech", "mtechnology", "masteroftechnology", "masterintech",
        "me", "meng", "masterofengineering", "masterinengineering",
    },
    {
        "bsc", "bscience", "bachelorofscience", "bs", "bscit", "bsccs",
        "bsccomputer", "bsccomputerscience",
    },
    {
        "msc", "mscience", "masterofscience", "ms", "mscs", "mscit",
        "mscomputerscience",
    },
    {
        "ba", "bart", "bachelorofarts", "bachelorinarts",
    },
    {
        "ma", "masterofarts", "masterinarts",
    },
    {
        "mba", "masterofbusinessadministration", "masterofbusiness",
        "pgdm", "pgdmin",
    },
    {
        "mca", "masterofcomputerapplications", "masterofcomputerapplication",
    },
    {
        "bca", "bachelorofcomputerapplications", "bachelorofcomputerapplication",
    },
    {
        "bcom", "bachelorofcommerce", "bachelorincommerce",
    },
    {
        "mcom", "masterofcommerce",
    },
    {
        "phd", "phddoctor", "doctorofphilosophy", "doctorate",
    },
    {
        "diploma", "polytechnic",
    },
]

# Field-of-study / major synonyms (loose)
FIELD_SYNONYM_GROUPS: list[set[str]] = [
    {
        "computerscience", "computersciences", "compsci", "cse", "cs",
        "computerengineering", "computerengg", "informationtechnology",
        "it", "softwareengineering", "computing",
    },
    {
        "electronics", "ece", "electronicsandcommunication",
        "electronicscommunication", "eee", "electrical",
    },
    {
        "mechanical", "mech", "mechanicalengineering",
    },
    {
        "datascience", "dataanalytics", "analytics", "machinelearning", "ai",
        "artificialintelligence",
    },
    {
        "business", "management", "finance", "marketing",
    },
]

# Skill aliases: normalized key → set of aliases
SKILL_ALIASES: dict[str, set[str]] = {
    "python": {"python", "python3", "py"},
    "javascript": {"javascript", "js", "ecmascript"},
    "typescript": {"typescript", "ts"},
    "nodejs": {"nodejs", "node", "node.js"},
    "react": {"react", "reactjs", "react.js"},
    "java": {"java"},
    "csharp": {"csharp", "c#", "dotnet", ".net", "aspnet"},
    "cplusplus": {"cplusplus", "c++", "cpp"},
    "golang": {"golang", "go"},
    "kubernetes": {"kubernetes", "k8s"},
    "aws": {"aws", "amazonwebservices"},
    "gcp": {"gcp", "googlecloud", "googlecloudplatform"},
    "azure": {"azure", "microsoftazure"},
    "fastapi": {"fastapi"},
    "django": {"django"},
    "flask": {"flask"},
    "sql": {"sql", "mysql", "postgresql", "postgres", "mssql"},
    "machinelearning": {"machinelearning", "ml", "deeplearning", "ai"},
}


def _normalize_text(s: Any) -> str:
    """Lowercase, strip accents, keep alphanumerics only for fuzzy compare."""
    if s is None:
        return ""
    text = str(s).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # unify common separators
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _normalize_loose(s: Any) -> str:
    """Lowercase with spaces collapsed (for contains with words)."""
    if s is None:
        return ""
    text = str(s).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[.\-_/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_synonyms(token: str, groups: list[set[str]]) -> set[str]:
    """Return all synonyms for a normalized token, including itself."""
    n = _normalize_text(token)
    if not n:
        return set()
    out = {n}
    for group in groups:
        if n in group or any(n in g or g in n for g in group if len(g) >= 2):
            # if token is in group or closely contained
            if n in group:
                out |= group
            else:
                for g in group:
                    if len(n) >= 3 and (n in g or g in n):
                        out |= group
                        break
    return out


def _texts_match(actual: Any, expected: Any, *, mode: str = "contains") -> bool:
    """
    Fuzzy match for HR-friendly text comparison.
    mode: contains | eq
    """
    if expected is None or str(expected).strip() == "":
        return True
    if actual is None:
        return False

    needles = (
        [str(x).strip() for x in expected]
        if isinstance(expected, list)
        else [str(expected).strip()]
    )
    haystacks = (
        [str(x) for x in actual]
        if isinstance(actual, list)
        else [str(actual)]
    )
    # Also join list into one blob
    if isinstance(actual, list):
        haystacks.append(" ".join(str(x) for x in actual if x))

    for needle in needles:
        if not needle:
            continue
        n_norm = _normalize_text(needle)
        n_loose = _normalize_loose(needle)
        n_syn = _expand_synonyms(needle, DEGREE_SYNONYM_GROUPS) | _expand_synonyms(
            needle, FIELD_SYNONYM_GROUPS
        )
        # skill aliases
        for canon, aliases in SKILL_ALIASES.items():
            if n_norm in {_normalize_text(a) for a in aliases} or n_norm == canon:
                n_syn |= {_normalize_text(a) for a in aliases} | {canon}

        for hay in haystacks:
            if not hay:
                continue
            h_norm = _normalize_text(hay)
            h_loose = _normalize_loose(hay)
            h_syn = _expand_synonyms(hay, DEGREE_SYNONYM_GROUPS) | _expand_synonyms(
                hay, FIELD_SYNONYM_GROUPS
            )

            if mode == "eq":
                if n_norm and n_norm == h_norm:
                    return True
                if n_syn & h_syn:
                    return True
                if n_norm and (n_norm in h_norm or h_norm in n_norm) and min(len(n_norm), len(h_norm)) >= 3:
                    return True
            else:
                # contains — either direction after normalize
                if n_norm and n_norm in h_norm:
                    return True
                if h_norm and len(h_norm) >= 4 and h_norm in n_norm:
                    return True
                if n_loose and n_loose in h_loose:
                    return True
                # synonym overlap (BTech vs Bachelor of Technology vs Engineering)
                if n_syn & h_syn:
                    return True
                # token-level: each significant word of needle in hay
                words = [w for w in n_loose.split() if len(_normalize_text(w)) >= 3]
                if words and all(_normalize_text(w) in h_norm for w in words):
                    return True
    return False


def apply_hard_filters(
    structured: dict[str, Any] | None,
    rules: list[dict[str, Any]] | None,
    *,
    raw_text: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate all hard-filter rules with fuzzy matching.

    Returns:
      {
        "passed": bool,
        "results": [
          {"field", "operator", "value", "label", "passed", "actual", "reason"}
        ]
      }
    """
    rules = rules or []
    structured = structured or {}
    results: list[dict[str, Any]] = []

    if not rules:
        return {"passed": True, "results": []}

    for rule in rules:
        field = (rule.get("field") or "").strip()
        operator = (rule.get("operator") or "contains").strip().lower()
        value = rule.get("value")
        label = rule.get("label") or f"{field} {operator} {value}"

        actual = _get_actual(structured, field, raw_text=raw_text)
        passed, reason = _evaluate(field, operator, value, actual, structured)

        results.append(
            {
                "field": field,
                "operator": operator,
                "value": value,
                "label": label,
                "passed": passed,
                "actual": _serialize_actual(actual),
                "reason": reason,
            }
        )

    all_passed = all(r["passed"] for r in results)
    return {"passed": all_passed, "results": results}


def _serialize_actual(actual: Any) -> Any:
    if isinstance(actual, (str, int, float, bool)) or actual is None:
        return actual
    if isinstance(actual, list):
        return [str(x) for x in actual[:30]]
    return str(actual)


def _education_blob(structured: dict[str, Any]) -> list[str]:
    """All education-related strings for fuzzy degree/field matching."""
    parts: list[str] = []
    for e in structured.get("education") or []:
        if isinstance(e, dict):
            chunk = " ".join(
                str(e.get(k) or "")
                for k in ("degree", "field", "institution", "year")
            ).strip()
            if chunk:
                parts.append(chunk)
            for k in ("degree", "field", "institution"):
                if e.get(k):
                    parts.append(str(e[k]))
        elif e:
            parts.append(str(e))
    # summary sometimes mentions education
    if structured.get("summary"):
        parts.append(str(structured["summary"]))
    return parts


def _get_actual(
    structured: dict[str, Any],
    field: str,
    *,
    raw_text: str | None = None,
) -> Any:
    field = field.lower().replace(" ", "_")
    if field in ("years_experience", "total_years_experience", "experience"):
        return structured.get("total_years_experience")
    if field in ("location", "city", "geo"):
        # location + raw snippet for fuzzy city match
        vals = []
        if structured.get("location"):
            vals.append(structured["location"])
        return vals if vals else structured.get("location")
    if field in ("notice_period", "notice_period_days", "notice"):
        return structured.get("notice_period_days")
    if field in ("certification", "certifications", "certs"):
        # Certs + licenses for HR fuzzy match
        certs = list(structured.get("certifications") or [])
        licenses = list(structured.get("licenses") or [])
        return certs + licenses
    if field in ("license", "licenses", "licence", "licences"):
        licenses = list(structured.get("licenses") or [])
        certs = list(structured.get("certifications") or [])
        # Also search education blob for license mentions
        return licenses + certs + _education_blob(structured)
    if field in ("skill", "skills"):
        # Domain-agnostic: skills list + summary (no tech-only expansion)
        skills = list(structured.get("skills") or [])
        if structured.get("summary"):
            skills.append(str(structured["summary"]))
        return skills
    if field in ("degree", "education_degree"):
        # Prefer full education lines so "B.Tech CSE" / "Engineering" match
        return _education_blob(structured)
    if field in ("field_of_study", "education_field", "major"):
        return _education_blob(structured)
    if field in ("education", "education_contains", "institution"):
        return _education_blob(structured)
    if field in ("name", "email", "phone", "summary"):
        return structured.get(field)
    return structured.get(field)


def _evaluate(
    field: str,
    operator: str,
    expected: Any,
    actual: Any,
    structured: dict[str, Any],
) -> tuple[bool, str]:
    field_l = field.lower().replace(" ", "_")
    text_fields = {
        "degree",
        "education_degree",
        "field_of_study",
        "education_field",
        "major",
        "education",
        "education_contains",
        "institution",
        "skill",
        "skills",
        "certification",
        "certifications",
        "certs",
        "location",
        "city",
        "geo",
        "name",
        "summary",
    }

    try:
        if operator in ("gte", "min", ">="):
            a = _to_float(actual)
            e = _to_float(expected)
            if a is None:
                return False, f"Missing value for {field} (required ≥ {e})"
            ok = a >= e
            return (
                ok,
                f"{field}={a} ≥ {e}" if ok else f"{field}={a} is below required {e}",
            )

        if operator in ("lte", "max", "<="):
            a = _to_float(actual)
            e = _to_float(expected)
            if a is None:
                # Unknown notice period: do not auto-reject (HR-friendly)
                if field_l in ("notice_period", "notice_period_days", "notice"):
                    return True, f"{field} unknown — skipped (not auto-rejected)"
                return False, f"Missing value for {field} (required ≤ {e})"
            ok = a <= e
            return (
                ok,
                f"{field}={a} ≤ {e}" if ok else f"{field}={a} exceeds max {e}",
            )

        if operator in ("eq", "equals", "=="):
            if field_l in text_fields or isinstance(actual, (str, list)):
                ok = _texts_match(actual, expected, mode="eq") or _texts_match(
                    actual, expected, mode="contains"
                )
            else:
                ok = _soft_eq(actual, expected)
            return (
                ok,
                f"{field} matches {expected!r}"
                if ok
                else f"{field}={_short(actual)} does not match {expected!r}",
            )

        if operator in ("contains", "includes"):
            ok = _texts_match(actual, expected, mode="contains")
            return (
                ok,
                f"{field} matches {expected!r} (fuzzy)"
                if ok
                else f"{field} does not match {expected!r} (found: {_short(actual)})",
            )

        if operator in ("not_contains", "excludes"):
            ok = not _texts_match(actual, expected, mode="contains")
            return (
                ok,
                f"{field} excludes {expected!r}"
                if ok
                else f"{field} unexpectedly matches {expected!r}",
            )

        if operator in ("in", "one_of"):
            options = expected if isinstance(expected, list) else [
                x.strip() for x in str(expected).split(",") if x.strip()
            ]
            ok = any(_texts_match(actual, opt, mode="contains") for opt in options)
            return (
                ok,
                f"{field} is one of {options}"
                if ok
                else f"{field}={_short(actual)} not in {options}",
            )

        # Default: fuzzy contains
        ok = _texts_match(actual, expected, mode="contains")
        return (
            ok,
            f"{field} matches {expected!r}"
            if ok
            else f"{field} does not match {expected!r}",
        )
    except Exception as exc:
        return False, f"Filter error on {field}: {exc}"


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return None
    try:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v))
        if m:
            return float(m.group())
    except (TypeError, ValueError):
        pass
    return None


def _soft_eq(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a == b
    return _normalize_text(a) == _normalize_text(b)


def _short(v: Any, n: int = 100) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def rejection_summary(filter_result: dict[str, Any]) -> str:
    """Human-readable rejection reasons for logging / UI."""
    if filter_result.get("passed"):
        return ""
    fails = [
        r.get("reason") or r.get("label") or "failed rule"
        for r in filter_result.get("results") or []
        if not r.get("passed")
    ]
    return "; ".join(fails) if fails else "Failed hard filters"
