"""LEGACY Streamlit dashboard (superseded by frontend/ React UI).

Prefer: cd frontend && npm run dev
This file is kept only as a reference; Streamlit is no longer a dependency.
"""

from __future__ import annotations

import html
import io
import time
from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st

try:
    from config import get_settings

    _cfg = get_settings()
    API_BASE = f"http://{_cfg.get('api_host', '127.0.0.1')}:{int(_cfg.get('api_port', 8000))}"
except Exception:
    API_BASE = "http://127.0.0.1:8000"
POLL_SECONDS = 1.5

st.set_page_config(
    page_title="Resume Screener",
    page_icon=":material/person_search:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Light visual polish — layout uses native Streamlit (no fragile multi-card HTML)
CUSTOM_CSS = """
<style>
/* Soft ambient background */
.stApp {
  background:
    radial-gradient(1000px 520px at 8% -12%, rgba(99,102,241,0.16), transparent 55%),
    radial-gradient(720px 420px at 100% 0%, rgba(14,165,233,0.09), transparent 50%),
    radial-gradient(600px 300px at 50% 100%, rgba(139,92,246,0.05), transparent 50%),
    #0B0F19;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] {
  background: rgba(11,15,25,0.72);
  backdrop-filter: blur(10px);
}
.block-container {
  padding-top: 1.25rem !important;
  max-width: 1080px;
  padding-bottom: 3rem !important;
}

/* Primary CTA glow */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, #6366F1 0%, #4F46E5 55%, #4338CA 100%) !important;
  border: none !important;
  box-shadow: 0 4px 16px rgba(99,102,241,0.28);
  font-weight: 600 !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
  box-shadow: 0 6px 22px rgba(99,102,241,0.4);
}

/* Metric polish inside bordered cards */
div[data-testid="stMetric"] {
  background: transparent;
}
div[data-testid="stMetric"] label {
  color: #94A3B8 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-weight: 700 !important;
  color: #F1F5F9 !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(148,163,184,0.12);
}

/* Profile key-value */
.kv { margin: 0.12rem 0 0.5rem 0; }
.kv .k {
  font-size: 0.7rem;
  color: #94A3B8;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  font-weight: 600;
}
.kv .v {
  color: #F1F5F9;
  font-size: 0.95rem;
  margin-top: 0.08rem;
  line-height: 1.45;
  word-break: break-word;
}

.step-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  background: rgba(99,102,241,0.12);
  border: 1px solid rgba(99,102,241,0.25);
  color: #C7D2FE;
  font-size: 0.78rem;
  font-weight: 600;
  margin-right: 0.4rem;
  margin-bottom: 0.35rem;
}
.rank-medal {
  display: inline-block;
  width: 1.75rem;
  height: 1.75rem;
  line-height: 1.75rem;
  text-align: center;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.85rem;
  margin-right: 0.35rem;
}
.rank-1 { background: linear-gradient(135deg,#FBBF24,#F59E0B); color: #1C1917; }
.rank-2 { background: linear-gradient(135deg,#E2E8F0,#94A3B8); color: #0F172A; }
.rank-3 { background: linear-gradient(135deg,#FDBA74,#EA580C); color: #1C1917; }
.rank-n { background: rgba(99,102,241,0.2); color: #C7D2FE; border: 1px solid rgba(99,102,241,0.35); }
.score-chip {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 6px;
  background: rgba(34,197,94,0.12);
  border: 1px solid rgba(34,197,94,0.28);
  color: #86EFAC;
  font-weight: 700;
  font-size: 0.9rem;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def api(
    method: str,
    path: str,
    *,
    json_body: Any = None,
    files: Any = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    url = f"{API_BASE}{path}"
    try:
        r = requests.request(method, url, json=json_body, files=files, timeout=timeout)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"API {r.status_code}: {detail}")
        return r.json()
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"API offline at {API_BASE}. In another terminal run: python app.py"
        ) from exc


def safe_health() -> Optional[dict[str, Any]]:
    try:
        return api("GET", "/health", timeout=4.0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults = {
        "view": "setup",
        "active_job_id": None,
        "hiring_prompt": "",
        "drive_name": "Hiring Drive",
        "concurrency": 2,
        "bulk_enabled": True,
        "bulk_llm_top_k": 30,
        "bulk_min_pre_score": 25.0,
        "bulk_always_full_below": 12,
        "hard_filters": [],
        "shortlist_size": 5,
        "min_final_score": 0.0,
        "exclude_duplicates": True,
        "w_ats": 20,
        "w_embed": 30,
        "w_llm": 50,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()
inject_css()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_num(v: Any, digits: int = 1) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def public_profile(structured: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(structured, dict):
        return {}
    return {k: v for k, v in structured.items() if not str(k).startswith("_")}


def format_education(edu: Any) -> str:
    if not edu:
        return "—"
    if not isinstance(edu, list):
        return str(edu)
    lines = []
    for e in edu:
        if not isinstance(e, dict):
            if e:
                lines.append(str(e))
            continue
        degree = (e.get("degree") or "").strip()
        field = (e.get("field") or "").strip()
        inst = (e.get("institution") or "").strip()
        year = e.get("year")
        if not degree and not field and not inst:
            continue
        parts = [p for p in (degree, field, inst) if p]
        line = " · ".join(parts)
        if year not in (None, "", "null", "NULL"):
            line = f"{line} ({year})"
        if line and line not in lines:
            lines.append(line)
    return "; ".join(lines) if lines else "—"


def format_list(items: Any, limit: int = 20) -> str:
    if not items:
        return "—"
    if isinstance(items, str):
        return items
    if isinstance(items, list):
        clean = [str(x).strip() for x in items if str(x).strip()]
        if not clean:
            return "—"
        if len(clean) > limit:
            return ", ".join(clean[:limit]) + f" (+{len(clean) - limit} more)"
        return ", ".join(clean)
    return str(items)


def render_trust_warnings(structured: Optional[dict[str, Any]]) -> None:
    """Show extraction confidence flags so HR does not trust bad fields blindly."""
    if not isinstance(structured, dict):
        return
    ext = structured.get("_extraction") or {}
    warnings = list(ext.get("trust_warnings") or [])
    conf = ext.get("field_confidence") or structured.get("_confidence") or {}
    if conf.get("name") == "low" and not any("Name" in w for w in warnings):
        warnings.append("Name is low-confidence — verify identity.")
    if conf.get("location") == "low" and not any("Location" in w for w in warnings):
        warnings.append("Location is low-confidence or was cleared.")
    if not structured.get("name"):
        warnings.append("No reliable name extracted — check resume file name.")
    if not warnings:
        overall = ext.get("overall") or "medium"
        if overall == "high":
            st.badge("Extraction: high confidence", icon=":material/verified:", color="green")
        elif overall == "medium":
            st.badge("Extraction: medium — spot-check", icon=":material/info:", color="blue")
        return
    st.warning(
        "**Trust flags — review before deciding**\n\n- " + "\n- ".join(warnings[:8]),
        icon=":material/warning:",
    )


def render_profile_fields(pub: dict[str, Any]) -> None:
    """HR-friendly profile (no raw JSON)."""
    rows = [
        ("Name", pub.get("name") or "—"),
        ("Email", pub.get("email") or "—"),
        ("Phone", pub.get("phone") or "—"),
        ("Location", pub.get("location") or "—"),
        (
            "Years of experience",
            pub.get("total_years_experience")
            if pub.get("total_years_experience") is not None
            else "—",
        ),
        (
            "Notice period (days)",
            pub.get("notice_period_days")
            if pub.get("notice_period_days") is not None
            else "—",
        ),
        ("Skills / strengths", format_list(pub.get("skills"))),
        ("Licenses", format_list(pub.get("licenses"))),
        ("Certifications", format_list(pub.get("certifications"))),
        ("Education", format_education(pub.get("education"))),
    ]
    for label, value in rows:
        st.markdown(
            f'<div class="kv"><div class="k">{html.escape(str(label))}</div>'
            f'<div class="v">{html.escape(str(value))}</div></div>',
            unsafe_allow_html=True,
        )
    work = pub.get("work_history") or []
    if isinstance(work, list) and work:
        st.markdown(
            '<div class="kv"><div class="k">Work history</div></div>',
            unsafe_allow_html=True,
        )
        for w in work[:6]:
            if isinstance(w, dict):
                title = w.get("title") or ""
                company = w.get("company") or ""
                duration = w.get("duration") or ""
                st.caption(
                    f"• {title} @ {company} ({duration})".replace(" @  ", " ").strip(" @()")
                )
            else:
                st.caption(f"• {w}")
    summary = (pub.get("summary") or "").strip()
    if summary:
        st.markdown(
            f'<div class="kv"><div class="k">Summary</div>'
            f'<div class="v">{html.escape(summary)}</div></div>',
            unsafe_allow_html=True,
        )


def _rank_medal_html(rank: int) -> str:
    cls = {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(int(rank), "rank-n")
    return f'<span class="rank-medal {cls}">{int(rank)}</span>'


def render_header() -> None:
    health = safe_health()
    c1, c2 = st.columns([4.2, 1.4], vertical_alignment="center")
    with c1:
        st.markdown("### :material/person_search: Resume Screener")
        st.caption("Paste a hiring brief · upload resumes · get a ranked shortlist")
    with c2:
        if health is None:
            st.badge("API offline", icon=":material/cloud_off:", color="red")
            st.caption("Run `python app.py`")
        else:
            ollama = health.get("ollama") or {}
            if health.get("ready"):
                st.badge("System ready", icon=":material/check_circle:", color="green")
            elif ollama.get("ok"):
                missing = []
                if not ollama.get("has_extract_model"):
                    missing.append(ollama.get("extract_model") or "LLM")
                if not ollama.get("has_embed_model"):
                    missing.append(ollama.get("embed_model") or "embed")
                st.badge("Models missing", icon=":material/download:", color="orange")
                st.caption("Pull " + ", ".join(missing))
            else:
                st.badge("Ollama down", icon=":material/warning:", color="orange")
                st.caption((ollama.get("error") or "")[:48])

    if health and not health.get("ready"):
        ollama = health.get("ollama") or {}
        cfg = health.get("config") or {}
        with st.expander("System health / models", expanded=True, icon=":material/monitor_heart:"):
            st.write(
                {
                    "api": API_BASE,
                    "ollama": ollama.get("base"),
                    "extract_model": cfg.get("extract_model"),
                    "embed_model": cfg.get("embed_model"),
                    "has_extract": ollama.get("has_extract_model"),
                    "has_embed": ollama.get("has_embed_model"),
                    "config": cfg.get("config_path"),
                }
            )
            if not ollama.get("has_extract_model"):
                st.code(f"ollama pull {cfg.get('extract_model') or 'llama3.1:8b'}")
            if not ollama.get("has_embed_model"):
                st.code(f"ollama pull {cfg.get('embed_model') or 'nomic-embed-text'}")


# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------


def view_setup() -> None:
    st.markdown("#### Start a screening")
    st.caption(
        "Describe who you want to hire, upload resumes, and get rankings on this page."
    )
    st.markdown(
        '<span class="step-pill">1 · Brief</span>'
        '<span class="step-pill">2 · Upload</span>'
        '<span class="step-pill">3 · Rank</span>',
        unsafe_allow_html=True,
    )

    try:
        jobs = api("GET", "/jobs").get("jobs") or []
    except Exception as exc:
        st.error(str(exc), icon=":material/error:")
        jobs = []

    if jobs:
        with st.expander("Open a previous run", icon=":material/history:"):
            opts = {
                j["id"]: (
                    f"#{j['id']} · {j.get('name')} · {j.get('status')} · "
                    f"{j.get('processed_count', 0)}/{j.get('total_count', 0)}"
                )
                for j in jobs[:20]
            }
            pick = st.selectbox(
                "Previous job",
                options=list(opts.keys()),
                format_func=lambda i: opts[i],
            )
            if st.button(
                "Open results",
                icon=":material/open_in_new:",
                width="stretch",
            ):
                st.session_state.active_job_id = pick
                st.session_state.view = "results"
                st.rerun()

    # Drive templates
    try:
        templates = api("GET", "/templates").get("templates") or []
    except Exception:
        templates = []
    with st.expander("Drive templates (save / load brief + settings)", icon=":material/bookmark:"):
        if templates:
            tnames = {t["id"]: t["name"] for t in templates}
            tid = st.selectbox(
                "Load template",
                options=[None] + list(tnames.keys()),
                format_func=lambda x: "— none —" if x is None else tnames[x],
            )
            if tid is not None and st.button(
                "Load template into form",
                icon=":material/file_open:",
            ):
                t = api("GET", f"/templates/{tid}")["template"]
                crit = t.get("criteria") or {}
                st.session_state.hiring_prompt = (
                    crit.get("hiring_prompt") or crit.get("job_description") or ""
                )
                st.session_state.shortlist_size = int(crit.get("shortlist_size") or 5)
                st.session_state.min_final_score = float(crit.get("min_final_score") or 0)
                st.session_state.hard_filters = list(crit.get("hard_filters") or [])
                sw = crit.get("score_weights") or {}
                if sw:
                    st.session_state.w_ats = int(float(sw.get("ats", 0.2)) * 100)
                    st.session_state.w_embed = int(float(sw.get("embed", 0.3)) * 100)
                    st.session_state.w_llm = int(float(sw.get("llm", 0.5)) * 100)
                st.success(f"Loaded “{t['name']}”", icon=":material/check:")
                st.rerun()
        tname = st.text_input("Save current form as template", key="save_tpl_name")
        if st.button("Save template", icon=":material/save:"):
            if not tname.strip():
                st.warning("Template name required", icon=":material/warning:")
            else:
                wt = max(
                    1,
                    int(st.session_state.w_ats)
                    + int(st.session_state.w_embed)
                    + int(st.session_state.w_llm),
                )
                snap = {
                    "hiring_prompt": st.session_state.hiring_prompt or "",
                    "job_description": st.session_state.hiring_prompt or "",
                    "shortlist_size": int(st.session_state.shortlist_size),
                    "min_final_score": float(st.session_state.min_final_score),
                    "exclude_duplicates_from_shortlist": bool(
                        st.session_state.exclude_duplicates
                    ),
                    "score_weights": {
                        "ats": int(st.session_state.w_ats) / wt,
                        "embed": int(st.session_state.w_embed) / wt,
                        "llm": int(st.session_state.w_llm) / wt,
                    },
                    "hard_filters": st.session_state.hard_filters or [],
                    "bulk_mode": {
                        "enabled": bool(st.session_state.bulk_enabled),
                        "llm_top_k": int(st.session_state.bulk_llm_top_k),
                        "min_pre_score": float(st.session_state.bulk_min_pre_score),
                        "always_full_below": int(st.session_state.bulk_always_full_below),
                    },
                }
                try:
                    api(
                        "POST",
                        "/templates",
                        json_body={"name": tname.strip(), "criteria": snap},
                    )
                    st.success("Template saved", icon=":material/check:")
                except Exception as exc:
                    st.error(str(exc), icon=":material/error:")

    with st.container(border=True):
        st.markdown("##### Job settings")
        n1, n2, n3 = st.columns([2, 1, 1])
        with n1:
            st.session_state.drive_name = st.text_input(
                "Job name",
                value=st.session_state.drive_name,
                placeholder="e.g. Backend hire — March",
            )
        with n2:
            st.session_state.shortlist_size = st.number_input(
                "Shortlist size (top N)",
                min_value=1,
                max_value=200,
                value=int(st.session_state.shortlist_size),
                help="Only the top N ranked candidates are marked Shortlisted.",
            )
        with n3:
            st.session_state.min_final_score = st.number_input(
                "Min score to shortlist",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.min_final_score),
                step=5.0,
                help="Candidates below this final score cannot enter the shortlist (0 = off).",
            )

        st.session_state.hiring_prompt = st.text_area(
            "Hiring brief (required)",
            value=st.session_state.hiring_prompt,
            height=220,
            placeholder=(
                "Works for any drive — tech, sales, clinical, operations, campus…\n\n"
                "Example (sales):\n"
                "B2B Account Executive, Mumbai, 3+ years enterprise sales.\n"
                "Must: quota attainment, CRM (Salesforce/HubSpot), strong communication.\n"
                "Prefer MBA or business degree. Notice under 60 days.\n\n"
                "Example (clinical):\n"
                "Staff Nurse (ICU), Bengaluru. Valid RN license required.\n"
                "2+ years hospital experience, night shifts OK."
            ),
            help="Primary ranking signal for every candidate. Domain-agnostic.",
        )

        # Brief quality checklist
        try:
            from scoring import analyze_hiring_brief

            brief_q = analyze_hiring_brief(st.session_state.hiring_prompt or "")
            ready = bool(brief_q.get("ready"))
            qscore = brief_q.get("score") or 0
            label = (
                f"Hiring brief checklist · quality {qscore}/100"
                + (" · ready" if ready else " · add more detail")
            )
            with st.expander(label, expanded=not ready, icon=":material/checklist:"):
                for c in brief_q.get("checks") or []:
                    if c.get("present"):
                        st.markdown(f":green-badge[✓] {c.get('label')}")
                    else:
                        st.markdown(f":gray-badge[○] {c.get('label')}")
                if brief_q.get("suggestions"):
                    st.markdown("**Suggestions to improve ranking**")
                    for s in brief_q["suggestions"]:
                        st.caption(f"• {s}")
        except Exception:
            pass

        uploads = st.file_uploader(
            "Resumes (PDF, DOCX, TXT, or ZIP)",
            type=["pdf", "docx", "doc", "txt", "zip"],
            accept_multiple_files=True,
            help="Upload individual files or a ZIP of resumes.",
        )
        if uploads:
            st.badge(
                f"{len(uploads)} file(s) selected",
                icon=":material/attach_file:",
                color="blue",
            )

    with st.expander(
        "Optional hard rules (only if HR needs a hard cut)",
        icon=":material/filter_alt:",
    ):
        st.caption(
            "Leave empty for pure prompt ranking. Use for must-pass gates like "
            "min years or a required license (fuzzy match: RN ≈ Registered Nurse)."
        )
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            hard_field = st.selectbox(
                "Field",
                options=[
                    "years_experience",
                    "license",
                    "certification",
                    "degree",
                    "skill",
                    "location",
                    "notice_period_days",
                    "education",
                ],
                format_func=lambda x: {
                    "years_experience": "Years of experience",
                    "license": "License / registration",
                    "certification": "Certification",
                    "degree": "Degree",
                    "skill": "Skill / capability",
                    "location": "Location",
                    "notice_period_days": "Notice period (days)",
                    "education": "Education (text)",
                }.get(x, x),
                key="hard_field_sel",
            )
        with hc2:
            hard_op = st.selectbox(
                "Operator",
                options=["gte", "lte", "contains", "eq", "in"],
                format_func=lambda x: {
                    "gte": "≥ minimum",
                    "lte": "≤ maximum",
                    "contains": "contains (fuzzy)",
                    "eq": "equals (fuzzy)",
                    "in": "one of (comma list)",
                }.get(x, x),
                key="hard_op_sel",
            )
        with hc3:
            hard_val = st.text_input("Value", key="hard_val_in", placeholder="e.g. 3 or RN")
        if st.button("Add hard rule", key="add_hard_rule", icon=":material/add:"):
            if not str(hard_val).strip():
                st.warning("Enter a value.", icon=":material/warning:")
            else:
                val: Any = hard_val.strip()
                if hard_field in ("years_experience", "notice_period_days") and hard_op in (
                    "gte",
                    "lte",
                    "eq",
                ):
                    try:
                        val = float(val) if "." in val else int(val)
                    except ValueError:
                        pass
                st.session_state.hard_filters.append(
                    {
                        "field": hard_field,
                        "operator": hard_op,
                        "value": val,
                        "label": f"{hard_field} {hard_op} {val}",
                    }
                )
                st.rerun()
        if st.session_state.hard_filters:
            for i, rule in enumerate(list(st.session_state.hard_filters)):
                c_a, c_b = st.columns([6, 1])
                c_a.caption(
                    f"• {rule.get('label')} (`{rule.get('field')}` {rule.get('operator')} {rule.get('value')})"
                )
                if c_b.button("Remove", key=f"rm_hf_{i}", icon=":material/close:"):
                    st.session_state.hard_filters.pop(i)
                    st.rerun()
        else:
            st.caption("No hard rules — recommended for most drives.")

    with st.expander("Advanced engine options", icon=":material/tune:"):
        st.session_state.concurrency = st.slider(
            "LLM concurrency", 1, 4, int(st.session_state.concurrency)
        )
        st.markdown("**Score weights** (relative; normalized automatically)")
        w1, w2, w3 = st.columns(3)
        with w1:
            st.session_state.w_ats = st.slider(
                "ATS keywords %", 0, 100, int(st.session_state.w_ats)
            )
        with w2:
            st.session_state.w_embed = st.slider(
                "Embedding similarity", 0, 100, int(st.session_state.w_embed)
            )
        with w3:
            st.session_state.w_llm = st.slider(
                "LLM fit score", 0, 100, int(st.session_state.w_llm)
            )
        st.session_state.exclude_duplicates = st.checkbox(
            "Exclude duplicate email/phone from shortlist (keep highest score)",
            value=bool(st.session_state.exclude_duplicates),
        )
        st.session_state.bulk_enabled = st.checkbox(
            "Fast bulk mode (LLM only top candidates)",
            value=bool(st.session_state.bulk_enabled),
        )
        if st.session_state.bulk_enabled:
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.bulk_llm_top_k = st.number_input(
                    "LLM top-K", 1, 500, int(st.session_state.bulk_llm_top_k)
                )
            with c2:
                st.session_state.bulk_min_pre_score = st.number_input(
                    "Min pre-score",
                    0.0,
                    100.0,
                    float(st.session_state.bulk_min_pre_score),
                )

    st.space("small")
    if not st.button(
        "Rank resumes",
        type="primary",
        width="stretch",
        icon=":material/play_arrow:",
    ):
        return

    prompt = (st.session_state.hiring_prompt or "").strip()
    if len(prompt) < 20:
        st.error(
            "Please paste a fuller hiring brief (role, skills, must-haves).",
            icon=":material/error:",
        )
        return
    if not uploads:
        st.error(
            "Please upload at least one resume or ZIP.",
            icon=":material/error:",
        )
        return

    wt = max(
        1,
        int(st.session_state.w_ats)
        + int(st.session_state.w_embed)
        + int(st.session_state.w_llm),
    )
    criteria = {
        "hiring_prompt": prompt,
        "job_description": prompt,
        "shortlist_size": int(st.session_state.shortlist_size),
        "min_final_score": float(st.session_state.min_final_score),
        "exclude_duplicates_from_shortlist": bool(st.session_state.exclude_duplicates),
        "score_weights": {
            "ats": int(st.session_state.w_ats) / wt,
            "embed": int(st.session_state.w_embed) / wt,
            "llm": int(st.session_state.w_llm) / wt,
        },
        "hard_filters": st.session_state.hard_filters or [],
        "bulk_mode": {
            "enabled": bool(st.session_state.bulk_enabled),
            "llm_top_k": int(st.session_state.bulk_llm_top_k),
            "min_pre_score": float(st.session_state.bulk_min_pre_score),
            "always_full_below": int(st.session_state.bulk_always_full_below),
        },
    }

    try:
        with st.spinner("Creating job…"):
            resp = api(
                "POST",
                "/jobs",
                json_body={
                    "name": st.session_state.drive_name or "Hiring Drive",
                    "criteria": criteria,
                    "concurrency": st.session_state.concurrency,
                },
            )
            job_id = resp["job"]["id"]

        files_payload = [
            ("files", (uf.name, uf.getvalue(), uf.type or "application/octet-stream"))
            for uf in uploads
        ]
        with st.spinner(f"Uploading {len(uploads)} file(s)…"):
            up = api(
                "POST",
                f"/jobs/{job_id}/upload",
                files=files_payload,
                timeout=300.0,
            )

        with st.spinner("Starting ranking…"):
            api(
                "POST",
                f"/jobs/{job_id}/start",
                json_body={"concurrency": st.session_state.concurrency},
            )

        st.session_state.active_job_id = job_id
        st.session_state.view = "results"
        st.success(
            f"Started · {up.get('resumes_registered', 0)} resumes · job #{job_id}",
            icon=":material/rocket_launch:",
        )
        time.sleep(0.3)
        st.rerun()
    except Exception as exc:
        st.error(str(exc), icon=":material/error:")


# ---------------------------------------------------------------------------
# RESULTS — native Streamlit only
# ---------------------------------------------------------------------------


def view_results() -> None:
    job_id = st.session_state.active_job_id
    if not job_id:
        st.session_state.view = "setup"
        st.rerun()
        return

    head_l, head_r = st.columns([3.5, 1.2], vertical_alignment="center")
    with head_l:
        st.markdown(f"#### Results · Job #{job_id}")
        st.caption("Ranked only against your hiring brief.")
    with head_r:
        if st.button(
            "New screening",
            icon=":material/arrow_back:",
            width="stretch",
        ):
            st.session_state.view = "setup"
            st.session_state.active_job_id = None
            st.rerun()

    auto = st.toggle("Auto-refresh while running", value=True)

    try:
        progress = api("GET", f"/jobs/{job_id}/progress")
        payload = api("GET", f"/jobs/{job_id}/results")
    except Exception as exc:
        st.error(str(exc), icon=":material/error:")
        return

    job = payload.get("job") or {}
    results = payload.get("results") or []
    total = progress.get("total_count") or 0
    done = progress.get("processed_count") or 0
    status = (progress.get("status") or job.get("status") or "—").lower()
    pct = (done / total) if total else 0.0
    is_running = bool(
        progress.get("is_running")
        or (status in ("processing", "pending") and done < total)
    )

    if status == "completed":
        status_line = "Ranking complete"
        status_color = "green"
        status_icon = ":material/check_circle:"
    elif status == "failed":
        status_line = "Ranking failed"
        status_color = "red"
        status_icon = ":material/error:"
    elif status == "paused":
        status_line = "Paused"
        status_color = "orange"
        status_icon = ":material/pause_circle:"
    elif is_running:
        status_line = "Ranking in progress…"
        status_color = "blue"
        status_icon = ":material/pending:"
    else:
        status_line = status.title()
        status_color = "gray"
        status_icon = ":material/info:"

    stage = progress.get("current_stage") or job.get("current_stage") or ""
    cur_file = progress.get("current_filename") or job.get("current_filename") or ""
    prog_extra = ""
    if is_running or status in ("processing", "paused"):
        if stage:
            prog_extra += f" · stage: {stage}"
        if cur_file:
            prog_extra += f" · file: {cur_file}"

    with st.container(border=True):
        st.badge(status_line, icon=status_icon, color=status_color)
        st.progress(
            min(1.0, pct),
            text=f"{done} / {total} processed  ·  {pct*100:.0f}%{prog_extra}",
        )
        with st.container(horizontal=True, gap="small"):
            if st.button(
                "Pause",
                icon=":material/pause:",
                disabled=not is_running,
                key="ctrl_pause",
            ):
                try:
                    api("POST", f"/jobs/{job_id}/pause")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if st.button("Resume", icon=":material/play_arrow:", key="ctrl_resume"):
                try:
                    api("POST", f"/jobs/{job_id}/resume")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if st.button("Cancel", icon=":material/stop:", key="ctrl_cancel"):
                try:
                    api("POST", f"/jobs/{job_id}/cancel")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if st.button(
                "Re-run failed only",
                icon=":material/replay:",
                key="ctrl_rerun",
            ):
                try:
                    r = api(
                        "POST",
                        f"/jobs/{job_id}/rerun-failed",
                        json_body={"concurrency": st.session_state.concurrency},
                    )
                    st.success(
                        f"Re-running {r.get('count', 0)} failed resume(s)",
                        icon=":material/replay:",
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    # Errors report
    try:
        err_payload = api("GET", f"/jobs/{job_id}/errors")
        if err_payload.get("count"):
            with st.expander(
                f"Error report ({err_payload['count']} failed)",
                icon=":material/bug_report:",
            ):
                for e in err_payload.get("errors") or []:
                    st.caption(
                        f"#{e.get('resume_id')} · {e.get('filename')} · "
                        f"{e.get('stage') or ''} · {e.get('error_message') or ''}"
                    )
    except Exception:
        pass

    with st.expander(
        "Shortlist settings (edit without re-running)",
        icon=":material/tune:",
    ):
        ss1, ss2, ss3 = st.columns(3)
        with ss1:
            new_n = st.number_input(
                "Shortlist size",
                1,
                500,
                int(
                    (results[0].get("shortlist_size") if results else None)
                    or (job.get("criteria") or {}).get("shortlist_size")
                    or 5
                ),
                key=f"post_sl_{job_id}",
            )
        with ss2:
            new_min = st.number_input(
                "Min final score",
                0.0,
                100.0,
                float(
                    (results[0].get("min_final_score") if results else None)
                    or (job.get("criteria") or {}).get("min_final_score")
                    or 0
                ),
                step=5.0,
                key=f"post_min_{job_id}",
            )
        with ss3:
            st.write("")
            st.write("")
            if st.button(
                "Apply shortlist settings",
                width="stretch",
                icon=":material/check:",
            ):
                try:
                    api(
                        "PATCH",
                        f"/jobs/{job_id}/shortlist-settings",
                        json_body={
                            "shortlist_size": int(new_n),
                            "min_final_score": float(new_min),
                        },
                    )
                    st.success("Shortlist updated", icon=":material/check:")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    # Export pack
    try:
        pack = api("GET", f"/jobs/{job_id}/export-pack")
        import json as _json

        st.download_button(
            "Download export pack (JSON)",
            data=_json.dumps(pack, indent=2),
            file_name=f"export_pack_job_{job_id}.json",
            mime="application/json",
            icon=":material/download:",
        )
    except Exception:
        pass

    # Resolve shortlist size
    shortlist_size = 5
    try:
        if results and results[0].get("shortlist_size") is not None:
            shortlist_size = int(results[0]["shortlist_size"])
        elif isinstance(job.get("criteria"), dict) and job["criteria"].get(
            "shortlist_size"
        ) is not None:
            shortlist_size = int(job["criteria"]["shortlist_size"])
    except (TypeError, ValueError):
        shortlist_size = 5
    shortlist_size = max(1, min(500, shortlist_size))

    min_final = 0.0
    try:
        if results and results[0].get("min_final_score") is not None:
            min_final = float(results[0]["min_final_score"])
        elif isinstance(job.get("criteria"), dict):
            min_final = float(job["criteria"].get("min_final_score") or 0)
    except (TypeError, ValueError):
        min_final = 0.0

    def _is_shortlisted(r: dict[str, Any]) -> bool:
        """Prefer server flag; recompute if needed."""
        if "shortlisted" in r and r.get("rank") is not None:
            if r.get("shortlisted") is True:
                return True
            if r.get("shortlisted") is False and (
                r.get("is_duplicate") or r.get("below_min_score")
            ):
                return False
        try:
            from scoring import is_shortlist_eligible

            rank_i = int(r["rank"]) if r.get("rank") is not None else None
            return is_shortlist_eligible(
                rank=rank_i,
                final_score=r.get("final_score"),
                hard_filter_pass=r.get("hard_filter_pass"),
                shortlist_size=shortlist_size,
                min_final_score=min_final,
                is_duplicate=bool(r.get("is_duplicate")),
                exclude_duplicates=True,
            )
        except Exception:
            try:
                hf = int(r.get("hard_filter_pass") or 0) == 1
                rank_i = int(r["rank"]) if r.get("rank") is not None else None
                score = float(r.get("final_score") or 0)
                return bool(
                    hf
                    and rank_i is not None
                    and rank_i <= shortlist_size
                    and score >= min_final
                    and not r.get("is_duplicate")
                )
            except Exception:
                return False

    for r in results:
        r["shortlisted"] = _is_shortlisted(r)
        r["shortlist_size"] = shortlist_size

    shortlisted = [r for r in results if r.get("shortlisted")]
    scored = sum(
        1
        for r in results
        if (int(r.get("hard_filter_pass") or 0) == 1)
        or r.get("hard_filter_pass") is True
    )
    dup_count = sum(1 for r in results if r.get("is_duplicate"))
    rejected = sum(1 for r in results if r.get("resume_status") == "rejected")
    failed = sum(1 for r in results if r.get("resume_status") == "failed")
    top_score = max(
        (float(r["final_score"]) for r in results if r.get("final_score") is not None),
        default=0.0,
    )

    # KPI strip
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", status.title(), border=True)
    m2.metric("Processed", f"{done}/{total}", border=True)
    m3.metric("Shortlisted", f"{len(shortlisted)} / {shortlist_size}", border=True)
    m4.metric("Top score", f"{top_score:.0f}", border=True)

    st.caption(
        f"All {scored} scored candidates are ranked. "
        f"Shortlist = top **{shortlist_size}**"
        + (f" with Final ≥ **{min_final:.0f}**" if min_final > 0 else "")
        + " (duplicates excluded)."
        + (f" · Duplicates: {dup_count}" if dup_count else "")
        + (f" · Rejected: {rejected}" if rejected else "")
        + (f" · Failed parse: {failed}" if failed else "")
    )
    if job.get("error_message"):
        st.error(job["error_message"], icon=":material/error:")

    shortlisted_sorted = sorted(
        shortlisted,
        key=lambda x: (x.get("rank") is None, x.get("rank") or 999),
    )

    tab_rank, tab_review = st.tabs(
        [
            ":material/leaderboard: Ranking & shortlist",
            ":material/person: Candidate review",
        ]
    )

    # ---- RANKING TAB ----
    with tab_rank:
        if shortlisted_sorted:
            st.markdown(f"##### Shortlist (top {shortlist_size})")
            show_n = min(3, len(shortlisted_sorted))
            cols = st.columns(show_n)
            for i, r in enumerate(shortlisted_sorted[:show_n]):
                with cols[i]:
                    with st.container(border=True, height="stretch"):
                        rank = r.get("rank") or (i + 1)
                        st.markdown(
                            f'{_rank_medal_html(int(rank))}'
                            f'<span style="color:#94A3B8;font-size:0.8rem;font-weight:600;">'
                            f"SHORTLISTED</span>",
                            unsafe_allow_html=True,
                        )
                        name = r.get("candidate_name") or r.get("filename") or "Unknown"
                        st.markdown(f"**{name}**")
                        if r.get("source_filename") or r.get("filename"):
                            st.caption(
                                f":material/description: "
                                f"{r.get('source_filename') or r.get('filename')}"
                            )
                        score_val = fmt_num(r.get("final_score"), 1)
                        st.markdown(
                            f'<span class="score-chip">{html.escape(score_val)}</span>'
                            f' <span style="color:#94A3B8;font-size:0.8rem;">final</span>',
                            unsafe_allow_html=True,
                        )
                        c_a, c_b, c_c = st.columns(3)
                        c_a.caption(f"ATS\n{fmt_num(r.get('ats_score'), 0)}")
                        c_b.caption(f"Emb\n{fmt_num(r.get('embedding_similarity'), 2)}")
                        c_c.caption(f"LLM\n{fmt_num(r.get('llm_score'), 1)}")
                        just = (r.get("llm_justification") or "").strip()
                        if just:
                            st.write(just[:200] + ("…" if len(just) > 200 else ""))
            if len(shortlisted_sorted) > 3:
                st.caption(
                    f"+ {len(shortlisted_sorted) - 3} more in shortlist (see table below)"
                )
        elif status == "completed" and scored:
            st.info(
                f"No one marked shortlisted yet. Check ranks — shortlist size is {shortlist_size}.",
                icon=":material/info:",
            )

        st.markdown("##### All ranked candidates")
        show_only_shortlist = st.toggle(
            "Show shortlisted only",
            value=False,
            help="Hide candidates outside the top N shortlist",
            key=f"toggle_sl_only_{job_id}",
        )
        if not results:
            st.info(
                "Waiting for the first resumes to finish…",
                icon=":material/hourglass_top:",
            )
        else:
            display_src = shortlisted_sorted if show_only_shortlist else results
            if show_only_shortlist and not display_src:
                st.warning("No shortlisted candidates to show.", icon=":material/warning:")
                display_src = []
            rows = []
            for r in display_src:
                rows.append(
                    {
                        "Rank": r.get("rank") if r.get("hard_filter_pass") else "—",
                        "Shortlist": "Yes" if r.get("shortlisted") else "No",
                        "Candidate": r.get("candidate_name") or r.get("filename"),
                        "File": r.get("source_filename") or r.get("filename") or "",
                        "Dup": "Yes" if r.get("is_duplicate") else "",
                        "Status": r.get("resume_status"),
                        "Final": r.get("final_score"),
                        "ATS %": r.get("ats_score"),
                        "Embed": r.get("embedding_similarity"),
                        "LLM": r.get("llm_score"),
                        "Why": r.get("llm_justification") or "",
                        "resume_id": r.get("resume_id"),
                    }
                )
            df = pd.DataFrame(rows)
            show = [
                "Rank",
                "Shortlist",
                "Candidate",
                "File",
                "Dup",
                "Status",
                "Final",
                "ATS %",
                "Embed",
                "LLM",
                "Why",
            ]
            st.dataframe(
                df[show] if not df.empty else df,
                width="stretch",
                hide_index=True,
                height=min(400, 42 + 34 * min(max(len(df), 1), 10)),
                column_config={
                    "Final": st.column_config.NumberColumn(format="%.1f"),
                    "ATS %": st.column_config.NumberColumn(format="%.1f"),
                    "Embed": st.column_config.NumberColumn(format="%.3f"),
                    "LLM": st.column_config.NumberColumn(format="%.1f"),
                    "Shortlist": st.column_config.TextColumn("Shortlist"),
                },
            )

            st.markdown("##### Compare candidates")
            compare_ids = st.multiselect(
                "Select up to 3 candidates to compare",
                options=[r["resume_id"] for r in results if r.get("resume_id") is not None],
                format_func=lambda i: next(
                    (
                        f"{x.get('candidate_name') or x.get('filename')} (#{i})"
                        for x in results
                        if x.get("resume_id") == i
                    ),
                    str(i),
                ),
                max_selections=3,
                key=f"compare_{job_id}",
            )
            if compare_ids:
                cols = st.columns(len(compare_ids))
                for col, cid in zip(cols, compare_ids):
                    r = next(x for x in results if x.get("resume_id") == cid)
                    with col:
                        with st.container(border=True, height="stretch"):
                            st.markdown(
                                f"**{r.get('candidate_name') or r.get('filename')}**"
                            )
                            st.caption(
                                f"Rank {r.get('rank') or '—'} · Final {fmt_num(r.get('final_score'),1)}"
                            )
                            st.caption(
                                f"ATS {fmt_num(r.get('ats_score'),1)} · "
                                f"Emb {fmt_num(r.get('embedding_similarity'),2)} · "
                                f"LLM {fmt_num(r.get('llm_score'),1)}"
                            )
                            st.write((r.get("llm_justification") or "")[:180])
                            hr = r.get("hr_status") or "none"
                            if hr == "starred":
                                st.badge("Starred", icon=":material/star:", color="orange")
                            elif hr == "rejected":
                                st.badge("Rejected", icon=":material/block:", color="red")
                            mp = r.get("match_points") or {}
                            if mp.get("matched"):
                                st.caption("Matched: " + ", ".join(mp["matched"][:4]))

            # CSV exports
            export_rows = []
            for r in results:
                mp = r.get("match_points") or {}
                if not mp and isinstance(r.get("structured"), dict):
                    mp = r["structured"].get("_ranking") or {}
                export_rows.append(
                    {
                        "Rank": r.get("rank") if r.get("hard_filter_pass") else "",
                        "Shortlist": "Yes" if r.get("shortlisted") else "No",
                        "Candidate": r.get("candidate_name") or r.get("filename"),
                        "Source file": r.get("source_filename") or r.get("filename") or "",
                        "Name confidence": r.get("name_confidence") or "",
                        "Duplicate": "Yes" if r.get("is_duplicate") else "No",
                        "Status": r.get("resume_status"),
                        "Final": r.get("final_score"),
                        "ATS %": r.get("ats_score"),
                        "Embed": r.get("embedding_similarity"),
                        "LLM": r.get("llm_score"),
                        "Why": r.get("llm_justification") or "",
                        "Matched brief points": "; ".join(mp.get("matched") or []),
                        "Missing brief points": "; ".join(mp.get("missing") or []),
                    }
                )
            export_df = pd.DataFrame(export_rows)
            blocked = [
                r
                for r in results
                if r.get("shortlisted")
                and (
                    r.get("name_confidence") == "low"
                    or (r.get("name_warning") and not r.get("candidate_name"))
                )
            ]
            csv_buf = io.StringIO()
            export_df.to_csv(csv_buf, index=False)
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                st.download_button(
                    "Download all ranked (CSV)",
                    data=csv_buf.getvalue(),
                    file_name=f"ranked_job_{job_id}.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    width="stretch",
                )
            with c_dl2:
                only = (
                    export_df[export_df["Shortlist"] == "Yes"]
                    if not export_df.empty
                    else export_df
                )
                buf2 = io.StringIO()
                only.to_csv(buf2, index=False)
                if blocked:
                    st.caption(
                        f":material/warning: {len(blocked)} shortlisted name(s) "
                        "low-confidence — confirm before outreach."
                    )
                st.download_button(
                    f"Download shortlist only (top {shortlist_size})",
                    data=buf2.getvalue(),
                    file_name=f"shortlist_job_{job_id}.csv",
                    mime="text/csv",
                    icon=":material/star:",
                    width="stretch",
                )

    # ---- REVIEW TAB ----
    with tab_review:
        if not results:
            st.info(
                "Waiting for the first resumes to finish…",
                icon=":material/hourglass_top:",
            )
        else:
            st.markdown("##### Candidate detail")
            options = {
                r["resume_id"]: (
                    f"{'★ ' if r.get('shortlisted') else ''}"
                    f"{'⭐ ' if (r.get('hr_status') or '') == 'starred' else ''}"
                    f"{'⛔ ' if (r.get('hr_status') or '') == 'rejected' else ''}"
                    f"{r.get('candidate_name') or r.get('filename')} "
                    f"(#{r['resume_id']}"
                    f"{', rank ' + str(r.get('rank')) if r.get('rank') else ''})"
                )
                for r in results
                if r.get("resume_id") is not None
            }
            if not options:
                return
            rid = st.selectbox(
                "Select candidate",
                options=list(options.keys()),
                format_func=lambda i: options[i],
            )
            detail = next(r for r in results if r.get("resume_id") == rid)

            # HR actions
            with st.container(border=True):
                st.caption("HR actions")
                h1, h2, h3, h4 = st.columns([1, 1, 1, 2])
                with h1:
                    if st.button(
                        "Star",
                        key=f"star_{rid}",
                        width="stretch",
                        icon=":material/star:",
                    ):
                        api(
                            "PATCH",
                            f"/resumes/{rid}/hr",
                            json_body={
                                "hr_status": "starred",
                                "hr_notes": detail.get("hr_notes") or "",
                            },
                        )
                        st.rerun()
                with h2:
                    if st.button(
                        "Reject",
                        key=f"rej_{rid}",
                        width="stretch",
                        icon=":material/block:",
                    ):
                        api(
                            "PATCH",
                            f"/resumes/{rid}/hr",
                            json_body={
                                "hr_status": "rejected",
                                "hr_notes": detail.get("hr_notes") or "",
                            },
                        )
                        st.rerun()
                with h3:
                    if st.button(
                        "Clear HR",
                        key=f"clr_{rid}",
                        width="stretch",
                        icon=":material/restart_alt:",
                    ):
                        api(
                            "PATCH",
                            f"/resumes/{rid}/hr",
                            json_body={"hr_status": "none", "hr_notes": ""},
                        )
                        st.rerun()
                with h4:
                    notes = st.text_input(
                        "HR notes",
                        value=detail.get("hr_notes") or "",
                        key=f"notes_{rid}",
                    )
                    if st.button(
                        "Save notes",
                        key=f"saven_{rid}",
                        icon=":material/save:",
                    ):
                        api(
                            "PATCH",
                            f"/resumes/{rid}/hr",
                            json_body={
                                "hr_status": detail.get("hr_status") or "none",
                                "hr_notes": notes,
                            },
                        )
                        st.success("Notes saved", icon=":material/check:")
                        st.rerun()

            if detail.get("shortlisted"):
                st.success(
                    f"Shortlisted · Rank #{detail.get('rank')} "
                    f"(drive shortlist size: {shortlist_size})",
                    icon=":material/star:",
                )
            elif detail.get("rank"):
                st.info(
                    f"Not shortlisted · Rank #{detail.get('rank')} "
                    f"(only top {shortlist_size} are shortlisted)",
                    icon=":material/info:",
                )
            if detail.get("name_warning") or detail.get("name_confidence") == "low":
                st.warning(
                    detail.get("name_warning")
                    or "Name is low-confidence — confirm before outreach.",
                    icon=":material/warning:",
                )
            elif detail.get("name_source"):
                st.caption(
                    f"Name source: {detail.get('name_source')} · "
                    f"confidence: {detail.get('name_confidence') or 'n/a'}"
                )

            structured = detail.get("structured")
            if not structured:
                try:
                    full = api("GET", f"/resumes/{rid}")
                    structured = (full.get("resume") or {}).get("structured")
                except Exception:
                    structured = None
            pub = public_profile(structured)

            left, right = st.columns(2)
            with left:
                with st.container(border=True):
                    st.markdown("**Scores**")
                    st.metric("Final", fmt_num(detail.get("final_score"), 1), border=True)
                    s1, s2, s3 = st.columns(3)
                    s1.metric("ATS %", fmt_num(detail.get("ats_score"), 1))
                    s2.metric("Embed", fmt_num(detail.get("embedding_similarity"), 3))
                    s3.metric("LLM", fmt_num(detail.get("llm_score"), 1))
                    st.markdown("**Why this score**")
                    st.write(detail.get("llm_justification") or "—")
                    mp = detail.get("match_points") or {}
                    if not mp and isinstance(structured, dict):
                        mp = structured.get("_ranking") or {}
                    if mp:
                        st.markdown("**Brief match checklist**")
                        st.caption(mp.get("summary") or "")
                        matched = mp.get("matched") or []
                        missing = mp.get("missing") or []
                        if matched:
                            st.success(
                                "Matched: " + "; ".join(matched[:8]),
                                icon=":material/check:",
                            )
                        if missing:
                            st.warning(
                                "Not clearly evidenced: " + "; ".join(missing[:8]),
                                icon=":material/priority_high:",
                            )
                    if detail.get("is_duplicate"):
                        st.error(
                            "Possible duplicate ("
                            + ", ".join(
                                detail.get("duplicate_reasons") or ["same contact"]
                            )
                            + "). Primary peer ids: "
                            + ", ".join(
                                str(x) for x in (detail.get("duplicate_of") or [])[:5]
                            ),
                            icon=":material/content_copy:",
                        )
                    if detail.get("error_message"):
                        st.warning(detail["error_message"], icon=":material/warning:")

            with right:
                with st.container(border=True):
                    st.markdown("**Profile**")
                    fn = detail.get("source_filename") or detail.get("filename")
                    if fn:
                        st.caption(f":material/description: **Source file:** `{fn}`")
                    render_trust_warnings(structured)
                    if pub:
                        render_profile_fields(pub)
                    else:
                        st.caption("No profile extracted yet.")

                    with st.expander(
                        "Technical JSON (for debugging)",
                        icon=":material/code:",
                    ):
                        if structured:
                            st.json(structured)
                        else:
                            st.caption("No data")

            with st.expander(
                "Fix fields & re-score (optional)",
                icon=":material/edit:",
            ):
                with st.form(f"ov_{rid}"):
                    n1, n2 = st.columns(2)
                    with n1:
                        ov_name = st.text_input("Name", value=str(pub.get("name") or ""))
                        ov_years = st.number_input(
                            "Years",
                            0.0,
                            50.0,
                            float(pub.get("total_years_experience") or 0.0),
                            0.5,
                        )
                    with n2:
                        ov_skills = st.text_input(
                            "Skills (comma-separated)",
                            value=format_list(pub.get("skills"))
                            if pub.get("skills")
                            else "",
                        )
                    if st.form_submit_button(
                        "Save & re-score",
                        width="stretch",
                        icon=":material/save:",
                    ):
                        try:
                            api(
                                "PATCH",
                                f"/resumes/{rid}/override",
                                json_body={
                                    "name": ov_name.strip() or None,
                                    "total_years_experience": float(ov_years),
                                    "skills": ov_skills if ov_skills != "—" else "",
                                    "rescore": True,
                                    "run_llm": True,
                                },
                                timeout=300.0,
                            )
                            st.success("Updated", icon=":material/check:")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc), icon=":material/error:")

    if auto and is_running:
        time.sleep(POLL_SECONDS)
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    render_header()
    st.space("small")
    if st.session_state.view == "results" and st.session_state.active_job_id:
        view_results()
    else:
        view_setup()


if __name__ == "__main__":
    main()
