"""Resume text extraction: layout-aware PDF/DOCX/TXT + OCR fallback + quality meta."""

from __future__ import annotations

import copy
import logging
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import extract_rules
from config import get_settings

logger = logging.getLogger(__name__)

# Minimum characters of non-whitespace text before we consider extraction successful
MIN_TEXT_CHARS = 80
# Simple extract is "good enough" above this (skip heavy layout/tables)
_FAST_TEXT_CHARS = 220

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".rtf"}

# Vertical clustering tolerance (points) for reading-order line building
_LINE_Y_TOL = 3.5
# Same-line x-gap that inserts a space / column separator
_COL_GAP = 28.0

# Dedicated parse thread pool (larger than default for bulk PDF jobs)
_parse_executor: Optional[ThreadPoolExecutor] = None
_parse_executor_n: int = 0


def _parse_settings() -> dict[str, Any]:
    return get_settings()


def get_parse_executor() -> ThreadPoolExecutor:
    """Shared thread pool for CPU-bound PDF/DOCX extraction."""
    global _parse_executor, _parse_executor_n
    n = max(2, int(_parse_settings().get("parse_workers") or 8))
    if _parse_executor is None or _parse_executor_n != n:
        if _parse_executor is not None:
            try:
                _parse_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                _parse_executor.shutdown(wait=False)
            except Exception:
                pass
        _parse_executor = ThreadPoolExecutor(
            max_workers=n, thread_name_prefix="parse"
        )
        _parse_executor_n = n
        logger.info("Parse thread pool workers=%s", n)
    return _parse_executor


def shutdown_parse_executor() -> None:
    global _parse_executor, _parse_executor_n
    if _parse_executor is not None:
        try:
            _parse_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            _parse_executor.shutdown(wait=False)
        except Exception:
            pass
        _parse_executor = None
        _parse_executor_n = 0


def is_supported_resume(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def extract_text(file_path: str | Path) -> str:
    """
    Extract raw text from a resume file (backward-compatible string API).
    Prefer extract_document() when quality metadata is needed.
    """
    return extract_document(file_path).get("raw_text") or ""


def _file_cache_key(path: Path) -> tuple[str, int, int]:
    """mtime_ns + size — invalidate when file changes."""
    try:
        st = path.stat()
        return (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return (str(path), 0, 0)


@lru_cache(maxsize=512)
def _extract_document_cached(
    resolved: str, mtime_ns: int, size: int
) -> dict[str, Any]:
    """In-process cache of parsed documents (same path+mtime)."""
    return _extract_document_uncached(Path(resolved))


def extract_document(file_path: str | Path) -> dict[str, Any]:
    """
    Full document extraction with quality signals and section splits.

    Returns:
      {
        raw_text, sections, ocr_used, method, text_chars, section_hits,
        quality ("high"|"medium"|"low"), page_count, hybrid (optional precompute)
      }
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        return {
            "raw_text": "",
            "sections": {},
            "ocr_used": False,
            "method": None,
            "text_chars": 0,
            "section_hits": [],
            "quality": "low",
            "page_count": 0,
            "filename": path.name if path else "",
            "multi_column": False,
            "multi_col_pages": 0,
        }
    try:
        key = _file_cache_key(path)
        doc = _extract_document_cached(key[0], key[1], key[2])
        # Deep copy so callers cannot corrupt the LRU cache entry
        out = copy.deepcopy(doc)
        out["filename"] = path.name
        return out
    except Exception as exc:
        logger.warning("Cache extract failed, direct path: %s", exc)
        return _extract_document_uncached(path)


def _extract_document_uncached(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "raw_text": "",
        "sections": {},
        "ocr_used": False,
        "method": None,
        "text_chars": 0,
        "section_hits": [],
        "quality": "low",
        "page_count": 0,
        "filename": path.name if path else "",
        "multi_column": False,
        "multi_col_pages": 0,
    }
    if not path.exists():
        return result

    ext = path.suffix.lower()
    text = ""
    method = None
    ocr_used = False
    page_count = 0
    settings = _parse_settings()
    ocr_enabled = bool(settings.get("ocr_enabled", True))

    layout_meta: dict[str, Any] = {"multi_column": False, "multi_col_pages": 0}
    try:
        if ext == ".pdf":
            text, method, page_count, layout_meta = _extract_pdf_layout(path)
            if _is_near_empty(text) and ocr_enabled:
                logger.info("PDF text near-empty, trying OCR: %s", path.name)
                ocr = _ocr_pdf(path)
                if not _is_near_empty(ocr):
                    text = ocr
                    method = "ocr"
                    ocr_used = True
        elif ext in (".docx", ".doc"):
            text, method = _extract_docx_rich(path)
        elif ext in (".txt", ".rtf"):
            text = _extract_plain(path)
            method = "plain"
        else:
            logger.warning("Unsupported file type: %s", ext)
            return result
    except Exception as exc:
        logger.exception("Extraction failed for %s: %s", path.name, exc)
        return result

    text = _clean_text(text)
    # Header/footer strip only when text is long enough to benefit
    if len(text) > 800:
        text = _strip_repeated_headers_footers(text)
    sections = extract_rules.split_sections(text)
    section_hits = [
        k
        for k in ("summary", "experience", "education", "skills", "certifications", "projects")
        if sections.get(k)
    ]
    chars = len("".join(text.split()))
    quality = _quality_score(chars, ocr_used, section_hits, method)

    result.update(
        {
            "raw_text": text,
            "sections": sections,
            "ocr_used": ocr_used,
            "method": method,
            "text_chars": chars,
            "section_hits": section_hits,
            "quality": quality,
            "page_count": page_count,
            "multi_column": bool(layout_meta.get("multi_column")),
            "multi_col_pages": int(layout_meta.get("multi_col_pages") or 0),
        }
    )
    return result


def _quality_score(
    chars: int, ocr_used: bool, section_hits: list[str], method: Optional[str]
) -> str:
    if chars < MIN_TEXT_CHARS:
        return "low"
    if ocr_used:
        return "medium" if chars >= 400 and len(section_hits) >= 1 else "low"
    if chars >= 400 and len(section_hits) >= 2:
        return "high"
    if chars >= 200 or len(section_hits) >= 1:
        return "medium"
    return "low"


def _is_near_empty(text: str) -> bool:
    return len("".join((text or "").split())) < MIN_TEXT_CHARS


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    # Drop empty lines but keep paragraph separation (single blank max)
    cleaned: list[str] = []
    blank = False
    for ln in lines:
        if not ln:
            if not blank and cleaned:
                cleaned.append("")
                blank = True
            continue
        cleaned.append(ln)
        blank = False
    return "\n".join(cleaned).strip()


def _strip_repeated_headers_footers(text: str, min_repeats: int = 3) -> str:
    """Remove lines that repeat on many pages (page numbers / headers)."""
    lines = text.split("\n")
    if len(lines) < 20:
        return text
    counts: dict[str, int] = defaultdict(int)
    for ln in lines:
        key = ln.strip().lower()
        if 3 <= len(key) <= 80:
            counts[key] += 1
    drop = {
        k
        for k, c in counts.items()
        if c >= min_repeats
        and (
            re.search(r"page\s*\d+", k)
            or re.fullmatch(r"\d+", k)
            or re.search(r"confidential|curriculum vitae|\bresume\b", k)
        )
    }
    if not drop:
        # Also drop pure page-number lines
        pass
    out = []
    for ln in lines:
        key = ln.strip().lower()
        if key in drop:
            continue
        if re.fullmatch(r"(?i)page\s*\d+(\s*of\s*\d+)?", ln.strip()):
            continue
        out.append(ln)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# PDF — layout-aware
# ---------------------------------------------------------------------------


def _text_looks_usable(text: str) -> bool:
    """Enough real text that we can skip heavier PDF extractors."""
    if _is_near_empty(text):
        return False
    chars = len("".join((text or "").split()))
    if chars < _FAST_TEXT_CHARS:
        return False
    # Heuristic: resumes usually have letters + some structure
    alpha = sum(1 for c in text if c.isalpha())
    if alpha < 80:
        return False
    return True


def _parse_quality_signals(text: str) -> dict[str, Any]:
    """
    Lightweight signals to decide whether fast text is good enough
    or we should escalate to multi-column / table extraction.
    """
    text = text or ""
    chars = len("".join(text.split()))
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sections = extract_rules.split_sections(text)
    section_hits = [
        k
        for k in (
            "summary",
            "experience",
            "education",
            "skills",
            "certifications",
            "projects",
        )
        if sections.get(k)
    ]
    has_email = bool(extract_rules.EMAIL_RE.search(text[:2500]))
    has_phone = False
    try:
        m = extract_rules.PHONE_RE.search(text[:2500])
        if m and len(re.sub(r"\D", "", m.group(0))) >= 10:
            has_phone = True
    except Exception:
        pass
    # Garbled multi-col often yields very short lines or sparse section headers
    short_lines = sum(1 for ln in lines[:40] if 0 < len(ln) < 12)
    avg_line = (sum(len(ln) for ln in lines[:50]) / max(len(lines[:50]), 1)) if lines else 0
    return {
        "chars": chars,
        "section_hits": section_hits,
        "section_count": len(section_hits),
        "has_email": has_email,
        "has_phone": has_phone,
        "short_line_ratio": short_lines / max(min(len(lines), 40), 1),
        "avg_line_len": avg_line,
        "line_count": len(lines),
    }


def _should_escalate_from_fast(text: str) -> bool:
    """
    Escalate when quality looks low: few sections, no contact, or
    line structure suggests multi-column scramble / missing table skills.
    """
    if not _text_looks_usable(text):
        return True
    sig = _parse_quality_signals(text)
    # Missing contact + thin structure
    if sig["section_count"] < 2 and not (sig["has_email"] or sig["has_phone"]):
        return True
    if sig["section_count"] == 0 and sig["chars"] < 1200:
        return True
    # Many tiny fragments often means columns mashed wrong
    if sig["short_line_ratio"] > 0.55 and sig["line_count"] > 25:
        return True
    # No experience/skills section on a long resume — likely layout miss
    hits = set(sig["section_hits"])
    if sig["chars"] > 1500 and not hits.intersection({"experience", "skills", "education"}):
        return True
    # Skills section empty-ish but resume mentions skills-like density without header
    if "skills" not in hits and sig["chars"] > 800:
        # table-like pipes rare in fast path; lack of section + medium length → escalate
        if sig["section_count"] <= 1:
            return True
    return False


def _extract_pdf_layout(path: Path) -> tuple[str, Optional[str], int, dict[str, Any]]:
    """
    Fast-first PDF extract; escalate to multi-column/tables when quality is low.
    Returns (text, method, page_count, layout_meta).
    """
    layout_meta: dict[str, Any] = {
        "multi_column": False,
        "multi_col_pages": 0,
        "escalated": False,
        "escalate_reason": None,
    }
    settings = _parse_settings()
    fast_mode = bool(settings.get("pdf_fast_mode", True))
    extract_tables = bool(settings.get("pdf_extract_tables", False))
    auto_tables = bool(settings.get("pdf_auto_tables_on_low_quality", True))

    simple_text = ""
    pages = 0

    # 0) Cheap path — most digital resumes are fine with plain text extract
    if fast_mode:
        simple_text, pages = _extract_pdf_pymupdf_simple_with_pages(path)
        if _text_looks_usable(simple_text) and not _should_escalate_from_fast(simple_text):
            return simple_text, "pymupdf_fast", pages, layout_meta
        if _text_looks_usable(simple_text) and _should_escalate_from_fast(simple_text):
            layout_meta["escalated"] = True
            layout_meta["escalate_reason"] = "low_quality_fast_path"
            # Prefer tables when escalating for missing skills/structure
            extract_tables = extract_tables or auto_tables
            logger.info(
                "PDF quality low for %s — escalating to multi-column/tables",
                path.name,
            )

    # 1) PyMuPDF dict blocks — multi-column aware (+ tables if escalated/enabled)
    text, pages_b, multi_n = _extract_pdf_pymupdf_blocks(
        path, extract_tables=extract_tables
    )
    pages = pages_b or pages
    if not _is_near_empty(text):
        # Prefer escalated layout if it recovers more structure than fast path
        if simple_text and _text_looks_usable(simple_text):
            fast_sig = _parse_quality_signals(simple_text)
            slow_sig = _parse_quality_signals(text)
            # Keep better of the two by section count then chars
            if (
                slow_sig["section_count"] < fast_sig["section_count"]
                and slow_sig["chars"] < fast_sig["chars"] * 0.85
            ):
                # Escalation worse — fall back to fast
                return simple_text, "pymupdf_fast", pages, {
                    **layout_meta,
                    "escalated": False,
                    "escalate_reason": "layout_worse_kept_fast",
                }
        layout_meta = {
            **layout_meta,
            "multi_column": multi_n > 0,
            "multi_col_pages": multi_n,
            "tables_extracted": extract_tables,
        }
        method = "pymupdf_multicol" if multi_n > 0 else "pymupdf_blocks"
        if extract_tables:
            method = method + "+tables"
        if layout_meta.get("escalated"):
            method = method + "+escalated"
        return text, method, pages, layout_meta

    # 2) If fast path was skipped or failed, try simple again
    if not fast_mode or not simple_text:
        simple_text, pages_s = _extract_pdf_pymupdf_simple_with_pages(path)
        pages = pages or pages_s
        if not _is_near_empty(simple_text):
            return simple_text, "pymupdf_simple", pages, layout_meta

    if simple_text and not _is_near_empty(simple_text):
        return simple_text, "pymupdf_fast_fallback", pages, layout_meta

    # 3) pdfplumber only when PyMuPDF failed (much slower)
    text2, pages2 = _extract_pdf_pdfplumber_simple_pages(path)
    if not _is_near_empty(text2):
        return text2, "pdfplumber_simple", pages2, layout_meta

    # 4) Last resort: full pdfplumber layout (tables + char clustering)
    text3, pages3 = _extract_pdf_pdfplumber_layout(path)
    if not _is_near_empty(text3):
        return text3, "pdfplumber_layout", pages3, layout_meta

    return "", None, pages or pages2 or pages3, layout_meta


def _extract_pdf_pymupdf_blocks(
    path: Path, *, extract_tables: bool = False
) -> tuple[str, int, int]:
    """Returns (text, page_count, multi_col_pages)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "", 0, 0
    page_texts: list[str] = []
    multi_col_pages = 0
    try:
        doc = fitz.open(path)
        for page in doc:
            text, is_multi = _page_blocks_to_text(page)
            if is_multi:
                multi_col_pages += 1
            page_texts.append(text)
            # find_tables is expensive — only when explicitly enabled
            if extract_tables:
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        for table in tabs.tables:
                            rows = table.extract()
                            for row in rows:
                                cells = [
                                    str(c).strip()
                                    for c in row
                                    if c and str(c).strip()
                                ]
                                if cells:
                                    page_texts[-1] += "\n" + " | ".join(cells)
                except Exception:
                    pass
        n = doc.page_count
        doc.close()
        joined = "\n\n".join(t for t in page_texts if t.strip())
        return joined, n, multi_col_pages
    except Exception as exc:
        logger.warning("PyMuPDF blocks failed on %s: %s", path.name, exc)
        return "", 0, 0


def _detect_column_split(
    spans: list[tuple[float, float, float, str]], page_width: float
) -> Optional[float]:
    """
    Detect a 2-column layout from span mid-x positions.
    spans: (y0, x0, x1, text)
    Returns split x coordinate or None if single-column.
    """
    if len(spans) < 8 or page_width <= 0:
        return None
    mids = [((x0 + x1) / 2.0) for _, x0, x1, _ in spans if x1 > x0]
    if len(mids) < 8:
        return None

    # Histogram into 20 bins across page width
    bins = 20
    hist = [0] * bins
    for m in mids:
        idx = min(bins - 1, max(0, int((m / page_width) * bins)))
        hist[idx] += 1

    # Find a valley in the middle 40–60% of the page with peaks on both sides
    left_peak = max(hist[: bins // 2]) if hist else 0
    right_peak = max(hist[bins // 2 :]) if hist else 0
    if left_peak < 3 or right_peak < 3:
        return None

    mid_start, mid_end = bins // 3, (2 * bins) // 3
    valley_idx = mid_start
    valley_val = hist[mid_start]
    for i in range(mid_start, mid_end + 1):
        if hist[i] < valley_val:
            valley_val = hist[i]
            valley_idx = i

    # Valley should be clearly lower than both peaks
    if valley_val > min(left_peak, right_peak) * 0.45:
        return None

    split_x = (valley_idx + 0.5) / bins * page_width
    # Require a real gap: left content left of split, right content right of split
    left_n = sum(1 for m in mids if m < split_x - page_width * 0.02)
    right_n = sum(1 for m in mids if m > split_x + page_width * 0.02)
    if left_n < 3 or right_n < 3:
        return None
    return split_x


def _lines_from_spans(spans: list[tuple[float, float, str]]) -> list[str]:
    """Cluster (y, x, text) spans into reading-order lines."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (round(s[0] / _LINE_Y_TOL), s[1]))
    lines_out: list[str] = []
    current_y: Optional[float] = None
    line_items: list[tuple[float, str]] = []

    def flush() -> None:
        nonlocal line_items
        if not line_items:
            return
        line_items.sort(key=lambda x: x[0])
        buf = line_items[0][1]
        for i in range(1, len(line_items)):
            prev_x = line_items[i - 1][0]
            if line_items[i][0] - prev_x > _COL_GAP * 2:
                buf += "  " + line_items[i][1]
            else:
                buf += " " + line_items[i][1]
        lines_out.append(buf.strip())
        line_items = []

    for y0, x0, text in spans:
        if current_y is None or abs(y0 - current_y) <= _LINE_Y_TOL:
            current_y = y0 if current_y is None else (current_y + y0) / 2
            line_items.append((x0, text))
        else:
            flush()
            current_y = y0
            line_items = [(x0, text)]
    flush()
    return lines_out


def _page_blocks_to_text(page: Any) -> tuple[str, bool]:
    """
    Sort text spans by reading order.
    If multi-column layout detected: left column top→bottom, then right column.
    Returns (text, is_multi_column).
    """
    try:
        data = page.get_text("dict", flags=0)
    except Exception:
        return (page.get_text("text") or ""), False

    try:
        page_width = float(page.rect.width)
    except Exception:
        page_width = 0.0

    # (y0, x0, x1, text)
    rich: list[tuple[float, float, float, str]] = []
    for block in data.get("blocks") or []:
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines") or []:
            parts = []
            for span in line.get("spans") or []:
                t = (span.get("text") or "").strip()
                if t:
                    parts.append(t)
            if not parts:
                continue
            bbox = line.get("bbox") or block.get("bbox") or (0, 0, 0, 0)
            x0, y0, x1 = float(bbox[0]), float(bbox[1]), float(bbox[2])
            rich.append((y0, x0, x1, " ".join(parts)))

    if not rich:
        return (page.get_text("text") or ""), False

    split_x = _detect_column_split(rich, page_width)
    if split_x is not None:
        left = [(y, x0, t) for y, x0, x1, t in rich if (x0 + x1) / 2.0 < split_x]
        right = [(y, x0, t) for y, x0, x1, t in rich if (x0 + x1) / 2.0 >= split_x]
        left_lines = _lines_from_spans(left)
        right_lines = _lines_from_spans(right)
        # Full left column, then full right (typical resume sidebar + main body)
        parts = []
        if left_lines:
            parts.append("\n".join(left_lines))
        if right_lines:
            parts.append("\n".join(right_lines))
        return "\n\n".join(parts), True

    # Single-column: top→bottom, left→right
    simple = [(y, x0, t) for y, x0, x1, t in rich]
    return "\n".join(_lines_from_spans(simple)), False


def _extract_pdf_pymupdf_simple(path: Path) -> str:
    text, _ = _extract_pdf_pymupdf_simple_with_pages(path)
    return text


def _extract_pdf_pymupdf_simple_with_pages(path: Path) -> tuple[str, int]:
    """Fastest digital-PDF path: native text layer only."""
    try:
        import fitz
    except ImportError:
        return "", 0
    parts: list[str] = []
    n = 0
    try:
        doc = fitz.open(path)
        n = doc.page_count
        # sort=True improves reading order slightly with little cost
        for page in doc:
            try:
                t = page.get_text("text", sort=True) or ""
            except TypeError:
                t = page.get_text("text") or ""
            if t.strip():
                parts.append(t)
        doc.close()
    except Exception as exc:
        logger.warning("PyMuPDF simple failed on %s: %s", path.name, exc)
        return "", 0
    return "\n\n".join(parts), n


def _extract_pdf_pdfplumber_layout(path: Path) -> tuple[str, int]:
    try:
        import pdfplumber
    except ImportError:
        return "", 0
    page_texts: list[str] = []
    n = 0
    try:
        with pdfplumber.open(path) as pdf:
            n = len(pdf.pages)
            for page in pdf.pages:
                page_texts.append(_pdfplumber_page_text(page))
    except Exception as exc:
        logger.warning("pdfplumber layout failed on %s: %s", path.name, exc)
        return "", 0
    return "\n\n".join(t for t in page_texts if t.strip()), n


def _pdfplumber_page_text(page: Any) -> str:
    chunks: list[str] = []
    # Tables first (skills grids, etc.)
    try:
        tables = page.extract_tables() or []
        for table in tables:
            for row in table:
                cells = [str(c).strip() for c in row if c and str(c).strip()]
                if cells:
                    chunks.append(" | ".join(cells))
    except Exception:
        pass

    # Char-level clustering for reading order
    try:
        chars = page.chars or []
        if chars:
            chunks.append(_chars_to_text(chars))
            return "\n".join(chunks)
    except Exception:
        pass

    t = page.extract_text() or ""
    if t.strip():
        chunks.append(t)
    return "\n".join(chunks)


def _chars_to_text(chars: list[dict]) -> str:
    """Group pdfplumber chars into lines sorted by y then x."""
    if not chars:
        return ""
    # Normalize to (y, x, text)
    items = []
    for c in chars:
        text = c.get("text") or ""
        if not text:
            continue
        items.append((float(c.get("top", 0)), float(c.get("x0", 0)), text))
    items.sort(key=lambda t: (round(t[0] / _LINE_Y_TOL), t[1]))

    lines: list[str] = []
    current_y: Optional[float] = None
    buf: list[tuple[float, str]] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        buf.sort(key=lambda x: x[0])
        line = ""
        prev_x: Optional[float] = None
        for x, ch in buf:
            if prev_x is not None and x - prev_x > _COL_GAP:
                line += "  " if x - prev_x < _COL_GAP * 3 else " | "
            line += ch
            prev_x = x + 1  # approximate
        lines.append(line.strip())
        buf = []

    for y, x, ch in items:
        if current_y is None or abs(y - current_y) <= _LINE_Y_TOL:
            current_y = y if current_y is None else (current_y + y) / 2
            buf.append((x, ch))
        else:
            flush()
            current_y = y
            buf = [(x, ch)]
    flush()
    return "\n".join(lines)


def _extract_pdf_pdfplumber_simple(path: Path) -> str:
    text, _ = _extract_pdf_pdfplumber_simple_pages(path)
    return text


def _extract_pdf_pdfplumber_simple_pages(path: Path) -> tuple[str, int]:
    try:
        import pdfplumber
    except ImportError:
        return "", 0
    parts: list[str] = []
    n = 0
    try:
        with pdfplumber.open(path) as pdf:
            n = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
    except Exception as exc:
        logger.warning("pdfplumber simple failed on %s: %s", path.name, exc)
        return "", 0
    return "\n\n".join(parts), n


def _ocr_pdf(path: Path) -> str:
    """OCR fallback using pdf2image + pytesseract (lower DPI, few pages)."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        logger.warning(
            "OCR deps missing (pdf2image/pytesseract). Skipping OCR for %s",
            path.name,
        )
        return ""
    settings = _parse_settings()
    dpi = max(72, int(settings.get("ocr_dpi") or 150))
    max_pages = max(1, int(settings.get("ocr_max_pages") or 3))
    try:
        images = convert_from_path(
            str(path), dpi=dpi, first_page=1, last_page=max_pages
        )
        chunks: list[str] = []
        # Prefer faster tesseract config for resumes (sparse text)
        tess_config = "--psm 6"
        for img in images:
            # Downscale wide images slightly for speed
            try:
                w, h = img.size
                if w > 1600:
                    ratio = 1600 / w
                    img = img.resize((1600, max(1, int(h * ratio))))
            except Exception:
                pass
            chunks.append(
                pytesseract.image_to_string(img, config=tess_config) or ""
            )
        return "\n\n".join(chunks)
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", path.name, exc)
        return ""


# ---------------------------------------------------------------------------
# DOCX / plain
# ---------------------------------------------------------------------------


def _extract_docx_rich(path: Path) -> tuple[str, Optional[str]]:
    if path.suffix.lower() == ".docx":
        text = _extract_python_docx(path)
        if not _is_near_empty(text):
            return text, "python_docx"
    text2 = _extract_docx2txt(path)
    if not _is_near_empty(text2):
        return text2, "docx2txt"
    return text2 or "", None


def _extract_python_docx(path: Path) -> str:
    try:
        import docx
    except ImportError:
        return ""
    try:
        document = docx.Document(str(path))
        paras = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                if cells:
                    paras.append(" | ".join(cells))
        return "\n".join(paras)
    except Exception as exc:
        logger.warning("python-docx failed on %s: %s", path.name, exc)
        return ""


def _extract_docx2txt(path: Path) -> str:
    try:
        import docx2txt
    except ImportError:
        return ""
    try:
        return docx2txt.process(str(path)) or ""
    except Exception as exc:
        logger.warning("docx2txt failed on %s: %s", path.name, exc)
        return ""


def _extract_plain(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            logger.warning("plain read failed %s: %s", path.name, exc)
            return ""
    return ""


# ---------------------------------------------------------------------------
# Zip / upload helpers
# ---------------------------------------------------------------------------


def extract_resumes_from_zip(
    zip_path: str | Path, dest_dir: str | Path
) -> list[Path]:
    """
    Extract supported resume files from a zip into dest_dir.
    Returns list of extracted file paths. Skips macOS junk / directories.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if not name or name.startswith("."):
                    continue
                if name.startswith("__MACOSX"):
                    continue
                if "__MACOSX" in Path(info.filename).parts:
                    continue
                suffix = Path(name).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    continue
                target = dest_dir / name
                if not str(target.resolve()).startswith(str(dest_dir.resolve())):
                    continue
                if target.exists():
                    stem, ext = target.stem, target.suffix
                    i = 1
                    while target.exists():
                        target = dest_dir / f"{stem}_{i}{ext}"
                        i += 1
                with zf.open(info) as src, open(target, "wb") as out:
                    out.write(src.read())
                if is_supported_resume(target):
                    extracted.append(target)
                else:
                    try:
                        target.unlink()
                    except OSError:
                        pass
    except zipfile.BadZipFile as exc:
        logger.error("Bad zip file %s: %s", zip_path, exc)
        raise ValueError(f"Invalid zip file: {zip_path.name}") from exc

    return extracted


def collect_upload_files(
    saved_paths: list[Path], extract_root: Path
) -> list[Path]:
    """
    Given uploaded files (may include .zip), expand zips and return
    a flat list of resume file paths.
    """
    resumes: list[Path] = []
    for p in saved_paths:
        if p.suffix.lower() == ".zip":
            sub = extract_root / f"_zip_{p.stem}"
            sub.mkdir(parents=True, exist_ok=True)
            try:
                resumes.extend(extract_resumes_from_zip(p, sub))
            except ValueError:
                logger.exception("Failed to extract zip %s", p.name)
        elif is_supported_resume(p):
            resumes.append(p)
        else:
            logger.warning("Skipping unsupported upload: %s", p.name)
    return resumes
