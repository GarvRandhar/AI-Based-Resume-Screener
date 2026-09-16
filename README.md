# AI-Based Resume Screener
*By Garv Randhar*

Local, privacy-first AI resume screening for recruiters. Paste one **hiring brief** (full instructions), upload 100+ resumes, and get an explainable ranked shortlist — all on one machine with **Ollama**, **FastAPI**, **React**, and **SQLite**.

No cloud APIs, no Docker, no Redis/Celery. Resume data never leaves your Mac.

## Architecture

| Layer | Choice |
|--------|--------|
| API | FastAPI (`app.py`) — single async process |
| UI | React + Vite (`frontend/`) |
| DB | SQLite (`data/screener.db`) |
| Queue | `asyncio` + dual semaphores (LLM **4** / embed **8**) |
| LLM extract | Ollama `llama3.2:3b` (fast structured extract / enrich) |
| LLM score | Ollama `llama3.1:8b` (`score_model` — fit only; falls back to extract) |
| Embeddings | Ollama `nomic-embed-text` |
| Parsing | pdfplumber / PyMuPDF, python-docx, OCR fallback |

**Pipeline (per resume):** layout-aware text extract → structured profile → optional hard filters → **rank vs hiring brief only** (ATS keyword % + embedding similarity + LLM fit) → blended final score.

**Prompt-first & multi-industry:** HR writes one brief for any drive (sales, clinical, ops, campus, tech…). Ranking uses only that brief (ATS + embeddings + LLM). Hybrid extract is industry-light (name/email/phone/years/education); skills come from the LLM. Optional hard rules (min years, license) with fuzzy match. Trust flags warn when name/location look unreliable.

**Extraction quality:** PDFs use multi-column-aware reading order + table cells; email/phone/skills/years are grounded by regex/lexicon; **two-pass years** are computed from work history in code (not LLM math). Dashboard shows extract confidence, OCR, and multi-column flags.

**Fast bulk mode (default):** Phase 1 is **hybrid-only** (no LLM chat) + ATS + embed for everyone. Phase 2 runs full LLM extract + fit only for the top-K by pre-score (default top 20, min pre-score 30). Small batches (&lt;12) still get full LLM for all. Re-score any row later.

**Recruiter overrides:** Edit name/years/skills/etc. on a candidate and re-score that row only (no full batch re-run).

**Shortlist enrich:** Triggered manually by HR on demand via **Enrich shortlist profiles** on the results page (`enrich_shortlist`, default off).

**Name verify queue:** Results API and UI flag low-confidence / heuristic-only names so HR can fix them before outreach.

**Eval harness:** `python eval/run_eval.py` — offline name-extraction regression checks (see `eval/README.md`).

## Prerequisites (MacBook Air M4)

1. **Python 3.11+** (3.12–3.14 fine)
2. **Node.js 18+** (for the React UI)
3. **Ollama** — [https://ollama.com](https://ollama.com)
4. Optional for scanned PDFs (OCR):
   ```bash
   brew install tesseract poppler
   ```

## Setup

```bash
cd "Resume Screener POC"

# Python backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# React frontend
cd frontend
npm install
cd ..

# Pull local models (one-time; needs network only here)
ollama pull llama3.2:3b
ollama pull llama3.1:8b   # fit scoring (optional but recommended)
ollama pull nomic-embed-text
```

Confirm Ollama is running (the macOS app usually keeps it up):

```bash
curl -s http://localhost:11434/api/tags | head
```

## Run

**Terminal 1 — API**

```bash
source .venv/bin/activate
python app.py
# → http://127.0.0.1:8000
# → docs at http://127.0.0.1:8000/docs
```

**Terminal 2 — React UI**

```bash
cd frontend
npm run dev
# → http://localhost:5173
```

Open **http://localhost:5173** in your browser.

Optional production build of the UI:

```bash
cd frontend
npm run build
npm run preview
```

API base URL defaults to `http://127.0.0.1:8000`. Override with:

```bash
# frontend/.env.local
VITE_API_BASE=http://127.0.0.1:8000
```

## Usage (one page)

1. Open the React dashboard (**http://localhost:5173**).
2. Paste a **hiring brief** (full instructions for who to hire). Use the checklist to strengthen it.
3. Set **Shortlist size (top N)** and optional **min score**.
4. Upload resumes (PDF/DOCX/TXT or ZIP).
5. Click **Rank resumes** — progress shows **stage + current file**; use Pause / Resume / Cancel.
6. After completion: edit shortlist size live, star/reject/notes, compare 2–3 candidates, export pack.
7. **Re-run failed only** reprocesses parse failures without redoing the whole batch.

### Config (`config.json`)

Edit models, ports, concurrency without code changes:

```json
{
  "api_host": "127.0.0.1",
  "api_port": 8000,
  "ollama_base": "http://localhost:11434",
  "extract_model": "llama3.2:3b",
  "score_model": "llama3.1:8b",
  "embed_model": "nomic-embed-text",
  "default_concurrency": 4,
  "embed_concurrency": 8,
  "hybrid_first_bulk": true,
  "enrich_shortlist": false,
  "pdf_fast_mode": true,
  "pdf_auto_tables_on_low_quality": true,
  "cross_job_dedupe": true
}
```

| Key | Effect |
|-----|--------|
| `extract_model` | Fast model for structured extract / shortlist enrich |
| `score_model` | Stronger model for fit score only (falls back to extract if missing) |
| `default_concurrency` | Parallel LLM chat calls (extract/score) |
| `embed_concurrency` | Parallel embedding calls (separate pool) |
| `hybrid_first_bulk` | Bulk phase-1 skips LLM extract; only top-K get fit |
| `enrich_shortlist` | After rank, LLM-fill skills/work history for shortlist |
| `pdf_fast_mode` | Prefer cheap PyMuPDF text first |
| `pdf_auto_tables_on_low_quality` | Escalate to multi-column + tables when fast text looks weak |
| `cross_job_dedupe` | Flag candidates seen in other jobs (email/phone) |
| `ocr_enabled` / `ocr_dpi` / `ocr_max_pages` | OCR only for scanned PDFs |

## Project layout

```
app.py              # FastAPI entrypoint
frontend/           # React + Vite recruiter UI
db.py               # SQLite schema + CRUD
parsing.py          # PDF/DOCX/TXT + OCR + zip extract
ollama_client.py    # HTTP client: extract, score, embed
filters.py          # Deterministic hard filters
scoring.py          # ATS overlap, cosine sim, final blend
job_manager.py      # Async batch orchestration
requirements.txt
README.md
data/               # SQLite DB (created at runtime)
uploads/            # Temp files per job (cleaned after processing)
```

> Legacy: `dashboard.py` is the old Streamlit UI (no longer required). Prefer `frontend/`.

## Scoring formula

Final score (0–100) blends:

| Component | Weight | Scale |
|-----------|--------|--------|
| LLM fit | 50% | 1–10 → ×10 |
| Embedding similarity | 30% | cosine 0–1 → ×100 |
| ATS keyword % | 20% | 0–100 |

All three appear separately in the UI — never a single opaque number.

## API sketch

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | API + Ollama model check |
| POST | `/brief/analyze` | Hiring brief quality checklist |
| POST | `/jobs` | Create job + criteria |
| POST | `/jobs/{id}/upload` | Multipart resume upload |
| POST | `/jobs/{id}/start` | Start async processing |
| GET | `/jobs/{id}/progress` | `processed_count` / `total_count` |
| GET | `/jobs/{id}/results` | Ranked shortlist |
| GET | `/resumes/{id}` | Full structured data |
| GET/POST/DELETE | `/templates` | Criteria templates |

## Notes for M4 Air (16 GB)

- Default **concurrency = 2** matches Ollama’s local serialization; raise carefully.
- First LLM call after idle can be slow (model load).
- 100 resumes will take a while with an 8B model — the UI shows partial results as each file finishes.
- One bad resume never aborts the batch (per-file try/except + retries on JSON parse).

## Offline / privacy

After `ollama pull`, everything runs offline. Uploaded files live under `uploads/job_{id}/` and are deleted when the job completes. Structured fields and scores remain in SQLite for auditability.

## Version Control (GitHub)

If you are committing this project to a git repository, make sure not to include the following generated files and folders, which are already ignored by `.gitignore`:
- `.venv/` / `__pycache__/` (Python virtual environment and cache)
- `frontend/node_modules/` & `frontend/dist/` (Node dependencies and React build)
- `data/` & `uploads/` (Local SQLite database and temporary parsed resumes)
- `.DS_Store` (macOS hidden files)
- `.env` files (Local secrets and configuration)
>>>>>>> b8eda12 (Initial commit: AI-Based Resume Screener)
