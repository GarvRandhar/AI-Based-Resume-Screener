# Eval harness

Small offline checks for **name extraction** (and basic contact). Does not call Ollama.

## Run

```bash
cd "Resume Screener POC"
source .venv/bin/activate
python eval/run_eval.py
python eval/run_eval.py --json   # CI-friendly
```

Exit code `0` = all pass, `1` = failures.

## Golden file

Edit `eval/golden.json` to add cases:

- `path` — file under repo root (optional if `filename_only` + `resume_text`)
- `expected_name` — exact display name, or `null` if no name should be used
- `must_not_contain` — banlist substrings (e.g. prose junk)
- `expected_email_contains` — optional substring check
- `min_name_confidence` — `low` | `medium` | `high`

## Not covered yet

- Full LLM extract quality
- Ranking / ATS scores
- End-to-end job API

Those can be added as separate suites when Ollama is available in CI.
