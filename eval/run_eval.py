#!/usr/bin/env python3
"""
Small offline eval harness (no Ollama required for name checks).

Usage (from repo root):
  python eval/run_eval.py
  python eval/run_eval.py --json

Exit code 0 if all cases pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import extract_rules  # noqa: E402
import parsing  # noqa: E402


def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


def _conf_rank(c: str | None) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get((c or "low").lower(), 0)


def run_case(case: dict) -> dict:
    case_id = case.get("id") or "unknown"
    path = case.get("path")
    filename = case.get("filename_only")
    text = case.get("resume_text") or ""

    if path:
        p = ROOT / path
        if not p.exists():
            return {
                "id": case_id,
                "ok": False,
                "error": f"missing file {path}",
            }
        doc = parsing.extract_document(p)
        text = doc.get("raw_text") or ""
        filename = filename or p.name

    hybrid = extract_rules.hybrid_extract(text, filename=filename)
    resolved = extract_rules.resolve_candidate_name(
        llm_name=None,
        hybrid_name=hybrid.get("name"),
        email=hybrid.get("email"),
        filename=filename,
        resume_text=text[:2000],
    )
    got_name = resolved.get("name")
    conf = resolved.get("confidence")
    source = resolved.get("source")

    failures: list[str] = []

    expected = case.get("expected_name")
    if "expected_name" in case:
        if expected is None:
            if got_name:
                # Allow only if not containing banned prose
                must_not = case.get("must_not_contain") or []
                if any(b.lower() in (got_name or "").lower() for b in must_not):
                    failures.append(f"got banned name {got_name!r}")
                # For null expected, empty is pass; non-empty without ban is soft fail
                if got_name and not must_not:
                    failures.append(f"expected no name, got {got_name!r}")
        else:
            if _norm(got_name) != _norm(expected):
                failures.append(f"name: got {got_name!r} want {expected!r}")

    email_sub = case.get("expected_email_contains")
    if email_sub:
        email = (hybrid.get("email") or "").lower()
        if email_sub.lower() not in email:
            failures.append(f"email missing {email_sub!r} (got {email!r})")

    min_conf = case.get("min_name_confidence")
    if min_conf and expected:
        if _conf_rank(conf) < _conf_rank(min_conf):
            failures.append(f"confidence {conf!r} < {min_conf!r}")

    must_not = case.get("must_not_contain") or []
    for ban in must_not:
        if ban.lower() in (got_name or "").lower():
            failures.append(f"name contains banned {ban!r}")

    return {
        "id": case_id,
        "ok": len(failures) == 0,
        "got_name": got_name,
        "confidence": conf,
        "source": source,
        "email": hybrid.get("email"),
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run resume screener golden eval")
    ap.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).parent / "golden.json",
        help="Path to golden.json",
    )
    ap.add_argument("--json", action="store_true", help="Machine-readable output")
    args = ap.parse_args()

    data = json.loads(args.golden.read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    results = [run_case(c) for c in cases]
    passed = sum(1 for r in results if r["ok"])
    failed = len(results) - passed

    if args.json:
        print(
            json.dumps(
                {
                    "passed": passed,
                    "failed": failed,
                    "total": len(results),
                    "results": results,
                },
                indent=2,
            )
        )
    else:
        print(f"Eval: {passed}/{len(results)} passed")
        for r in results:
            mark = "OK" if r["ok"] else "FAIL"
            print(
                f"  [{mark}] {r['id']}: name={r.get('got_name')!r} "
                f"src={r.get('source')} conf={r.get('confidence')}"
            )
            for f in r.get("failures") or []:
                print(f"         · {f}")
        if failed:
            print(f"\n{failed} case(s) failed.")
        else:
            print("\nAll cases passed.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
