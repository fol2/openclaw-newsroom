#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.eval_dataset import replay_record  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {lineno} is not a JSON object")
            rows.append(obj)
    return rows


def _summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    parse_ok = sum(1 for r in results if bool(r.get("parse_ok")))
    target_match = sum(1 for r in results if bool(r.get("target_match")))
    correctness_match = sum(1 for r in results if bool(r.get("correctness_match")))

    error_counts: Counter[str] = Counter()
    for r in results:
        if bool(r.get("parse_ok")):
            continue
        err = str(r.get("error") or "unknown_error")
        error_counts[err] += 1

    return {
        "rows": len(results),
        "parse_ok_rows": parse_ok,
        "target_match_rows": target_match,
        "correctness_match_rows": correctness_match,
        "parse_ok_rate": (parse_ok / len(results)) if results else 0.0,
        "target_match_rate": (target_match / len(results)) if results else 0.0,
        "correctness_match_rate": (correctness_match / len(results)) if results else 0.0,
        "parse_errors": dict(sorted(error_counts.items())),
    }


def main(argv: list[str]) -> int:
    # Keep replay output deterministic and quiet unless the script itself fails.
    logging.getLogger("newsroom.event_manager").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Replay labelled clustering eval dataset rows through parser logic.")
    parser.add_argument(
        "--dataset",
        default=str(OPENCLAW_HOME / "newsroom" / "evals" / "clustering_eval_dataset_v1.jsonl"),
        help="Input dataset JSONL path.",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional output path for replay summary JSON.",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=0.22,
        help="Threshold passed into replay helper for metadata parity.",
    )
    parser.add_argument(
        "--fail-on-parse-error",
        action="store_true",
        help="Exit non-zero when any row fails to parse during replay.",
    )
    parser.add_argument(
        "--fail-on-correctness-mismatch",
        action="store_true",
        help="Exit non-zero when replay-derived correctness differs from labels.correctness.",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    rows = _load_jsonl(dataset_path)
    results = [replay_record(r, low_score_threshold=float(args.low_score_threshold)) for r in rows]
    summary = _summarise(results)

    out = {
        "ok": True,
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "low_score_threshold": float(args.low_score_threshold),
        "summary": summary,
    }

    if args.summary_out:
        out_path = Path(args.summary_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    print(json.dumps(out, ensure_ascii=False))

    parse_errors = int(summary["rows"]) - int(summary["parse_ok_rows"])
    correctness_mismatches = int(summary["rows"]) - int(summary["correctness_match_rows"])

    if bool(args.fail_on_parse_error) and parse_errors > 0:
        return 1
    if bool(args.fail_on_correctness_mismatch) and correctness_mismatches > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
