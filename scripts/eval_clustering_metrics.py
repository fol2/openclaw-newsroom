#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.eval_dataset import replay_record  # noqa: E402
from newsroom.eval_metrics import (  # noqa: E402
    METRIC_KEYS,
    compute_clustering_quality_metrics,
    evaluate_regression,
    parse_threshold_overrides,
)

DEFAULT_REGRESSION_THRESHOLDS: dict[str, float] = {
    "duplicate_rate_fragmentation": 0.01,
    "misassignment_rate": 0.02,
    "fragmentation_score": 0.01,
    "snowball_sink_metric": 0.03,
    "parse_error_rate": 0.0,
}


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def _build_eval_rows(
    dataset_rows: list[dict[str, Any]],
    *,
    low_score_threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in dataset_rows:
        replay = replay_record(rec, low_score_threshold=float(low_score_threshold))
        labels = rec.get("labels") if isinstance(rec.get("labels"), dict) else {}
        out.append(
            {
                "sample_id": rec.get("sample_id"),
                "expected_target": labels.get("event_id_or_new_event"),
                "predicted_target": replay.get("predicted_target"),
                "parse_ok": bool(replay.get("parse_ok")),
                "parse_error": replay.get("error"),
            }
        )
    return out


def _baseline_thresholds(baseline_obj: dict[str, Any]) -> dict[str, float]:
    base = baseline_obj.get("regression_thresholds")
    if not isinstance(base, dict):
        return dict(DEFAULT_REGRESSION_THRESHOLDS)

    out = dict(DEFAULT_REGRESSION_THRESHOLDS)
    for key in METRIC_KEYS:
        value = base.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def main(argv: list[str]) -> int:
    # Keep CI/log output stable even when old replay rows miss confidence.
    logging.getLogger("newsroom.event_manager").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Compute clustering quality metrics and enforce a baseline regression gate.")
    parser.add_argument(
        "--dataset",
        default=str(OPENCLAW_HOME / "newsroom" / "evals" / "clustering_eval_dataset_v1.jsonl"),
        help="Dataset JSONL path.",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional output path for metrics summary JSON.",
    )
    parser.add_argument(
        "--baseline",
        default="",
        help="Optional baseline JSON path for regression checks.",
    )
    parser.add_argument(
        "--baseline-out",
        default="",
        help="Optional output path for writing/refreshing baseline JSON.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write baseline JSON using current metrics (requires --baseline-out or --baseline).",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Override threshold(s) as metric=value. Repeatable or comma-separated.",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=0.22,
        help="Passed into replay_record for consistency with dataset labels.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when current metrics regress beyond thresholds.",
    )
    parser.add_argument(
        "--allow-dataset-drift",
        action="store_true",
        help="Allow baseline checks when dataset hash differs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting --baseline-out when writing a baseline.",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    dataset_rows = _load_jsonl(dataset_path)
    dataset_sha256 = _sha256_file(dataset_path)
    eval_rows = _build_eval_rows(dataset_rows, low_score_threshold=float(args.low_score_threshold))
    metrics_payload = compute_clustering_quality_metrics(eval_rows)

    generated_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    output: dict[str, Any] = {
        "schema_version": "clustering_eval_metrics_v1",
        "generated_at": generated_at,
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "rows": len(dataset_rows),
        },
        "low_score_threshold": float(args.low_score_threshold),
        **metrics_payload,
    }

    baseline_result: dict[str, Any] = {
        "enabled": False,
        "ok": None,
        "dataset_match": None,
        "dataset_match_reason": None,
        "thresholds": {},
        "regressions": [],
    }

    baseline_obj: dict[str, Any] | None = None
    baseline_path = Path(args.baseline).expanduser() if args.baseline else None
    if baseline_path:
        if not baseline_path.exists():
            raise SystemExit(f"Baseline not found: {baseline_path}")
        loaded = json.loads(baseline_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit(f"Baseline JSON is not an object: {baseline_path}")
        baseline_obj = loaded

    thresholds = dict(DEFAULT_REGRESSION_THRESHOLDS)
    if baseline_obj is not None:
        thresholds = _baseline_thresholds(baseline_obj)
    thresholds.update(parse_threshold_overrides(list(args.threshold or [])))

    if baseline_obj is not None:
        baseline_dataset = baseline_obj.get("dataset") if isinstance(baseline_obj.get("dataset"), dict) else {}
        baseline_sha = baseline_dataset.get("sha256")
        baseline_rows = baseline_dataset.get("rows")

        dataset_match = True
        dataset_match_reason = "ok"
        if baseline_sha and baseline_sha != dataset_sha256:
            dataset_match = False
            dataset_match_reason = "sha256_mismatch"
        if isinstance(baseline_rows, int) and int(baseline_rows) != len(dataset_rows):
            dataset_match = False
            dataset_match_reason = (
                dataset_match_reason if dataset_match_reason != "ok" else "row_count_mismatch"
            )

        baseline_metrics = baseline_obj.get("metrics") if isinstance(baseline_obj.get("metrics"), dict) else {}
        regression = evaluate_regression(
            current_metrics=output["metrics"],
            baseline_metrics=baseline_metrics,
            thresholds=thresholds,
        )

        baseline_ok = bool(regression["ok"])
        if not dataset_match and not bool(args.allow_dataset_drift):
            baseline_ok = False

        baseline_result = {
            "enabled": True,
            "baseline_path": str(baseline_path),
            "ok": baseline_ok,
            "dataset_match": dataset_match,
            "dataset_match_reason": dataset_match_reason if not dataset_match else "ok",
            "thresholds": thresholds,
            "regressions": regression["regressions"],
            "baseline_metrics": baseline_metrics,
        }

        if not dataset_match and not bool(args.allow_dataset_drift):
            baseline_result["regressions"].append(
                {
                    "metric": "dataset_sha256",
                    "current": dataset_sha256,
                    "baseline": baseline_sha,
                    "delta": "mismatch",
                    "allowed_regression": "none",
                }
            )

    output["baseline_check"] = baseline_result

    baseline_out_path: Path | None = None
    if bool(args.write_baseline):
        if args.baseline_out:
            baseline_out_path = Path(args.baseline_out).expanduser()
        elif baseline_path is not None:
            baseline_out_path = baseline_path
        else:
            raise SystemExit("--write-baseline requires --baseline-out or --baseline")

        if baseline_out_path.exists() and not bool(args.overwrite):
            raise SystemExit(f"Baseline output exists: {baseline_out_path} (use --overwrite)")

        baseline_doc = {
            "schema_version": "clustering_eval_metrics_baseline_v1",
            "generated_at": generated_at,
            "dataset": {
                "path": str(dataset_path),
                "sha256": dataset_sha256,
                "rows": len(dataset_rows),
            },
            "low_score_threshold": float(args.low_score_threshold),
            "metrics": output["metrics"],
            "regression_thresholds": thresholds,
        }
        _write_json(baseline_out_path, baseline_doc)
        output["baseline_written_to"] = str(baseline_out_path)

    if args.summary_out:
        _write_json(Path(args.summary_out).expanduser(), output)

    print(json.dumps(output, ensure_ascii=False))

    if bool(args.fail_on_regression):
        if baseline_obj is None:
            raise SystemExit("--fail-on-regression requires --baseline")
        if not bool(output["baseline_check"]["ok"]):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
