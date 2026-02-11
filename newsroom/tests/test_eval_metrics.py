from __future__ import annotations

import hashlib
import json
from pathlib import Path

from newsroom.eval_metrics import (
    compute_clustering_quality_metrics,
    evaluate_regression,
    parse_threshold_overrides,
)
from scripts.eval_clustering_metrics import main as eval_main


def _base_replay_row(*, expected_target: int | str) -> dict:
    return {
        "schema_version": "clustering_eval_sample_v1",
        "sample_id": "sample_0001",
        "labels": {
            "event_id_or_new_event": expected_target,
            "language": "en",
            "correctness": "correct",
            "flags": [],
        },
        "replay": {
            "link": {
                "id": 1,
                "url": "https://example.com/a",
                "norm_url": "https://example.com/a",
                "title": "Government policy update",
                "description": "Cabinet confirms policy package and timeline",
                "lang_hint": "en",
            },
            "candidate_events": [
                {
                    "id": 101,
                    "status": "new",
                    "summary_en": "Government policy update and next steps",
                }
            ],
            "raw_llm_response": {
                "action": "assign",
                "event_id": 101,
                "confidence": 0.91,
                "summary_en": "Government policy update and next steps",
                "match_basis": ["entity_overlap", "timeline_match"],
                "link_flags": [],
            },
        },
    }


def test_compute_clustering_quality_metrics_core_rates() -> None:
    rows = [
        {"sample_id": "s1", "expected_target": 10, "predicted_target": 10, "parse_ok": True},
        {"sample_id": "s2", "expected_target": 10, "predicted_target": "new_event", "parse_ok": True},
        {"sample_id": "s3", "expected_target": 11, "predicted_target": 12, "parse_ok": True},
        {"sample_id": "s4", "expected_target": "new_event", "predicted_target": 12, "parse_ok": True},
        {"sample_id": "s5", "expected_target": 11, "predicted_target": None, "parse_ok": False},
    ]

    report = compute_clustering_quality_metrics(rows)
    metrics = report["metrics"]
    counts = report["counts"]

    assert report["rows"] == 5
    assert counts["parse_error_rows"] == 1
    assert counts["expected_existing_rows"] == 4
    assert counts["predicted_assign_rows"] == 3
    assert counts["duplicate_fragmentation_rows"] == 1
    assert counts["misassignment_rows"] == 2

    assert metrics["duplicate_rate_fragmentation"] == 0.25
    assert metrics["misassignment_rate"] == (2.0 / 3.0)
    assert metrics["fragmentation_score"] == 0.5
    assert metrics["snowball_sink_metric"] == (2.0 / 3.0)
    assert metrics["parse_error_rate"] == 0.2

    assert counts["snowball_sink_event_id"] == 12
    assert counts["snowball_sink_assign_count"] == 2
    assert counts["fragmented_events"] == 2


def test_parse_threshold_overrides_supports_repeatable_and_comma_tokens() -> None:
    overrides = parse_threshold_overrides(
        [
            "misassignment_rate=0.05",
            "fragmentation_score=0.02,snowball_sink_metric=0.03",
        ]
    )

    assert overrides == {
        "misassignment_rate": 0.05,
        "fragmentation_score": 0.02,
        "snowball_sink_metric": 0.03,
    }


def test_parse_threshold_overrides_rejects_unknown_metric() -> None:
    try:
        parse_threshold_overrides(["unknown_metric=0.1"])
    except ValueError as exc:
        assert "Unknown threshold metric" in str(exc)
        return
    raise AssertionError("Expected ValueError for unknown threshold metric")


def test_evaluate_regression_respects_allowed_slack() -> None:
    result = evaluate_regression(
        current_metrics={
            "duplicate_rate_fragmentation": 0.03,
            "misassignment_rate": 0.12,
            "fragmentation_score": 0.02,
            "snowball_sink_metric": 0.20,
            "parse_error_rate": 0.00,
        },
        baseline_metrics={
            "duplicate_rate_fragmentation": 0.02,
            "misassignment_rate": 0.10,
            "fragmentation_score": 0.02,
            "snowball_sink_metric": 0.19,
            "parse_error_rate": 0.00,
        },
        thresholds={
            "duplicate_rate_fragmentation": 0.01,
            "misassignment_rate": 0.01,
            "fragmentation_score": 0.00,
            "snowball_sink_metric": 0.00,
            "parse_error_rate": 0.00,
        },
    )

    assert result["ok"] is False
    metrics = {row["metric"] for row in result["regressions"]}
    assert metrics == {"misassignment_rate", "snowball_sink_metric"}


def test_eval_metrics_cli_writes_baseline(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    baseline_path = tmp_path / "baseline.json"

    row = _base_replay_row(expected_target=101)
    dataset_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    rc = eval_main(
        [
            "--dataset",
            str(dataset_path),
            "--baseline-out",
            str(baseline_path),
            "--write-baseline",
            "--overwrite",
        ]
    )

    assert rc == 0
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "clustering_eval_metrics_baseline_v1"
    assert baseline["dataset"]["rows"] == 1
    assert "misassignment_rate" in baseline["metrics"]


def test_eval_metrics_cli_fails_on_metric_regression(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    baseline_path = tmp_path / "baseline.json"

    row = _base_replay_row(expected_target="new_event")
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    dataset_path.write_text(payload, encoding="utf-8")

    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    baseline = {
        "schema_version": "clustering_eval_metrics_baseline_v1",
        "generated_at": "2026-02-11T00:00:00+00:00",
        "dataset": {"path": str(dataset_path), "sha256": sha, "rows": 1},
        "low_score_threshold": 0.22,
        "metrics": {
            "duplicate_rate_fragmentation": 0.0,
            "misassignment_rate": 0.0,
            "fragmentation_score": 0.0,
            "snowball_sink_metric": 0.0,
            "parse_error_rate": 0.0,
        },
        "regression_thresholds": {
            "duplicate_rate_fragmentation": 0.0,
            "misassignment_rate": 0.0,
            "fragmentation_score": 0.0,
            "snowball_sink_metric": 0.0,
            "parse_error_rate": 0.0,
        },
    }
    baseline_path.write_text(json.dumps(baseline, ensure_ascii=False), encoding="utf-8")

    rc = eval_main(
        [
            "--dataset",
            str(dataset_path),
            "--baseline",
            str(baseline_path),
            "--fail-on-regression",
        ]
    )
    assert rc == 1
