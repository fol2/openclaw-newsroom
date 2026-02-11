from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Literal

EvalTarget = int | Literal["new_event"]

METRIC_KEYS: tuple[str, ...] = (
    "duplicate_rate_fragmentation",
    "misassignment_rate",
    "fragmentation_score",
    "snowball_sink_metric",
    "parse_error_rate",
)


def coerce_target(value: Any) -> EvalTarget | None:
    """Normalise event targets from dataset/replay payloads."""
    if value == "new_event":
        return "new_event"
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        text = value.strip()
        if text == "new_event":
            return "new_event"
        if text.isdigit():
            parsed = int(text)
            if parsed > 0:
                return parsed
    return None


def parse_threshold_overrides(tokens: list[str] | None) -> dict[str, float]:
    """Parse repeated/comma-separated `metric=value` threshold tokens."""
    overrides: dict[str, float] = {}
    for raw in tokens or []:
        for part in str(raw).split(","):
            token = part.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(f"Invalid threshold token: {token!r} (expected metric=value)")
            metric, value = token.split("=", 1)
            key = metric.strip()
            if key not in METRIC_KEYS:
                raise ValueError(f"Unknown threshold metric: {key!r}")
            try:
                parsed = float(value.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid numeric threshold value in token: {token!r}") from exc
            if parsed < 0:
                raise ValueError(f"Threshold must be >= 0 for metric {key!r}")
            overrides[key] = parsed
    return overrides


def compute_clustering_quality_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute evaluation metrics from replay rows.

    Each row should provide:
    - `expected_target`: int or `"new_event"` (or value coercible by `coerce_target`)
    - `predicted_target`: int or `"new_event"` (or value coercible by `coerce_target`)
    - `parse_ok`: bool
    """
    normalised: list[dict[str, Any]] = []
    for row in rows:
        expected = coerce_target(row.get("expected_target"))
        predicted = coerce_target(row.get("predicted_target"))
        parse_ok = bool(row.get("parse_ok"))
        if not parse_ok:
            predicted = None
        normalised.append(
            {
                "sample_id": row.get("sample_id"),
                "expected_target": expected,
                "predicted_target": predicted,
                "parse_ok": parse_ok,
            }
        )

    total_rows = len(normalised)
    parse_ok_rows = sum(1 for row in normalised if bool(row["parse_ok"]))
    parse_error_rows = total_rows - parse_ok_rows

    expected_existing_rows = [row for row in normalised if isinstance(row["expected_target"], int)]
    predicted_assign_rows = [row for row in normalised if isinstance(row["predicted_target"], int)]

    duplicate_fragmentation_rows = [
        row
        for row in expected_existing_rows
        if row["predicted_target"] == "new_event"
    ]
    misassignment_rows = [
        row
        for row in predicted_assign_rows
        if row["expected_target"] != row["predicted_target"]
    ]

    # Event-level fragmentation:
    # for each expected event, measure how widely rows get split across predictions.
    preds_by_expected: dict[int, list[EvalTarget | Literal["parse_error"]]] = defaultdict(list)
    for row in expected_existing_rows:
        expected = row["expected_target"]
        if not isinstance(expected, int):
            continue
        predicted = row["predicted_target"] if row["parse_ok"] else "parse_error"
        if not isinstance(predicted, int) and predicted not in {"new_event", "parse_error"}:
            predicted = "parse_error"
        preds_by_expected[expected].append(predicted)

    dominant_sum = 0
    fragmented_events = 0
    for preds in preds_by_expected.values():
        if not preds:
            continue
        counts = Counter(preds)
        dominant_sum += int(max(counts.values()))
        if len(counts) > 1:
            fragmented_events += 1

    expected_existing_count = len(expected_existing_rows)
    fragmentation_score = (
        1.0 - (float(dominant_sum) / float(expected_existing_count))
        if expected_existing_count > 0
        else 0.0
    )

    # Snowball sink proxy: assignment concentration in one sink event.
    assign_counts = Counter(int(row["predicted_target"]) for row in predicted_assign_rows)
    sink_event_id: int | None = None
    sink_assign_count = 0
    if assign_counts:
        sink_event_id, sink_assign_count = assign_counts.most_common(1)[0]
    predicted_assign_count = len(predicted_assign_rows)
    snowball_sink_metric = (
        float(sink_assign_count) / float(predicted_assign_count)
        if predicted_assign_count > 0
        else 0.0
    )

    metrics = {
        "duplicate_rate_fragmentation": (
            float(len(duplicate_fragmentation_rows)) / float(expected_existing_count)
            if expected_existing_count > 0
            else 0.0
        ),
        "misassignment_rate": (
            float(len(misassignment_rows)) / float(predicted_assign_count)
            if predicted_assign_count > 0
            else 0.0
        ),
        "fragmentation_score": fragmentation_score,
        "snowball_sink_metric": snowball_sink_metric,
        "parse_error_rate": (float(parse_error_rows) / float(total_rows)) if total_rows > 0 else 0.0,
    }

    return {
        "rows": total_rows,
        "metrics": metrics,
        "counts": {
            "parse_ok_rows": parse_ok_rows,
            "parse_error_rows": parse_error_rows,
            "expected_existing_rows": expected_existing_count,
            "predicted_assign_rows": predicted_assign_count,
            "duplicate_fragmentation_rows": len(duplicate_fragmentation_rows),
            "misassignment_rows": len(misassignment_rows),
            "fragmented_events": fragmented_events,
            "snowball_sink_event_id": sink_event_id,
            "snowball_sink_assign_count": sink_assign_count,
            "snowball_sink_unique_assigned_events": len(assign_counts),
        },
    }


def evaluate_regression(
    *,
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Compare current metrics against baseline + allowed regression thresholds."""
    regressions: list[dict[str, Any]] = []
    for key in METRIC_KEYS:
        current = float(current_metrics.get(key, 0.0))
        baseline = float(baseline_metrics.get(key, 0.0))
        allowed = float(thresholds.get(key, 0.0))
        delta = current - baseline
        if delta > allowed:
            regressions.append(
                {
                    "metric": key,
                    "current": current,
                    "baseline": baseline,
                    "delta": delta,
                    "allowed_regression": allowed,
                }
            )

    return {
        "ok": not regressions,
        "regressions": regressions,
    }
