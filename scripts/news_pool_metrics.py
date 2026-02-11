#!/usr/bin/env python3
from __future__ import annotations

"""Deterministic metrics for the SQLite newsroom pool database.

This script is intended for one-off maintenance and audit tasks. It does not
mutate the database.

It can optionally write:
  - A JSON metrics report (machine-readable).
  - A Markdown summary (runbook-friendly).
  - A snapshot of posted-event identity fields (hashed) for before/after safety checks.

Percentiles use the nearest-rank method.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


def _default_db_path() -> Path:
    env_home = os.environ.get("OPENCLAW_HOME")
    if env_home:
        candidate = Path(env_home).expanduser() / "data" / "newsroom" / "news_pool.sqlite3"
        if candidate.exists():
            return candidate
    return Path.home() / ".openclaw" / "data" / "newsroom" / "news_pool.sqlite3"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_one_int(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    if not row:
        return 0
    v = row[0]
    return int(v) if v is not None else 0


def _fetch_one_float(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> float | None:
    row = conn.execute(sql, params).fetchone()
    if not row:
        return None
    v = row[0]
    return float(v) if v is not None else None


def _sha256_hex(v: str | None) -> str | None:
    if v is None:
        return None
    s = str(v)
    if not s:
        return None
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _percentile_nearest_rank(values_sorted: list[int], p: float) -> int | None:
    """Nearest-rank percentile.

    Args:
        values_sorted: Ascending list of ints.
        p: 0.0-1.0.
    """
    if not values_sorted:
        return None
    if p <= 0:
        return int(values_sorted[0])
    if p >= 1:
        return int(values_sorted[-1])

    # Nearest-rank definition.
    # https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method
    import math

    n = len(values_sorted)
    k = int(math.ceil(float(p) * float(n))) - 1
    k = max(0, min(n - 1, k))
    return int(values_sorted[k])


def _category_distribution(conn: sqlite3.Connection, *, root_only: bool) -> list[dict[str, Any]]:
    where = "WHERE parent_event_id IS NULL" if root_only else ""
    rows = conn.execute(
        f"""
        SELECT COALESCE(category, '(null)') AS category, COUNT(1) AS n
        FROM events
        {where}
        GROUP BY COALESCE(category, '(null)')
        ORDER BY n DESC, category ASC
        """
    ).fetchall()
    return [{"category": str(r["category"]), "count": int(r["n"])} for r in rows]


def _load_posted_events(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id AS event_id, posted_at_ts, thread_id, run_id
        FROM events
        WHERE status = 'posted'
        ORDER BY id ASC
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "event_id": int(r["event_id"]),
                "posted_at_ts": (int(r["posted_at_ts"]) if r["posted_at_ts"] is not None else None),
                "thread_id": (str(r["thread_id"]) if r["thread_id"] is not None else None),
                "run_id": (str(r["run_id"]) if r["run_id"] is not None else None),
            }
        )
    return out


def _load_root_link_counts(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        "SELECT link_count FROM events WHERE parent_event_id IS NULL ORDER BY link_count ASC"
    ).fetchall()
    out: list[int] = []
    for r in rows:
        try:
            out.append(int(r["link_count"]))
        except Exception:
            continue
    return out


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _markdown_summary(report: dict[str, Any]) -> str:
    db_path = report.get("db_path") or ""
    generated_at = report.get("generated_at_utc") or ""

    ev = report.get("events") or {}
    posted_check = report.get("posted_integrity") or {}

    avg_lc = ev.get("avg_link_count_root")
    avg_lc_str = f"{avg_lc:.3f}" if isinstance(avg_lc, (int, float)) else "NULL"

    lines: list[str] = []
    lines.append(f"DB: `{db_path}`")
    lines.append(f"Generated at (UTC): `{generated_at}`")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total events | {int(ev.get('total_events', 0))} |")
    lines.append(f"| Root events total | {int(ev.get('root_events_total', 0))} |")
    lines.append(f"| Root events posted | {int(ev.get('root_events_posted', 0))} |")
    lines.append(f"| Root events unposted | {int(ev.get('root_events_unposted', 0))} |")
    lines.append(f"| Avg link_count (root events) | {avg_lc_str} |")
    lines.append(f"| p50 link_count (root events) | {str(ev.get('p50_link_count_root') or 'NULL')} |")
    lines.append(f"| p90 link_count (root events) | {str(ev.get('p90_link_count_root') or 'NULL')} |")
    lines.append(f"| Max link_count (root events) | {str(ev.get('max_link_count_root') or 'NULL')} |")
    lines.append(f"| Posted events | {int(ev.get('posted_events', 0))} |")
    lines.append(f"| Posted missing thread_id | {int(ev.get('posted_missing_thread_id', 0))} |")
    lines.append(f"| Posted missing run_id | {int(ev.get('posted_missing_run_id', 0))} |")
    lines.append("")

    if posted_check.get("compare_enabled"):
        ok = bool(posted_check.get("ok"))
        lines.append(f"Posted event identity check: {'OK' if ok else 'FAIL'}")
        lines.append(
            f"Baseline posted events: {int(posted_check.get('baseline_posted_events', 0))}"
        )
        lines.append(
            f"Current posted events: {int(posted_check.get('current_posted_events', 0))}"
        )
        missing = posted_check.get("missing_event_ids") or []
        changed = posted_check.get("changed") or []
        if missing:
            lines.append("")
            lines.append("Missing posted event IDs:")
            for eid in missing[:50]:
                lines.append(f"- {int(eid)}")
            if len(missing) > 50:
                lines.append(f"- ... ({len(missing) - 50} more)")
        if changed:
            lines.append("")
            lines.append("Changed posted event fields:")
            for ch in changed[:50]:
                lines.append(
                    f"- event_id={ch.get('event_id')} field={ch.get('field')} before={ch.get('before')!r} after={ch.get('after')!r}"
                )
            if len(changed) > 50:
                lines.append(f"- ... ({len(changed) - 50} more)")
        lines.append("")

    lines.append("Category distribution (root events):")
    lines.append("")
    lines.append("| Category | Root events |")
    lines.append("|---|---:|")
    for row in (report.get("category_distribution_root") or []):
        cat = str(row.get("category") or "")
        n = int(row.get("count") or 0)
        lines.append(f"| {cat} | {n} |")

    lines.append("")
    return "\n".join(lines)


def _posted_snapshot_compare(
    *,
    baseline: dict[str, Any],
    current_posted: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_events = baseline.get("posted_events") or []
    base_by_id: dict[int, dict[str, Any]] = {}
    for ev in baseline_events:
        try:
            eid = int(ev.get("event_id"))
        except Exception:
            continue
        base_by_id[eid] = ev

    cur_by_id: dict[int, dict[str, Any]] = {int(ev["event_id"]): ev for ev in current_posted}

    missing: list[int] = []
    changed: list[dict[str, Any]] = []

    for eid, b in sorted(base_by_id.items(), key=lambda kv: kv[0]):
        c = cur_by_id.get(eid)
        if not c:
            missing.append(eid)
            continue

        for field in ("thread_id", "run_id"):
            hash_key = f"{field}_sha256"
            if hash_key in b:
                before = b.get(hash_key)
                after = _sha256_hex(c.get(field))
            else:
                before = b.get(field)
                after = c.get(field)

            if before != after:
                changed.append(
                    {
                        "event_id": eid,
                        "field": field,
                        "before": before,
                        "after": after,
                    }
                )

    ok = (not missing) and (not changed)
    return {
        "ok": ok,
        "missing_event_ids": missing,
        "changed": changed,
        "baseline_posted_events": len(base_by_id),
        "current_posted_events": len(cur_by_id),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compute deterministic metrics for the newsroom news pool DB.")
    parser.add_argument("--db-path", default=str(_default_db_path()), help="Path to news_pool.sqlite3.")
    parser.add_argument("--out-json", default=None, help="Write the JSON report to this path (optional).")
    parser.add_argument("--out-md", default=None, help="Write a Markdown summary to this path (optional).")
    parser.add_argument(
        "--posted-snapshot-out",
        default=None,
        help=(
            "Write a snapshot of posted events (event_id + hashed thread_id/run_id) to this path (optional)."
        ),
    )
    parser.add_argument(
        "--compare-posted-snapshot",
        default=None,
        help="Compare current DB posted events against a baseline snapshot JSON file (optional).",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db_path).expanduser()
    if not db_path.exists():
        print(f"ERROR: db does not exist: {db_path}", file=sys.stderr)
        return 2

    now_ts = int(time.time())
    generated_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))

    with _connect(db_path) as conn:
        row = conn.execute("SELECT v FROM meta WHERE k = 'schema_version'").fetchone()
        schema_version_str = str(row["v"]) if row and row["v"] is not None else None

        total_events = _fetch_one_int(conn, "SELECT COUNT(1) FROM events")
        posted_events = _fetch_one_int(conn, "SELECT COUNT(1) FROM events WHERE status = 'posted'")

        root_events_total = _fetch_one_int(conn, "SELECT COUNT(1) FROM events WHERE parent_event_id IS NULL")
        root_events_posted = _fetch_one_int(
            conn,
            "SELECT COUNT(1) FROM events WHERE parent_event_id IS NULL AND status = 'posted'",
        )
        root_events_unposted = _fetch_one_int(
            conn,
            "SELECT COUNT(1) FROM events WHERE parent_event_id IS NULL AND status IN ('new', 'active')",
        )

        root_link_counts = _load_root_link_counts(conn)
        avg_link_count_root = _fetch_one_float(conn, "SELECT AVG(link_count) FROM events WHERE parent_event_id IS NULL")
        p50_link_count_root = _percentile_nearest_rank(root_link_counts, 0.50)
        p90_link_count_root = _percentile_nearest_rank(root_link_counts, 0.90)
        max_link_count_root = _fetch_one_int(
            conn,
            "SELECT COALESCE(MAX(link_count), 0) FROM events WHERE parent_event_id IS NULL",
        )

        posted_missing_thread_id = _fetch_one_int(
            conn,
            "SELECT COUNT(1) FROM events WHERE status='posted' AND (thread_id IS NULL OR TRIM(thread_id) = '')",
        )
        posted_missing_run_id = _fetch_one_int(
            conn,
            "SELECT COUNT(1) FROM events WHERE status='posted' AND (run_id IS NULL OR TRIM(run_id) = '')",
        )

        category_root = _category_distribution(conn, root_only=True)
        category_all = _category_distribution(conn, root_only=False)

        posted_rows = _load_posted_events(conn)

    posted_integrity: dict[str, Any] = {
        "compare_enabled": False,
        "ok": None,
        "missing_event_ids": [],
        "changed": [],
        "baseline_posted_events": None,
        "current_posted_events": None,
    }

    if args.compare_posted_snapshot:
        baseline_path = Path(args.compare_posted_snapshot).expanduser()
        baseline_obj = json.loads(baseline_path.read_text(encoding="utf-8"))
        cmp_result = _posted_snapshot_compare(baseline=baseline_obj, current_posted=posted_rows)
        posted_integrity = {
            "compare_enabled": True,
            "baseline_path": str(baseline_path),
            **cmp_result,
        }

    report: dict[str, Any] = {
        "schema_version": "news_pool_metrics_v1",
        "db_path": str(db_path),
        "generated_at_ts": now_ts,
        "generated_at_utc": generated_at_utc,
        "db": {"schema_version": schema_version_str},
        "events": {
            "total_events": total_events,
            "root_events_total": root_events_total,
            "root_events_posted": root_events_posted,
            "root_events_unposted": root_events_unposted,
            "avg_link_count_root": avg_link_count_root,
            "p50_link_count_root": p50_link_count_root,
            "p90_link_count_root": p90_link_count_root,
            "max_link_count_root": max_link_count_root,
            "posted_events": posted_events,
            "posted_missing_thread_id": posted_missing_thread_id,
            "posted_missing_run_id": posted_missing_run_id,
        },
        "category_distribution_root": category_root,
        "category_distribution_all": category_all,
        "posted_integrity": posted_integrity,
    }

    if args.posted_snapshot_out:
        snapshot_path = Path(args.posted_snapshot_out).expanduser()
        posted_events_hashed: list[dict[str, Any]] = []
        for ev in posted_rows:
            posted_events_hashed.append(
                {
                    "event_id": int(ev.get("event_id") or 0),
                    "posted_at_ts": ev.get("posted_at_ts"),
                    "thread_id_sha256": _sha256_hex(ev.get("thread_id")),
                    "run_id_sha256": _sha256_hex(ev.get("run_id")),
                }
            )

        snapshot_obj = {
            "schema_version": "news_pool_posted_snapshot_v2",
            "db_path": str(db_path),
            "captured_at_ts": now_ts,
            "captured_at_utc": generated_at_utc,
            "posted_events": posted_events_hashed,
        }
        _write_json(snapshot_path, snapshot_obj)

    if args.out_json:
        _write_json(Path(args.out_json).expanduser(), report)

    if args.out_md:
        md_path = Path(args.out_md).expanduser()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_markdown_summary(report) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
