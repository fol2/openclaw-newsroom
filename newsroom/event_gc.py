from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .news_pool_db import NewsPoolDB


def run_periodic_unposted_event_gc(
    *,
    db_path: Path,
    state_key: str,
    min_interval_seconds: int,
    stale_after_hours: int = 96,
    low_link_max: int = 1,
    low_quality_summary_chars: int = 80,
    now_ts: int | None = None,
) -> dict[str, Any]:
    """Run periodic stale-event GC and return a compact execution summary."""
    now = int(time.time()) if now_ts is None else int(now_ts)
    interval = max(0, int(min_interval_seconds))
    key = str(state_key).strip() or "event_gc_unposted"

    with NewsPoolDB(path=Path(db_path).expanduser()) as db:
        state = db.fetch_state(key)
        last_gc_ts = int(state.get("last_fetch_ts", 0))
        run_count = int(state.get("run_count", 0))
        should_gc = (now - last_gc_ts) >= interval

        out: dict[str, Any] = {
            "ok": True,
            "state_key": key,
            "should_gc": bool(should_gc),
            "run_count": run_count,
            "last_gc_ts": last_gc_ts,
            "criteria": {
                "stale_after_hours": int(stale_after_hours),
                "low_link_max": int(low_link_max),
                "low_quality_summary_chars": int(low_quality_summary_chars),
            },
        }

        if not should_gc:
            out["pruned"] = 0
            out["reason"] = "min_interval"
            return out

        pruned = db.prune_stale_unposted_events(
            stale_after_hours=int(stale_after_hours),
            low_link_max=int(low_link_max),
            low_quality_summary_chars=int(low_quality_summary_chars),
            now_ts=now,
        )
        next_run = run_count + 1
        db.update_fetch_state(key=key, last_fetch_ts=now, last_offset=0, run_count=next_run)
        out["pruned"] = int(pruned)
        out["run_count"] = int(next_run)
        out["last_gc_ts"] = int(now)
        return out
