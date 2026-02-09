#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.news_pool_db import NewsPoolDB  # noqa: E402
from newsroom.story_index import cluster_links, rank_clusters  # noqa: E402


def _iso(ts: int | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(int(ts), tz=UTC).isoformat(timespec="seconds")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Show current SQLite news pool status + Brave API usage (from pool_runs ledger).")
    parser.add_argument("--db", default=str(OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"), help="SQLite db path.")
    parser.add_argument("--window-hours", type=int, default=48, help="Window for link/domain stats (default: 48).")
    parser.add_argument("--usage-hours", type=int, default=48, help="Window for Brave API usage stats (default: 48).")
    parser.add_argument("--index-max-age-hours", type=int, default=12, help="Index eligibility window in hours (default: 12).")
    parser.add_argument("--index-min-cluster-size", type=int, default=2, help="Index eligibility min sources per cluster (default: 2).")
    parser.add_argument(
        "--states",
        default="hourly,daily",
        help="Comma-separated fetch_state keys to show (default: hourly,daily). Use '*' to show all.",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser()
    now_ts = int(time.time())
    window_h = int(max(1, args.window_hours))
    usage_h = int(max(1, args.usage_hours))
    cutoff_ts = now_ts - window_h * 3600
    usage_cutoff_ts = now_ts - usage_h * 3600

    with NewsPoolDB(path=db_path) as db:
        cur = db._conn.cursor()  # intentional: internal DB utility for fast status queries

        # Story stock: pool -> clustering -> eligible recent clusters.
        index_max_age_h = int(max(1, args.index_max_age_hours))
        index_min_size = int(max(1, args.index_min_cluster_size))
        index_max_age_s = index_max_age_h * 3600
        links_for_index = list(db.iter_links_since(cutoff_ts=cutoff_ts))
        clusters = rank_clusters(cluster_links(links_for_index, now_ts=now_ts), now_ts=now_ts)
        eligible = 0
        breaking_2h = 0
        developing_6h = 0
        for c in clusters:
            if len(getattr(c, "docs", []) or []) < index_min_size:
                continue
            pub_ts = int(getattr(c, "best_published_ts", None) or getattr(c, "last_seen_ts", 0) or 0) or 0
            if pub_ts and (now_ts - pub_ts) > index_max_age_s:
                continue
            eligible += 1
            if pub_ts:
                age_s = max(0, now_ts - pub_ts)
                if age_s <= 2 * 3600:
                    breaking_2h += 1
                if age_s <= 6 * 3600:
                    developing_6h += 1

        links_total = int(cur.execute("SELECT COUNT(1) AS n FROM links").fetchone()["n"])
        links_window = int(cur.execute("SELECT COUNT(1) AS n FROM links WHERE last_seen_ts >= ?", (cutoff_ts,)).fetchone()["n"])
        domain_window = int(
            cur.execute(
                "SELECT COUNT(DISTINCT domain) AS n FROM links WHERE last_seen_ts >= ? AND domain IS NOT NULL AND domain != ''",
                (cutoff_ts,),
            ).fetchone()["n"]
        )

        seen = cur.execute(
            "SELECT MIN(last_seen_ts) AS min_ts, MAX(last_seen_ts) AS max_ts FROM links WHERE last_seen_ts >= ?",
            (cutoff_ts,),
        ).fetchone()
        min_ts = int(seen["min_ts"]) if seen and seen["min_ts"] is not None else None
        max_ts = int(seen["max_ts"]) if seen and seen["max_ts"] is not None else None

        article_cache_total = int(cur.execute("SELECT COUNT(1) AS n FROM article_cache").fetchone()["n"])
        article_cache_window = int(
            cur.execute("SELECT COUNT(1) AS n FROM article_cache WHERE fetched_at_ts >= ?", (cutoff_ts,)).fetchone()["n"]
        )

        # Brave API usage from pool_runs.
        pool_runs_rows = int(cur.execute("SELECT COUNT(1) AS n FROM pool_runs").fetchone()["n"])
        usage_total = int(cur.execute("SELECT COALESCE(SUM(requests_made), 0) AS n FROM pool_runs").fetchone()["n"])
        usage_window = int(
            cur.execute(
                "SELECT COALESCE(SUM(requests_made), 0) AS n FROM pool_runs WHERE run_ts >= ?",
                (usage_cutoff_ts,),
            ).fetchone()["n"]
        )
        usage_24h = int(
            cur.execute(
                "SELECT COALESCE(SUM(requests_made), 0) AS n FROM pool_runs WHERE run_ts >= ?",
                (now_ts - 24 * 3600,),
            ).fetchone()["n"]
        )

        by_state = db.requests_made_by_state(since_ts=usage_cutoff_ts)

        last_call = cur.execute(
            "SELECT run_ts, state_key, query, requests_made, notes FROM pool_runs WHERE requests_made > 0 ORDER BY run_ts DESC LIMIT 1"
        ).fetchone()
        last_call_obj: dict[str, Any] | None = None
        quota_obj: dict[str, Any] | None = None
        if last_call:
            notes_obj = None
            notes = last_call["notes"]
            if isinstance(notes, str) and notes.strip().startswith("{"):
                try:
                    notes_obj = json.loads(notes)
                except Exception:
                    notes_obj = None

            rate_limit = None
            keys_used = None
            if isinstance(notes_obj, dict):
                rate_limit = notes_obj.get("rate_limit")
                keys_used = notes_obj.get("keys_used")

            if isinstance(rate_limit, dict):
                lim = rate_limit.get("limit")
                rem = rate_limit.get("remaining")
                rst = rate_limit.get("reset")

                def pick(v: Any, i: int) -> int | None:
                    if isinstance(v, list) and len(v) > i and isinstance(v[i], int):
                        return int(v[i])
                    return None

                per_sec_limit = pick(lim, 0)
                monthly_limit = pick(lim, 1)
                per_sec_remaining = pick(rem, 0)
                monthly_remaining = pick(rem, 1)
                per_sec_reset_s = pick(rst, 0)
                monthly_reset_s = pick(rst, 1)

                quota_obj = {
                    "per_second": {
                        "limit": per_sec_limit,
                        "remaining": per_sec_remaining,
                        "reset_seconds": per_sec_reset_s,
                    },
                    "monthly": {
                        "limit": monthly_limit,
                        "remaining": monthly_remaining,
                        "reset_seconds": monthly_reset_s,
                        "reset_at": (_iso(int(last_call["run_ts"]) + int(monthly_reset_s)) if monthly_reset_s else None),
                    },
                }

            last_call_obj = {
                "run_at": _iso(int(last_call["run_ts"])),
                "state_key": str(last_call["state_key"]),
                "query": last_call["query"],
                "requests_made": int(last_call["requests_made"]),
            }
            if isinstance(keys_used, list):
                last_call_obj["keys_used"] = keys_used

        # fetch_state snapshot
        wanted = str(args.states or "").strip()
        states: list[str] = []
        if wanted == "*":
            rows = cur.execute("SELECT key FROM fetch_state ORDER BY key ASC").fetchall()
            states = [str(r["key"]) for r in rows]
        else:
            states = [s.strip() for s in wanted.split(",") if s.strip()]

        fetch_state_obj: dict[str, Any] = {}
        for k in states:
            st = db.fetch_state(k)
            fetch_state_obj[k] = {
                "run_count": int(st.get("run_count", 0)),
                "last_fetch_at": _iso(int(st.get("last_fetch_ts", 0)) or None),
                "last_offset": int(st.get("last_offset", 1)),
            }

    # Brave key quota state (best-effort; populated only after at least one real network call).
    key_state_path = OPENCLAW_HOME / "data" / "newsroom" / "brave_key_state.json"
    key_state_rows: list[dict[str, Any]] = []
    try:
        obj = json.loads(key_state_path.read_text(encoding="utf-8"))
    except Exception:
        obj = None
    if isinstance(obj, dict):
        keys_obj = obj.get("keys")
        if isinstance(keys_obj, dict):
            for key_id, row in keys_obj.items():
                if not isinstance(row, dict):
                    continue
                key_state_rows.append(
                    {
                        "key_id": str(key_id),
                        "label": row.get("label"),
                        "last_seen_at": _iso(int(row.get("last_seen_ts")) if row.get("last_seen_ts") is not None else None),
                        "monthly_limit": row.get("monthly_limit"),
                        "monthly_remaining": row.get("monthly_remaining"),
                        "monthly_reset_at": _iso(int(row.get("monthly_reset_at_ts")) if row.get("monthly_reset_at_ts") is not None else None),
                        "exhausted_until": _iso(int(row.get("exhausted_until_ts")) if row.get("exhausted_until_ts") is not None else None),
                    }
                )
    key_state_rows.sort(key=lambda r: ((r.get("label") or ""), (r.get("key_id") or "")))

    out = {
        "ok": True,
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "db": str(db_path),
        "pool": {
            "window_hours": window_h,
            "links_total": links_total,
            "links_window": links_window,
            "domains_window": domain_window,
            "seen_window": {"min_last_seen_at": _iso(min_ts), "max_last_seen_at": _iso(max_ts)},
            "article_cache_total": article_cache_total,
            "article_cache_window": article_cache_window,
        },
        "index": {
            "window_hours": window_h,
            "clusters_total": len(clusters),
            "eligible_clusters": eligible,
            "breaking_clusters_2h": breaking_2h,
            "developing_clusters_6h": developing_6h,
            "eligibility": {"max_age_hours": index_max_age_h, "min_cluster_size": index_min_size},
        },
        "brave_api": {
            "usage_hours": usage_h,
            "ledger_rows": pool_runs_rows,
            "note": ("pool_runs ledger is empty; call counts start after next pool update run" if pool_runs_rows == 0 else None),
            "calls_total": usage_total,
            "calls_usage_window": usage_window,
            "calls_24h": usage_24h,
            "calls_by_state_usage_window": [{"state_key": k, "calls": n} for k, n in by_state],
            "last_network_call": last_call_obj,
            "quota": quota_obj,
            "key_state": key_state_rows,
        },
        "fetch_state": fetch_state_obj,
    }

    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
