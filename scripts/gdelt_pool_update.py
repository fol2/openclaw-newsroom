#!/usr/bin/env python3
"""Update the news pool from GDELT DOC 2.0 API (free, no auth).

Same pattern as news_pool_update.py: argparse → fetch_state rotation →
fetch → PoolLink → upsert → log_pool_run → print JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.brave_news import normalize_url  # noqa: E402
from newsroom.gdelt_news import fetch_gdelt_articles  # noqa: E402
from newsroom.news_pool_db import NewsPoolDB, PoolLink  # noqa: E402

DEFAULT_QUERIES = [
    "hong kong",
    "united kingdom politics parliament",
    "artificial intelligence AI",
    "stocks earnings nasdaq wall street",
    "entertainment celebrity film",
    "sports tournament",
]


def _domain(url: str) -> str | None:
    try:
        return urlsplit(url).hostname
    except Exception:
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Update the news pool from GDELT DOC 2.0 API.")
    parser.add_argument("--db", default=str(OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"), help="SQLite db path.")
    parser.add_argument("--hours", type=int, default=48, help="Retention window in hours (default: 48).")
    parser.add_argument("--state-key", default="gdelt", help="Rotation key for fetch_state table.")
    parser.add_argument("--min-interval-seconds", type=int, default=900, help="Minimum seconds between API fetches.")
    parser.add_argument("--query", action="append", default=[], help="GDELT query (repeatable). If omitted, uses defaults.")
    parser.add_argument("--maxrecords", type=int, default=75, help="Max records per query (default: 75).")
    parser.add_argument("--timespan", default="24h", help="GDELT timespan (default: 24h).")
    parser.add_argument("--sourcelang", default="english", help="GDELT source language (default: english).")
    parser.add_argument("--cache-ttl-seconds", type=int, default=900, help="File cache TTL seconds (default: 900).")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser()
    hours = int(max(1, args.hours))
    cutoff_ts = int(time.time() - hours * 3600)

    queries = [q.strip() for q in (args.query or []) if isinstance(q, str) and q.strip()]
    if not queries:
        queries = list(DEFAULT_QUERIES)

    maxrecords = int(max(1, min(250, args.maxrecords)))
    timespan = str(args.timespan).strip() or "24h"
    sourcelang = str(args.sourcelang).strip() or None

    cache_dir = OPENCLAW_HOME / "data" / "newsroom" / "gdelt_cache"

    now_ts = int(time.time())
    with NewsPoolDB(path=db_path) as db:
        pruned = db.prune_links(cutoff_ts=cutoff_ts)
        pruned_articles = db.prune_article_cache(cutoff_ts=cutoff_ts)

        state_key = str(args.state_key).strip() or "gdelt"
        state = db.fetch_state(state_key)
        last_fetch_ts = int(state.get("last_fetch_ts", 0))
        run_count = int(state.get("run_count", 0))

        should_fetch = (now_ts - last_fetch_ts) >= int(max(0, args.min_interval_seconds))

        fetch_summary: dict[str, Any] = {"should_fetch": should_fetch}
        upsert = {"inserted": 0, "updated": 0}
        notes: str | None = None

        if should_fetch:
            next_run = run_count + 1
            # Rotate through queries.
            q_idx = (next_run - 1) % len(queries)
            q = queries[q_idx]

            requests_made = 0
            total_results = 0
            links: list[PoolLink] = []
            last_ts = 0.0

            fetched, last_ts = fetch_gdelt_articles(
                query=q,
                maxrecords=maxrecords,
                timespan=timespan,
                sourcelang=sourcelang,
                cache_dir=cache_dir,
                ttl_seconds=int(args.cache_ttl_seconds),
                last_request_ts=last_ts,
            )
            requests_made += int(fetched.requests_made)
            total_results += len(fetched.results)

            for r in fetched.results:
                url = r.get("url")
                if not isinstance(url, str) or not url.strip():
                    continue
                url = normalize_url(url)
                if not url:
                    continue
                domain = r.get("domain")
                if not isinstance(domain, str) or not domain.strip():
                    domain = _domain(url)
                links.append(
                    PoolLink(
                        url=url,
                        norm_url=url,  # already normalized
                        domain=str(domain).lower() if isinstance(domain, str) and domain else None,
                        title=r.get("title") if isinstance(r.get("title"), str) else None,
                        description=None,  # GDELT ArtList has no description
                        age=None,
                        page_age=r.get("page_age") if isinstance(r.get("page_age"), str) else None,
                        query=f"gdelt:{q}",
                        offset=0,
                        fetched_at_ts=now_ts,
                    )
                )

            upsert = db.upsert_links(links, now_ts=now_ts)
            db.update_fetch_state(key=state_key, last_fetch_ts=now_ts, last_offset=0, run_count=next_run)

            fetch_summary = {
                "should_fetch": True,
                "query": q,
                "gdelt_query_prefix": f"gdelt:{q}",
                "requests_made": int(requests_made),
                "results": int(total_results),
                "cached": bool(fetched.cached),
                "run_count": int(next_run),
            }
            if fetched.error:
                fetch_summary["error"] = fetched.error
            notes = json.dumps({"source": "gdelt", "query": q, "ok": fetched.ok}, ensure_ascii=True, separators=(",", ":"))
        else:
            fetch_summary = {
                "should_fetch": False,
                "reason": "min_interval",
                "last_fetch_ts": int(last_fetch_ts),
                "run_count": int(run_count),
            }
            notes = json.dumps({"source": "gdelt", "reason": "min_interval"}, ensure_ascii=True, separators=(",", ":"))

        db.log_pool_run(
            run_ts=now_ts,
            state_key=state_key,
            window_hours=hours,
            should_fetch=bool(should_fetch),
            query=str(fetch_summary.get("query")) if should_fetch and isinstance(fetch_summary.get("query"), str) else None,
            offset_start=0,
            pages=1,
            count=maxrecords,
            freshness=timespan,
            requests_made=int(fetch_summary.get("requests_made") or 0) if should_fetch else 0,
            results=int(fetch_summary.get("results") or 0) if should_fetch else 0,
            inserted=int(upsert.get("inserted") or 0),
            updated=int(upsert.get("updated") or 0),
            pruned=int(pruned),
            pruned_articles=int(pruned_articles),
            notes=notes,
        )

    out = {
        "ok": True,
        "source": "gdelt",
        "db": str(db_path),
        "window_hours": hours,
        "pruned": int(pruned),
        "pruned_articles": int(pruned_articles),
        "upsert": upsert,
        "fetch": fetch_summary,
    }
    print(NewsPoolDB.dumps_compact(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
