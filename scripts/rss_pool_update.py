#!/usr/bin/env python3
"""Update the news pool from curated RSS/Atom feeds.

Same pattern as news_pool_update.py: argparse → fetch_state check →
fetch all feeds → PoolLink → upsert → log_pool_run → print JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.brave_news import normalize_url  # noqa: E402
from newsroom.news_pool_db import NewsPoolDB, PoolLink  # noqa: E402
from newsroom.rss_news import RssFeed, fetch_rss_feed, load_feeds  # noqa: E402

_DEFAULT_FEEDS_CONFIG = OPENCLAW_HOME / "newsroom" / "rss_feeds.yaml"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Update the news pool from RSS/Atom feeds.")
    parser.add_argument("--db", default=str(OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"), help="SQLite db path.")
    parser.add_argument("--hours", type=int, default=48, help="Retention window in hours (default: 48).")
    parser.add_argument("--state-key", default="rss", help="Rotation key for fetch_state table.")
    parser.add_argument("--min-interval-seconds", type=int, default=1800, help="Minimum seconds between fetches (default: 1800).")
    parser.add_argument("--feeds-config", default=str(_DEFAULT_FEEDS_CONFIG), help="Path to rss_feeds.yaml.")
    parser.add_argument("--region", default=None, help="Optional filter: 'uk', 'hk', 'global'.")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser()
    hours = int(max(1, args.hours))
    cutoff_ts = int(time.time() - hours * 3600)

    feeds = load_feeds(config_path=Path(args.feeds_config).expanduser())
    if args.region:
        region = str(args.region).strip().lower()
        feeds = [f for f in feeds if f.region == region]

    now_ts = int(time.time())
    with NewsPoolDB(path=db_path) as db:
        pruned = db.prune_links(cutoff_ts=cutoff_ts)
        pruned_articles = db.prune_article_cache(cutoff_ts=cutoff_ts)

        state_key = str(args.state_key).strip() or "rss"
        state = db.fetch_state(state_key)
        last_fetch_ts = int(state.get("last_fetch_ts", 0))
        run_count = int(state.get("run_count", 0))

        should_fetch = (now_ts - last_fetch_ts) >= int(max(0, args.min_interval_seconds))

        fetch_summary: dict[str, Any] = {"should_fetch": should_fetch}
        upsert = {"inserted": 0, "updated": 0}
        notes: str | None = None

        if should_fetch:
            next_run = run_count + 1

            feeds_ok = 0
            feeds_failed = 0
            errors: list[dict[str, str]] = []
            total_requests = 0
            total_results = 0
            links: list[PoolLink] = []
            last_ts = 0.0

            for feed in feeds:
                try:
                    result, last_ts = fetch_rss_feed(feed, last_request_ts=last_ts)
                    total_requests += int(result.requests_made)

                    if result.ok:
                        feeds_ok += 1
                        for art in result.articles:
                            url = art.url
                            if not url:
                                continue
                            norm = normalize_url(url)
                            if not norm:
                                continue
                            total_results += 1
                            links.append(
                                PoolLink(
                                    url=url,
                                    norm_url=norm,
                                    domain=art.domain,
                                    title=art.title,
                                    description=art.description,
                                    age=None,
                                    page_age=art.published,
                                    query=f"rss:{feed.key}",
                                    offset=0,
                                    fetched_at_ts=now_ts,
                                )
                            )
                    else:
                        feeds_failed += 1
                        errors.append({"feed": feed.key, "error": result.error or "unknown"})
                except Exception as e:
                    feeds_failed += 1
                    errors.append({"feed": feed.key, "error": str(e)})

            upsert = db.upsert_links(links, now_ts=now_ts)
            db.update_fetch_state(key=state_key, last_fetch_ts=now_ts, last_offset=0, run_count=next_run)

            fetch_summary = {
                "should_fetch": True,
                "feeds_total": len(feeds),
                "feeds_ok": feeds_ok,
                "feeds_failed": feeds_failed,
                "requests_made": total_requests,
                "results": total_results,
                "run_count": int(next_run),
            }
            if errors:
                fetch_summary["errors"] = errors

            notes = json.dumps(
                {"source": "rss", "feeds_ok": feeds_ok, "feeds_failed": feeds_failed},
                ensure_ascii=True, separators=(",", ":"),
            )
        else:
            fetch_summary = {
                "should_fetch": False,
                "reason": "min_interval",
                "last_fetch_ts": int(last_fetch_ts),
                "run_count": int(run_count),
            }
            notes = json.dumps({"source": "rss", "reason": "min_interval"}, ensure_ascii=True, separators=(",", ":"))

        db.log_pool_run(
            run_ts=now_ts,
            state_key=state_key,
            window_hours=hours,
            should_fetch=bool(should_fetch),
            query="rss:all" if should_fetch else None,
            offset_start=0,
            pages=1,
            count=len(feeds),
            freshness=None,
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
        "source": "rss",
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
