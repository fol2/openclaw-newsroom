#!/usr/bin/env python3
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

from newsroom.brave_news import (  # noqa: E402
    BraveApiKey,
    fetch_brave_news,
    load_brave_api_keys,
    normalize_url,
    record_brave_rate_limit,
    select_brave_api_key,
)
from newsroom.news_pool_db import NewsPoolDB, PoolLink  # noqa: E402


DEFAULT_QUERIES = [
    # Broad pool (global mix): HK + politics + AI/tech + entertainment + sports + finance.
    # NOTE: Brave enforces a 50-word limit (whitespace-tokenised) on `q`. Keep this compact.
    "HK OR 香港 OR politics OR election OR 政治 OR sports OR entertainment OR 娛樂 OR film OR AI OR 人工智能 OR technology OR stocks OR shares OR earnings OR futures OR Fed OR CPI OR Treasury OR yields OR Nasdaq OR NYSE",
    # UK (broad UK outlets): keep it UK-first by restricting to UK-ish domains without being too narrow.
    "UK (site:co.uk OR site:theguardian.com OR site:news.sky.com OR site:itv.com OR site:channel4.com)",
    # UK politics (PMQs / Westminster): broader UK outlets (+ LBC) to avoid missing coverage.
    "(PMQs OR Westminster OR Parliament) (site:co.uk OR site:theguardian.com OR site:news.sky.com OR site:itv.com OR site:channel4.com OR site:lbc.co.uk)",
    # Finance focus (kept separate so finance appears more often in the hourly rotation).
    "(stocks OR shares OR earnings OR futures OR Fed OR CPI OR Treasury OR yields) (Nasdaq OR NYSE OR \"S&P 500\" OR Dow OR \"Wall Street\")",
    # Bias towards HK entertainment (explicit user request).
    "Hong Kong entertainment celebrity concert film TV",
    # AI-specific (policy/funding/safety).
    "AI model release regulation safety funding",
    # Sports-specific.
    "sports match result tournament ban suspension",
]


def _domain(url: str) -> str | None:
    try:
        return urlsplit(url).hostname
    except Exception:
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Update the 48h SQLite news pool from Brave News API (deduped by normalized URL).")
    parser.add_argument("--db", default=str(OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"), help="SQLite db path.")
    parser.add_argument("--hours", type=int, default=48, help="Retention window in hours (default: 48).")
    parser.add_argument("--state-key", default="default", help="Rotation key for fetch_state table.")
    parser.add_argument("--min-interval-seconds", type=int, default=900, help="Minimum seconds between API fetches for this state key.")
    parser.add_argument("--query", action="append", default=[], help="Query string (repeatable). If omitted, uses defaults.")
    parser.add_argument("--freshness", default="day", help="Brave freshness (day|week|...).")
    parser.add_argument("--count", type=int, default=10, help="Results per request (default: 10).")
    parser.add_argument("--pages", type=int, default=2, help="How many pages (offsets) to fetch per run starting at the rotated offset (default: 2).")
    parser.add_argument("--max-offset", type=int, default=2, help="Rotate start offsets 0..max_offset-1 (default: 2).")
    parser.add_argument("--cache-ttl-seconds", type=int, default=900, help="File cache TTL seconds (default: 900).")
    parser.add_argument(
        "--key-label",
        default=None,
        help="Prefer a specific Brave key label from brave_search_api_keys*.txt (e.g. free, paid).",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser()
    hours = int(max(1, args.hours))
    cutoff_ts = int(time.time() - hours * 3600)

    queries = [q.strip() for q in (args.query or []) if isinstance(q, str) and q.strip()]
    if not queries:
        queries = list(DEFAULT_QUERIES)

    max_offset = int(max(1, min(10, args.max_offset)))
    # Brave supports larger `count` values; in practice it may still return fewer than requested.
    count = int(max(1, min(100, args.count)))
    pages = int(max(1, min(3, args.pages)))
    freshness = str(args.freshness).strip() or None

    cache_dir = OPENCLAW_HOME / "data" / "newsroom" / "brave_news_cache"

    now_ts = int(time.time())
    with NewsPoolDB(path=db_path) as db:
        pruned = db.prune_links(cutoff_ts=cutoff_ts)
        pruned_articles = db.prune_article_cache(cutoff_ts=cutoff_ts)
        pruned_semantic = db.prune_semantic_keys(cutoff_ts=cutoff_ts)

        state_key = str(args.state_key).strip() or "default"
        state = db.fetch_state(state_key)
        last_fetch_ts = int(state.get("last_fetch_ts", 0))
        run_count = int(state.get("run_count", 0))

        should_fetch = (now_ts - last_fetch_ts) >= int(max(0, args.min_interval_seconds))

        fetch_summary: dict[str, Any] = {"should_fetch": should_fetch}
        upsert = {"inserted": 0, "updated": 0}
        notes: str | None = None
        had_success = False

        if should_fetch:
            # Rotate query and offset deterministically across runs.
            next_run = run_count + 1
            # Use (next_run-1) so the very first run starts at queries[0], offset=0.
            q_idx = (next_run - 1) % len(queries)
            offset = (((next_run - 1) // len(queries)) % max_offset)
            q = queries[q_idx]

            api_keys: list[BraveApiKey] = load_brave_api_keys(openclaw_home=OPENCLAW_HOME)
            per_page: list[dict[str, Any]] = []
            requests_made = 0
            total_results = 0
            links: list[PoolLink] = []
            last_ts = 0.0
            last_rate_limit: dict[str, Any] | None = None
            keys_used: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []

            offsets = [o for o in range(offset, min(10, offset + pages))]
            for off in offsets:
                try:
                    key = select_brave_api_key(
                        openclaw_home=OPENCLAW_HOME,
                        keys=api_keys,
                        prefer_label=(str(args.key_label).strip() or None),
                        now_ts=now_ts,
                    )
                    fetched, last_ts = fetch_brave_news(
                        api_key=key.key,
                        q=q,
                        count=count,
                        offset=int(off),
                        freshness=freshness,
                        cache_dir=cache_dir,
                        ttl_seconds=int(args.cache_ttl_seconds),
                        last_request_ts=last_ts,
                    )
                    had_success = True
                except Exception as e:
                    errors.append({"offset": int(off), "error": str(e)})
                    continue

                keys_used.append({"key_id": key.key_id, "label": key.label})
                if isinstance(getattr(fetched, "rate_limit", None), dict):
                    last_rate_limit = fetched.rate_limit
                    record_brave_rate_limit(openclaw_home=OPENCLAW_HOME, key=key, rate_limit=fetched.rate_limit, now_ts=now_ts)
                requests_made += int(fetched.requests_made)
                total_results += len(fetched.results)
                per_page.append(
                    {
                        "offset": int(off),
                        "cached": bool(fetched.cached),
                        "requests_made": int(fetched.requests_made),
                        "results": len(fetched.results),
                        "fetched_at": fetched.fetched_at,
                    }
                )

                for r in fetched.results:
                    url = r.get("url")
                    if not isinstance(url, str) or not url.strip():
                        continue
                    url = normalize_url(url)
                    if not url:
                        continue
                    norm = url  # already normalized
                    domain = r.get("domain")
                    if not isinstance(domain, str) or not domain.strip():
                        domain = _domain(url)
                    links.append(
                        PoolLink(
                            url=url,
                            norm_url=norm,
                            domain=str(domain).lower() if isinstance(domain, str) and domain else None,
                            title=r.get("title") if isinstance(r.get("title"), str) else None,
                            description=r.get("description") if isinstance(r.get("description"), str) else None,
                            age=r.get("age") if isinstance(r.get("age"), str) else None,
                            page_age=r.get("page_age") if isinstance(r.get("page_age"), str) else None,
                            query=q,
                            offset=int(off),
                            fetched_at_ts=now_ts,
                        )
                    )

            upsert = db.upsert_links(links, now_ts=now_ts)
            if had_success:
                db.update_fetch_state(key=state_key, last_fetch_ts=now_ts, last_offset=int(offset), run_count=next_run)

            fetch_summary = {
                "should_fetch": True,
                "query": q,
                "offset_start": int(offset),
                "pages": int(pages),
                "per_page": per_page,
                "requests_made": int(requests_made),
                "results": int(total_results),
                "run_count": int(next_run),
            }
            notes_payload: dict[str, Any] = {"keys_used": keys_used}
            if last_rate_limit:
                notes_payload["rate_limit"] = last_rate_limit
            if errors:
                fetch_summary["errors"] = errors
                notes_payload["errors"] = errors
            notes = json.dumps(notes_payload, ensure_ascii=True, separators=(",", ":"))
        else:
            fetch_summary = {
                "should_fetch": False,
                "reason": "min_interval",
                "last_fetch_ts": int(last_fetch_ts),
                "run_count": int(run_count),
            }
            notes = json.dumps({"reason": "min_interval"}, ensure_ascii=True, separators=(",", ":"))

        # Persist metrics for later status/usage reporting.
        db.log_pool_run(
            run_ts=now_ts,
            state_key=state_key,
            window_hours=hours,
            should_fetch=bool(should_fetch),
            query=(str(fetch_summary.get("query")) if should_fetch and isinstance(fetch_summary.get("query"), str) else None),
            offset_start=(int(fetch_summary.get("offset_start")) if should_fetch and fetch_summary.get("offset_start") is not None else None),
            pages=(int(fetch_summary.get("pages")) if should_fetch and fetch_summary.get("pages") is not None else int(args.pages)),
            count=int(count),
            freshness=freshness,
            requests_made=int(fetch_summary.get("requests_made") or 0) if should_fetch else 0,
            results=int(fetch_summary.get("results") or 0) if should_fetch else 0,
            inserted=int(upsert.get("inserted") or 0),
            updated=int(upsert.get("updated") or 0),
            pruned=int(pruned),
            pruned_articles=int(pruned_articles),
            notes=notes,
        )

    out = {
        "ok": bool((not should_fetch) or had_success),
        "db": str(db_path),
        "window_hours": hours,
        "pruned": int(pruned),
        "pruned_articles": int(pruned_articles),
        "pruned_semantic": int(pruned_semantic),
        "upsert": upsert,
        "fetch": fetch_summary,
    }
    # Compact output for planner consumption.
    print(NewsPoolDB.dumps_compact(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
