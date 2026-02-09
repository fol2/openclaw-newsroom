#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qsl

import requests
import sys


OPENCLAW_HOME = Path(__file__).resolve().parents[1]


BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"


def _load_api_key(*, key_path: Path | None) -> str:
    env = os.environ.get("BRAVE_SEARCH_API_KEY")
    if env and env.strip():
        return env.strip()
    if key_path is None:
        key_path = OPENCLAW_HOME / "secrets" / "brave_search_api_key.txt"
    if not key_path.exists():
        raise SystemExit(f"Missing Brave API key. Set BRAVE_SEARCH_API_KEY or create {key_path}.")
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(f"Brave API key file is empty: {key_path}")
    return key


def _normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except Exception:
        return raw

    # Drop fragment.
    parsed = parsed._replace(fragment="")

    # Strip common tracking query params.
    if parsed.query:
        q = parse_qsl(parsed.query, keep_blank_values=True)
        filtered: list[tuple[str, str]] = []
        for k, v in q:
            lk = k.lower()
            if lk.startswith("utm_"):
                continue
            if lk in {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid"}:
                continue
            filtered.append((k, v))
        parsed = parsed._replace(query=urlencode(filtered, doseq=True))

    return urlunsplit(parsed)


def _cache_key(*, q: str, count: int, offset: int, freshness: str | None) -> str:
    payload = json.dumps({"q": q, "count": count, "offset": offset, "freshness": freshness}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def _read_cache(path: Path, *, ttl_seconds: int) -> dict[str, Any] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    if ttl_seconds >= 0 and (time.time() - st.st_mtime) > ttl_seconds:
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class QueryResult:
    query: str
    cached: bool
    count: int
    results: list[dict[str, Any]]


def _fetch_one(
    *,
    api_key: str,
    q: str,
    count: int,
    offset: int,
    freshness: str | None,
    cache_path: Path | None,
    ttl_seconds: int,
    last_request_ts: float,
) -> tuple[QueryResult, float]:
    if cache_path is not None:
        cached = _read_cache(cache_path, ttl_seconds=ttl_seconds)
        if cached is not None:
            results = cached.get("results")
            if isinstance(results, list):
                return QueryResult(query=q, cached=True, count=len(results), results=results), last_request_ts

    # Brave Free tier: 1 request/second. Enforce a small buffer.
    now = time.time()
    delta = now - last_request_ts
    if delta < 1.05:
        time.sleep(1.05 - delta)
    last_request_ts = time.time()

    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params: dict[str, Any] = {"q": q, "count": int(count), "offset": int(offset)}
    if freshness:
        params["freshness"] = freshness

    # Retry briefly on 429/503.
    for attempt in range(1, 4):
        resp = requests.get(BRAVE_NEWS_ENDPOINT, headers=headers, params=params, timeout=20)
        if resp.status_code in (429, 503):
            time.sleep(min(2.0 * attempt, 5.0))
            continue
        break

    if resp.status_code != 200:
        try:
            err = resp.json().get("error")
        except Exception:
            err = {"status": resp.status_code, "detail": resp.text[:200]}
        raise SystemExit(f"Brave API error (status {resp.status_code}): {err}")

    raw = resp.json()
    items = raw.get("results", [])
    if not isinstance(items, list):
        items = []

    # Compact rows for LLM consumption.
    out_items: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        url = it.get("url")
        title = it.get("title")
        if not (isinstance(url, str) and url.strip() and isinstance(title, str) and title.strip()):
            continue
        desc = it.get("description")
        if isinstance(desc, str):
            desc = " ".join(desc.split())
            if len(desc) > 240:
                desc = desc[:237] + "..."
        else:
            desc = None

        meta_url = it.get("meta_url") or {}
        domain = meta_url.get("hostname") if isinstance(meta_url, dict) else None
        if not isinstance(domain, str):
            try:
                domain = urlsplit(url).hostname
            except Exception:
                domain = None

        out_items.append(
            {
                "title": " ".join(title.split())[:180],
                "url": _normalize_url(url),
                "domain": domain,
                "description": desc,
                "age": it.get("age"),
                "page_age": it.get("page_age"),
            }
        )

    payload = {
        "ok": True,
        "endpoint": BRAVE_NEWS_ENDPOINT,
        "query": q,
        "count": int(count),
        "offset": int(offset),
        "freshness": freshness,
        "fetched_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "results": out_items,
    }
    if cache_path is not None:
        _write_cache(cache_path, payload)

    return QueryResult(query=q, cached=False, count=len(out_items), results=out_items), last_request_ts


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Fetch a pooled set of news candidates from Brave Search API (News endpoint).")
    parser.add_argument("--query", action="append", default=[], help="Search query (repeatable).")
    parser.add_argument("--count", type=int, default=50, help="Results per query (max 50).")
    parser.add_argument("--offset", type=int, default=1, help="Start page offset (1..9).")
    parser.add_argument("--pages", type=int, default=1, help="How many pages to fetch starting at --offset (default 1).")
    parser.add_argument("--freshness", default="day", help="Freshness window (e.g. day, week).")
    parser.add_argument("--max-queries", type=int, default=2, help="Safety cap for repeatable queries.")
    parser.add_argument("--ttl-seconds", type=int, default=900, help="Cache TTL seconds (default 900).")
    parser.add_argument("--cache-dir", default=None, help="Cache dir (default: data/newsroom/brave_news_cache).")
    parser.add_argument("--key-path", default=None, help="API key file path (default: secrets/brave_search_api_key.txt).")
    args = parser.parse_args(argv)

    queries = [q.strip() for q in (args.query or []) if isinstance(q, str) and q.strip()]
    if not queries:
        queries = ["breaking news"]

    max_q = int(max(1, args.max_queries))
    if len(queries) > max_q:
        queries = queries[:max_q]

    count = int(max(1, min(50, args.count)))
    offset = int(max(1, min(9, args.offset)))
    pages = int(max(1, min(3, args.pages)))
    offsets = [o for o in range(offset, min(10, offset + pages))]
    freshness = str(args.freshness).strip() or None

    key_path = Path(args.key_path) if args.key_path else None
    api_key = _load_api_key(key_path=key_path)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (OPENCLAW_HOME / "data" / "newsroom" / "brave_news_cache")

    last_ts = 0.0
    per_query: list[dict[str, Any]] = []
    pooled: list[dict[str, Any]] = []
    seen: set[str] = set()
    requests_made = 0

    for q in queries:
        for off in offsets:
            cache_path = cache_dir / f"{_cache_key(q=q, count=count, offset=off, freshness=freshness)}.json"
            qr, last_ts = _fetch_one(
                api_key=api_key,
                q=q,
                count=count,
                offset=off,
                freshness=freshness,
                cache_path=cache_path,
                ttl_seconds=int(args.ttl_seconds),
                last_request_ts=last_ts,
            )
            if not qr.cached:
                requests_made += 1
            per_query.append(
                {"query": qr.query, "offset": off, "cached": qr.cached, "count": qr.count, "cache_path": str(cache_path)}
            )
            for r in qr.results:
                u = r.get("url")
                if not isinstance(u, str) or not u.strip():
                    continue
                nu = _normalize_url(u.strip())
                if nu in seen:
                    continue
                seen.add(nu)
                pooled.append(r)

    out = {
        "ok": True,
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "requests_made": requests_made,
        "queries": per_query,
        "results": pooled,
        "count": len(pooled),
    }
    # Keep output compact to reduce planner token usage.
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
