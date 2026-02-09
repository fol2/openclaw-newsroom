from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from .brave_news import normalize_url

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass(frozen=True)
class GdeltFetch:
    ok: bool
    cached: bool
    requests_made: int
    query: str
    maxrecords: int
    timespan: str
    fetched_at: str
    results: list[dict[str, Any]]
    error: str | None = None


def parse_gdelt_seendate(s: str | None) -> str | None:
    """Parse GDELT seendate format to ISO 8601.

    '20260208T143000Z' → '2026-02-08T14:30:00+00:00'
    """
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    try:
        dt = datetime.strptime(s, "%Y%m%dT%H%M%SZ")
        dt = dt.replace(tzinfo=UTC)
        return dt.isoformat(timespec="seconds")
    except (ValueError, OverflowError):
        return None


def _cache_key(*, query: str, maxrecords: int, timespan: str, sourcelang: str | None) -> str:
    payload = json.dumps(
        {"query": query, "maxrecords": maxrecords, "timespan": timespan, "sourcelang": sourcelang},
        sort_keys=True,
    )
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


def _domain_from_url(url: str) -> str | None:
    try:
        return urlsplit(url).hostname
    except Exception:
        return None


def fetch_gdelt_articles(
    *,
    query: str,
    maxrecords: int = 75,
    timespan: str = "24h",
    sourcelang: str | None = "english",
    cache_dir: Path | None = None,
    ttl_seconds: int = 900,
    last_request_ts: float = 0.0,
) -> tuple[GdeltFetch, float]:
    """Fetch articles from GDELT DOC 2.0 API.

    Returns (GdeltFetch, last_request_ts).
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")

    maxrecords = int(max(1, min(250, maxrecords)))

    # Build the actual query string for GDELT.
    gdelt_query = query
    if sourcelang:
        gdelt_query = f"{query} sourcelang:{sourcelang}"

    # Check cache.
    cache_path: Path | None = None
    if cache_dir is not None:
        key = _cache_key(query=query, maxrecords=maxrecords, timespan=timespan, sourcelang=sourcelang)
        cache_path = cache_dir / f"{key}.json"
        hit = _read_cache(cache_path, ttl_seconds=int(ttl_seconds))
        if hit is not None and isinstance(hit.get("results"), list):
            fetched_at = str(hit.get("fetched_at") or datetime.now(tz=UTC).isoformat(timespec="seconds"))
            return (
                GdeltFetch(
                    ok=True,
                    cached=True,
                    requests_made=0,
                    query=query,
                    maxrecords=maxrecords,
                    timespan=timespan,
                    fetched_at=fetched_at,
                    results=list(hit["results"]),
                ),
                last_request_ts,
            )

    # Politeness delay (1.05s between requests, same as Brave).
    now = time.time()
    delta = now - last_request_ts
    if delta < 1.05:
        time.sleep(1.05 - delta)
    last_request_ts = time.time()

    params: dict[str, Any] = {
        "query": gdelt_query,
        "mode": "ArtList",
        "maxrecords": maxrecords,
        "format": "json",
        "timespan": timespan,
    }

    try:
        resp = requests.get(GDELT_DOC_URL, params=params, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        return (
            GdeltFetch(
                ok=False,
                cached=False,
                requests_made=1,
                query=query,
                maxrecords=maxrecords,
                timespan=timespan,
                fetched_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
                results=[],
                error=str(e),
            ),
            last_request_ts,
        )

    try:
        raw = resp.json()
    except Exception:
        return (
            GdeltFetch(
                ok=False,
                cached=False,
                requests_made=1,
                query=query,
                maxrecords=maxrecords,
                timespan=timespan,
                fetched_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
                results=[],
                error="invalid_json_response",
            ),
            last_request_ts,
        )

    articles = raw.get("articles") if isinstance(raw, dict) else None
    if not isinstance(articles, list):
        articles = []

    out_items: list[dict[str, Any]] = []
    for art in articles:
        if not isinstance(art, dict):
            continue
        url = art.get("url")
        title = art.get("title")
        if not (isinstance(url, str) and url.strip() and isinstance(title, str) and title.strip()):
            continue

        norm = normalize_url(url)
        if not norm:
            continue

        domain = art.get("domain")
        if not isinstance(domain, str) or not domain.strip():
            domain = _domain_from_url(url)

        seendate = art.get("seendate")
        page_age = parse_gdelt_seendate(seendate) if isinstance(seendate, str) else None

        out_items.append(
            {
                "title": " ".join(title.split())[:200],
                "url": norm,
                "domain": domain.lower() if isinstance(domain, str) and domain else None,
                "page_age": page_age,
                "language": art.get("language"),
                "sourcecountry": art.get("sourcecountry"),
            }
        )

    fetched_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    payload = {
        "ok": True,
        "endpoint": GDELT_DOC_URL,
        "query": query,
        "maxrecords": maxrecords,
        "timespan": timespan,
        "fetched_at": fetched_at,
        "results": out_items,
    }
    if cache_path is not None:
        _write_cache(cache_path, payload)

    return (
        GdeltFetch(
            ok=True,
            cached=False,
            requests_made=1,
            query=query,
            maxrecords=maxrecords,
            timespan=timespan,
            fetched_at=fetched_at,
            results=out_items,
        ),
        last_request_ts,
    )
