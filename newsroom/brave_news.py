from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests


BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"

_KEY_STATE_VERSION = 1


class BraveApiError(RuntimeError):
    """Brave News API error with optional rate limit context."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None,
        error: Any | None,
        rate_limit: dict[str, Any] | None,
        retry_after_s: int | None,
        requests_made: int,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code) if isinstance(status_code, int) else None
        self.error = error
        self.rate_limit = rate_limit
        self.retry_after_s = int(retry_after_s) if isinstance(retry_after_s, int) else None
        self.requests_made = int(requests_made)


@dataclass(frozen=True)
class BraveApiKey:
    """A Brave Search API key.

    `label` is optional and can be used to express intent (e.g. "free", "paid").
    The key itself is never written to logs/state.
    """

    key: str
    label: str | None = None

    @property
    def key_id(self) -> str:
        return hashlib.sha256(self.key.encode("utf-8")).hexdigest()[:12]


def _key_state_path(*, openclaw_home: Path) -> Path:
    return openclaw_home / "data" / "newsroom" / "brave_key_state.json"


def _load_key_state(*, path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": _KEY_STATE_VERSION, "keys": {}}
    except Exception:
        return {"version": _KEY_STATE_VERSION, "keys": {}}

    if not isinstance(obj, dict):
        return {"version": _KEY_STATE_VERSION, "keys": {}}
    if obj.get("version") != _KEY_STATE_VERSION:
        return {"version": _KEY_STATE_VERSION, "keys": {}}
    keys = obj.get("keys")
    if not isinstance(keys, dict):
        obj["keys"] = {}
    return obj


def _save_key_state(*, path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_key_spec(spec: str) -> BraveApiKey | None:
    s = (spec or "").strip()
    if not s or s.startswith("#"):
        return None
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    if not s:
        return None

    label: str | None = None
    key: str | None = None

    # Format: "label:key"
    if ":" in s and not any(ch.isspace() for ch in s):
        left, right = s.split(":", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            label = left
            key = right

    # Format: "label key"
    if key is None and any(ch.isspace() for ch in s):
        parts = [p for p in s.split() if p]
        if len(parts) >= 2:
            label = parts[0]
            key = parts[-1]

    if key is None:
        key = s

    key = key.strip()
    if not key:
        return None
    return BraveApiKey(key=key, label=(label.strip() if isinstance(label, str) and label.strip() else None))


def load_brave_api_keys(*, openclaw_home: Path, keys_path: Path | None = None) -> list[BraveApiKey]:
    """Load 1+ Brave API keys.

    Priority order:
    1) BRAVE_SEARCH_API_KEYS (comma/newline separated; supports label:key items)
    2) BRAVE_SEARCH_API_KEY (single key)
    3) secrets/brave_search_api_keys.local.txt (multi-line; supports label:key)
    4) secrets/brave_search_api_keys.txt (multi-line; supports label:key)
    5) legacy single-key file: secrets/brave_search_api_key.local.txt or secrets/brave_search_api_key.txt
    """

    multi_env = os.environ.get("BRAVE_SEARCH_API_KEYS")
    if isinstance(multi_env, str) and multi_env.strip():
        out: list[BraveApiKey] = []
        raw = multi_env.replace("\n", ",")
        for spec in [p.strip() for p in raw.split(",") if p.strip()]:
            k = _parse_key_spec(spec)
            if k:
                out.append(k)
        if out:
            return out

    env = os.environ.get("BRAVE_SEARCH_API_KEY")
    if isinstance(env, str) and env.strip():
        return [BraveApiKey(key=env.strip(), label="env")]

    if keys_path is None:
        local = openclaw_home / "secrets" / "brave_search_api_keys.local.txt"
        keys_path = local if local.exists() else (openclaw_home / "secrets" / "brave_search_api_keys.txt")

    if keys_path.exists():
        lines = keys_path.read_text(encoding="utf-8").splitlines()
        out: list[BraveApiKey] = []
        for line in lines:
            k = _parse_key_spec(line)
            if k:
                out.append(k)
        if out:
            return out

    # Legacy single key.
    return [BraveApiKey(key=load_brave_api_key(openclaw_home=openclaw_home), label="legacy")]


def select_brave_api_key(
    *,
    openclaw_home: Path,
    keys: list[BraveApiKey],
    prefer_label: str | None = None,
    now_ts: int | None = None,
) -> BraveApiKey:
    if not keys:
        raise RuntimeError("No Brave API keys loaded")

    now_ts = int(time.time()) if now_ts is None else int(now_ts)
    state_path = _key_state_path(openclaw_home=openclaw_home)
    state = _load_key_state(path=state_path)
    ks = state.get("keys") if isinstance(state, dict) else {}
    if not isinstance(ks, dict):
        ks = {}

    def is_exhausted(k: BraveApiKey) -> bool:
        row = ks.get(k.key_id)
        if not isinstance(row, dict):
            return False
        until = row.get("exhausted_until_ts")
        try:
            return until is not None and int(until) > now_ts
        except Exception:
            return False

    def is_cooled_down(k: BraveApiKey) -> bool:
        row = ks.get(k.key_id)
        if not isinstance(row, dict):
            return False
        until = row.get("cooldown_until_ts")
        try:
            return until is not None and int(until) > now_ts
        except Exception:
            return False

    available = [k for k in keys if (not is_exhausted(k)) and (not is_cooled_down(k))]

    if prefer_label:
        for k in available:
            if k.label == prefer_label:
                return k
    if available:
        return available[0]
    # Fall back to the first key even if we think it's exhausted; this lets callers
    # decide how to degrade when all keys are out of quota.
    return keys[0]


def _quota_from_rate_limit(rate_limit: dict[str, Any] | None) -> dict[str, int | None]:
    """Extract per-second + monthly quota from Brave rate limit headers.

    Brave returns comma-separated values; index 0 is typically per-second, index 1 monthly.
    """

    if not isinstance(rate_limit, dict):
        return {
            "per_second_limit": None,
            "per_second_remaining": None,
            "per_second_reset_s": None,
            "monthly_limit": None,
            "monthly_remaining": None,
            "monthly_reset_s": None,
        }

    def pick(v: Any, i: int) -> int | None:
        if isinstance(v, list) and len(v) > i and isinstance(v[i], int):
            return int(v[i])
        return None

    lim = rate_limit.get("limit")
    rem = rate_limit.get("remaining")
    rst = rate_limit.get("reset")

    return {
        "per_second_limit": pick(lim, 0),
        "per_second_remaining": pick(rem, 0),
        "per_second_reset_s": pick(rst, 0),
        "monthly_limit": pick(lim, 1),
        "monthly_remaining": pick(rem, 1),
        "monthly_reset_s": pick(rst, 1),
    }


def record_brave_rate_limit(
    *,
    openclaw_home: Path,
    key: BraveApiKey,
    rate_limit: dict[str, Any] | None,
    now_ts: int | None = None,
) -> None:
    """Persist the last-known quota state for a Brave API key (by key_id)."""

    if not isinstance(rate_limit, dict):
        return

    now_ts = int(time.time()) if now_ts is None else int(now_ts)
    q = _quota_from_rate_limit(rate_limit)
    monthly_limit = q.get("monthly_limit")
    monthly_remaining = q.get("monthly_remaining")
    monthly_reset_s = q.get("monthly_reset_s")
    reset_at_ts = (now_ts + int(monthly_reset_s)) if isinstance(monthly_reset_s, int) and monthly_reset_s > 0 else None

    state_path = _key_state_path(openclaw_home=openclaw_home)
    state = _load_key_state(path=state_path)
    keys_obj = state.get("keys")
    if not isinstance(keys_obj, dict):
        keys_obj = {}
        state["keys"] = keys_obj

    row = keys_obj.get(key.key_id)
    if not isinstance(row, dict):
        row = {}
        keys_obj[key.key_id] = row

    row["label"] = key.label
    row["last_seen_ts"] = int(now_ts)
    if isinstance(monthly_limit, int):
        row["monthly_limit"] = monthly_limit
    if isinstance(monthly_remaining, int):
        row["monthly_remaining"] = monthly_remaining
    if isinstance(monthly_reset_s, int):
        row["monthly_reset_s"] = monthly_reset_s
    if reset_at_ts:
        row["monthly_reset_at_ts"] = int(reset_at_ts)

    # When remaining hits 0, mark the key as exhausted until reset (best-effort).
    if isinstance(monthly_remaining, int):
        if monthly_remaining <= 0 and reset_at_ts:
            row["exhausted_until_ts"] = int(reset_at_ts)
        elif monthly_remaining > 0:
            row["exhausted_until_ts"] = None

    _save_key_state(path=state_path, state=state)


def record_brave_cooldown(
    *,
    openclaw_home: Path,
    key: BraveApiKey,
    cooldown_seconds: int | None = None,
    cooldown_until_ts: int | None = None,
    now_ts: int | None = None,
    reason: str | None = None,
) -> None:
    """Persist a short-term cooldown for a Brave API key (e.g. after 429/503)."""

    now_ts = int(time.time()) if now_ts is None else int(now_ts)
    until_ts: int | None
    if cooldown_until_ts is not None:
        try:
            until_ts = int(cooldown_until_ts)
        except Exception:
            until_ts = None
    elif cooldown_seconds is not None:
        try:
            sec = int(cooldown_seconds)
        except Exception:
            sec = 0
        until_ts = int(now_ts + max(0, sec))
    else:
        until_ts = None

    if until_ts is None:
        return

    state_path = _key_state_path(openclaw_home=openclaw_home)
    state = _load_key_state(path=state_path)
    keys_obj = state.get("keys")
    if not isinstance(keys_obj, dict):
        keys_obj = {}
        state["keys"] = keys_obj

    row = keys_obj.get(key.key_id)
    if not isinstance(row, dict):
        row = {}
        keys_obj[key.key_id] = row

    row["label"] = key.label
    row["last_seen_ts"] = int(now_ts)

    prev = row.get("cooldown_until_ts")
    try:
        prev_ts = int(prev) if prev is not None else None
    except Exception:
        prev_ts = None
    row["cooldown_until_ts"] = int(max(prev_ts or 0, until_ts))
    if isinstance(reason, str) and reason.strip():
        row["cooldown_reason"] = reason.strip()

    _save_key_state(path=state_path, state=state)


def load_brave_api_key(*, openclaw_home: Path, key_path: Path | None = None) -> str:
    env = os.environ.get("BRAVE_SEARCH_API_KEY")
    if env and env.strip():
        return env.strip()
    if key_path is None:
        # Prefer an untracked local key file to avoid committing secrets.
        local = openclaw_home / "secrets" / "brave_search_api_key.local.txt"
        key_path = local if local.exists() else (openclaw_home / "secrets" / "brave_search_api_key.txt")
    if not key_path.exists():
        raise RuntimeError(f"Missing Brave API key. Set BRAVE_SEARCH_API_KEY or create {key_path}.")
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"Brave API key file is empty: {key_path}")
    return key


def normalize_url(url: str) -> str:
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

    # Normalize scheme+host case and trim trailing slash.
    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    path = parsed.path or ""
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    parsed = parsed._replace(scheme=scheme, netloc=netloc, path=path)

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
class BraveFetch:
    ok: bool
    cached: bool
    requests_made: int
    query: str
    count: int
    offset: int
    freshness: str | None
    fetched_at: str
    results: list[dict[str, Any]]
    rate_limit: dict[str, Any] | None


def _parse_rate_limit_header(value: str | None) -> list[int] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    out: list[int] = []
    for p in value.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out or None


def _rate_limit_info(resp: requests.Response) -> dict[str, Any]:
    # Brave documents rate limit/quota info in:
    # X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset.
    # Values may be comma-separated per-window (e.g. per-second, monthly).
    limit_raw = resp.headers.get("X-RateLimit-Limit")
    remaining_raw = resp.headers.get("X-RateLimit-Remaining")
    reset_raw = resp.headers.get("X-RateLimit-Reset")
    return {
        "limit_raw": limit_raw,
        "remaining_raw": remaining_raw,
        "reset_raw": reset_raw,
        "limit": _parse_rate_limit_header(limit_raw),
        "remaining": _parse_rate_limit_header(remaining_raw),
        "reset": _parse_rate_limit_header(reset_raw),
    }


def _parse_retry_after(value: str | None) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def fetch_brave_news(
    *,
    api_key: str,
    q: str,
    count: int = 20,
    offset: int = 1,
    freshness: str | None = "day",
    cache_dir: Path | None = None,
    ttl_seconds: int = 900,
    last_request_ts: float = 0.0,
) -> tuple[BraveFetch, float]:
    q = (q or "").strip()
    if not q:
        raise ValueError("q is required")

    # Brave supports larger `count` values; in practice it may still return fewer than requested.
    count = int(max(1, min(100, count)))
    # Brave uses a small bounded offset window for pagination.
    offset = int(max(0, min(9, offset)))
    freshness = str(freshness).strip() if freshness else None

    cached = False
    requests_made = 0
    cache_path: Path | None = None

    if cache_dir is not None:
        cache_path = cache_dir / f"{_cache_key(q=q, count=count, offset=offset, freshness=freshness)}.json"
        hit = _read_cache(cache_path, ttl_seconds=int(ttl_seconds))
        if hit is not None and isinstance(hit.get("results"), list):
            cached = True
            fetched_at = str(hit.get("fetched_at") or datetime.now(tz=UTC).isoformat(timespec="seconds"))
            return (
                BraveFetch(
                    ok=True,
                    cached=True,
                    requests_made=0,
                    query=q,
                    count=count,
                    offset=offset,
                    freshness=freshness,
                    fetched_at=fetched_at,
                    results=list(hit.get("results") or []),
                    rate_limit=None,
                ),
                last_request_ts,
            )

    # Brave free tier: 1 request/second. Enforce a small buffer.
    now = time.time()
    delta = now - last_request_ts
    if delta < 1.05:
        time.sleep(1.05 - delta)
    last_request_ts = time.time()

    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params: dict[str, Any] = {"q": q, "count": count, "offset": offset}
    if freshness:
        params["freshness"] = freshness

    # Retry briefly on 429/503.
    resp: requests.Response | None = None
    rate_limit: dict[str, Any] | None = None
    retry_after_s: int | None = None
    for attempt in range(1, 4):
        requests_made += 1
        try:
            resp = requests.get(BRAVE_NEWS_ENDPOINT, headers=headers, params=params, timeout=20)
        except requests.RequestException as e:
            if attempt < 3:
                time.sleep(min(2.0 * attempt, 5.0))
                continue
            raise BraveApiError(
                f"Brave API request failed: {e}",
                status_code=None,
                error=str(e),
                rate_limit=None,
                retry_after_s=None,
                requests_made=requests_made,
            ) from e

        # Track the most recent attempt so callers can keep a politeness delay
        # across sequential page fetches even when we retry.
        last_request_ts = time.time()

        rate_limit = _rate_limit_info(resp)
        retry_after_s = _parse_retry_after(resp.headers.get("Retry-After"))

        if resp.status_code in (429, 503) and attempt < 3:
            time.sleep(min(2.0 * attempt, 5.0))
            continue
        break
    if resp is None:
        raise RuntimeError("Brave API request did not execute")

    if resp.status_code != 200:
        try:
            err = resp.json().get("error")
        except Exception:
            err = {"status": resp.status_code, "detail": resp.text[:200]}
        raise BraveApiError(
            f"Brave API error (status {resp.status_code}): {err}",
            status_code=resp.status_code,
            error=err,
            rate_limit=rate_limit,
            retry_after_s=retry_after_s,
            requests_made=requests_made,
        )

    raw = resp.json()
    items = raw.get("results", [])
    if not isinstance(items, list):
        items = []

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
            if len(desc) > 280:
                desc = desc[:277] + "..."
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
                "title": " ".join(title.split())[:200],
                "url": normalize_url(url),
                "domain": domain.lower() if isinstance(domain, str) and domain else None,
                "description": desc,
                "age": it.get("age"),
                "page_age": it.get("page_age"),
            }
        )

    fetched_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    payload = {
        "ok": True,
        "endpoint": BRAVE_NEWS_ENDPOINT,
        "query": q,
        "count": count,
        "offset": offset,
        "freshness": freshness,
        "fetched_at": fetched_at,
        "results": out_items,
    }
    if cache_path is not None:
        _write_cache(cache_path, payload)

    return (
        BraveFetch(
            ok=True,
            cached=cached,
            requests_made=requests_made,
            query=q,
            count=count,
            offset=offset,
            freshness=freshness,
            fetched_at=fetched_at,
            results=out_items,
            rate_limit=rate_limit,
        ),
        last_request_ts,
    )
