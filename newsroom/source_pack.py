from __future__ import annotations

import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from .brave_news import normalize_url
from .image_fetch import extract_og_image_url
from .news_pool_db import NewsPoolDB
from .story_index import choose_key_tokens, tokenize_text


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

# Keep cache rows bounded to avoid pathological pages bloating SQLite.
_MAX_CACHED_TEXT_CHARS = 200_000

_JSONLD_SCRIPT_RE = re.compile(
    r"<script[^>]+type=[\"']application/ld\\+json[\"'][^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

# Anchor terms are used for deterministic “on-topic” scoring so we don't treat long but
# irrelevant extracted text as usable evidence.
_ANCHOR_NUM_RE = re.compile(r"\b\d[\d,\.]{2,}\b")  # >=3 digits or has comma/dot
_ANCHOR_TICKER_RE = re.compile(r"\b[A-Z]{2,6}\b")
_ANCHOR_CAPWORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_ANCHOR_CAMEL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]{2,}\b")
_ANCHOR_CJK_RE = re.compile(r"[\u4E00-\u9FFF]{2,6}")

_ANCHOR_GENERIC = {
    # Boilerplate / non-entities that pollute anchors.
    "breaking",
    "update",
    "updates",
    "latest",
    "live",
    "report",
    "reports",
    "reported",
    "says",
    "said",
    "discover",
    "explore",
    "provided",
    "local",
    "complete",
    "every",
    "list",
    # Publisher/section labels (avoid gluing unrelated stories from the same outlet).
    "sky",
    "bbc",
    "guardian",
    "money",
    "business",
    # Directions / broad regions.
    "north",
    "south",
    "east",
    "west",
    "central",
    # Common geography words that are too broad on their own.
    "hong",
    "kong",
    "china",
    "taiwan",
    "japan",
    "korea",
    "america",
    "britain",
    "england",
    "uk",
    "london",
}

_GENERIC_TICKERS = {
    # Region/currency/boilerplate that often appears in headlines and should not be treated as an "anchor entity".
    "UK",
    "US",
    "EU",
    "UAE",
    "USD",
    "HK",
    "HKSAR",
    "AI",
}

# Paywall / bot-wall hints (very rough; used only for quality scoring).
_PAYWALL_HINTS = (
    "subscribe",
    "subscription",
    "sign in",
    "log in",
    "already a subscriber",
    "register to continue",
    "to continue reading",
    "enable javascript",
    "turn on javascript",
    # zh
    "訂閱",
    "登入",
    "登錄",
    "會員",
    "繼續閱讀",
    "付費",
)


def _focus_anchor_terms(text: str | None) -> set[str]:
    text = (text or "").strip()
    if not text:
        return set()

    terms: set[str] = set()

    for m in _ANCHOR_NUM_RE.findall(text):
        t = m.strip()
        if not t:
            continue
        # Drop probable years.
        if len(t) == 4 and t.isdigit():
            continue
        terms.add(t)

    # Keep short all-caps tickers as-is (case sensitive).
    for m in _ANCHOR_TICKER_RE.findall(text):
        t = m.strip()
        if not t:
            continue
        if t in _GENERIC_TICKERS:
            continue
        terms.add(t)

    # Capitalized words as rough entities (lowercased).
    for m in _ANCHOR_CAPWORD_RE.findall(text):
        t = m.strip()
        if not t:
            continue
        low = t.lower()
        if low in _ANCHOR_GENERIC:
            continue
        terms.add(low)

    # Mixed-case / camelcase entities (SpaceX, OpenAI, xAI, iPhone...).
    for m in _ANCHOR_CAMEL_RE.findall(text):
        t = m.strip()
        if not t:
            continue
        if t.isupper() or t.islower():
            continue
        low = t.lower()
        if low in _ANCHOR_GENERIC:
            continue
        terms.add(low)

    # CJK chunks (keep as-is).
    for m in _ANCHOR_CJK_RE.findall(text):
        t = m.strip()
        if not t:
            continue
        if t in {"最新", "突發", "即時", "報道"}:
            continue
        terms.add(t)

    # Bound to keep JSON small and deterministic.
    if len(terms) > 20:
        terms = set(sorted(terms)[:20])
    return terms


def _count_anchor_hits(haystack: str, terms: set[str]) -> int:
    if not haystack or not terms:
        return 0
    low = haystack.lower()
    hits = 0
    for t in terms:
        if not t:
            continue
        # CJK: match directly. ASCII: case-insensitive.
        if any("\u4e00" <= ch <= "\u9fff" for ch in t):
            if t in haystack:
                hits += 1
        else:
            if t.lower() in low:
                hits += 1
    return hits


def _domain(url: str) -> str | None:
    try:
        return urlsplit(url).hostname
    except Exception:
        return None


def _clean_text(s: str) -> str:
    # Preserve newlines, but normalise spaces.
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_title_from_html(html_text: str) -> str | None:
    m = _TITLE_RE.search(html_text or "")
    if not m:
        return None
    raw = html.unescape(m.group(1) or "")
    t = " ".join(raw.split()).strip()
    return t[:200] if t else None


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    final_url: str
    status: int | None
    content_type: str | None
    html: str | None
    error: str | None


def _fetch_html(
    *,
    url: str,
    timeout_seconds: int,
    max_bytes: int,
    user_agent: str,
) -> FetchResult:
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-HK,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            },
            timeout=timeout_seconds,
            allow_redirects=True,
            stream=True,
        )
    except Exception as e:
        return FetchResult(ok=False, final_url=url, status=None, content_type=None, html=None, error=str(e))

    status = int(resp.status_code)
    final_url = str(getattr(resp, "url", url) or url)
    ctype = str(resp.headers.get("content-type") or "") or None

    if status < 200 or status >= 400:
        return FetchResult(ok=False, final_url=final_url, status=status, content_type=ctype, html=None, error=f"http_status:{status}")

    buf = bytearray()
    try:
        for chunk in resp.iter_content(chunk_size=16_384):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) >= max_bytes:
                break
    except Exception as e:
        return FetchResult(ok=False, final_url=final_url, status=status, content_type=ctype, html=None, error=str(e))

    encoding = resp.encoding or "utf-8"
    try:
        html_text = bytes(buf).decode(encoding, errors="ignore")
    except Exception:
        html_text = bytes(buf).decode("utf-8", errors="ignore")

    return FetchResult(ok=True, final_url=final_url, status=status, content_type=ctype, html=html_text, error=None)


@dataclass(frozen=True)
class ExtractResult:
    ok: bool
    extractor: str | None
    title: str | None
    text: str | None
    error: str | None


def _extract_jsonld(html_text: str) -> ExtractResult:
    for raw in _JSONLD_SCRIPT_RE.findall(html_text or ""):
        payload = (raw or "").strip()
        if not payload:
            continue
        # Some sites wrap JSON-LD in HTML comments or weird whitespace.
        payload = payload.strip().strip("<!--").strip("-->")
        try:
            obj = json.loads(payload)
        except Exception:
            continue

        candidates: list[dict[str, Any]] = []
        if isinstance(obj, dict):
            candidates = [obj]
        elif isinstance(obj, list):
            candidates = [x for x in obj if isinstance(x, dict)]

        for it in candidates:
            typ = it.get("@type") or it.get("type")
            typ_s = " ".join(typ) if isinstance(typ, list) else str(typ or "")
            if "article" not in typ_s.lower():
                # Still allow if articleBody exists.
                if not isinstance(it.get("articleBody"), str):
                    continue
            body = it.get("articleBody")
            if not isinstance(body, str) or not body.strip():
                continue
            title = it.get("headline") if isinstance(it.get("headline"), str) else None
            if not title:
                title = it.get("name") if isinstance(it.get("name"), str) else None
            body_txt = _clean_text(html.unescape(body))
            if len(body_txt) < 400:
                continue
            return ExtractResult(ok=True, extractor="jsonld", title=(" ".join(title.split())[:200] if isinstance(title, str) else None), text=body_txt, error=None)
    return ExtractResult(ok=False, extractor="jsonld", title=None, text=None, error="no_jsonld_articleBody")


def _extract_trafilatura(html_text: str, *, base_url: str | None) -> ExtractResult:
    try:
        import trafilatura  # type: ignore[import-not-found]
    except Exception:
        return ExtractResult(ok=False, extractor="trafilatura", title=None, text=None, error="missing_dependency:trafilatura")
    try:
        text = trafilatura.extract(
            html_text,
            url=base_url,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
    except Exception as e:
        return ExtractResult(ok=False, extractor="trafilatura", title=None, text=None, error=str(e))
    if not isinstance(text, str) or not text.strip():
        return ExtractResult(ok=False, extractor="trafilatura", title=None, text=None, error="empty")
    txt = _clean_text(text)
    if len(txt) < 400:
        return ExtractResult(ok=False, extractor="trafilatura", title=None, text=None, error="too_short")
    # Title: best-effort (trafilatura may not return it in extract()).
    title = _extract_title_from_html(html_text)
    return ExtractResult(ok=True, extractor="trafilatura", title=title, text=txt, error=None)


def _extract_readability(html_text: str) -> ExtractResult:
    try:
        from readability import Document  # type: ignore[import-not-found]
    except Exception:
        return ExtractResult(ok=False, extractor="readability", title=None, text=None, error="missing_dependency:readability-lxml")
    try:
        doc = Document(html_text or "")
        title = doc.short_title()
        summary_html = doc.summary(html_partial=True)
    except Exception as e:
        return ExtractResult(ok=False, extractor="readability", title=None, text=None, error=str(e))

    try:
        import lxml.html  # type: ignore[import-not-found]
    except Exception:
        return ExtractResult(ok=False, extractor="readability", title=None, text=None, error="missing_dependency:lxml")

    try:
        root = lxml.html.fromstring(summary_html)
        text = root.text_content()
    except Exception as e:
        return ExtractResult(ok=False, extractor="readability", title=None, text=None, error=str(e))

    txt = _clean_text(text)
    if len(txt) < 400:
        return ExtractResult(ok=False, extractor="readability", title=None, text=None, error="too_short")
    t = " ".join(str(title or "").split()).strip() or None
    return ExtractResult(ok=True, extractor="readability", title=(t[:200] if t else None), text=txt, error=None)


def _extract_bs4(html_text: str) -> ExtractResult:
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
    except Exception:
        return ExtractResult(ok=False, extractor="bs4", title=None, text=None, error="missing_dependency:bs4")

    def extract_title(soup: Any) -> str | None:
        # Prefer OG title where present.
        try:
            for meta in soup.find_all("meta"):
                prop = (meta.get("property") or "").strip().lower()
                name = (meta.get("name") or "").strip().lower()
                if prop in {"og:title", "twitter:title"} or name in {"og:title", "twitter:title"}:
                    content = meta.get("content")
                    if isinstance(content, str) and content.strip():
                        return " ".join(content.split())[:200]
        except Exception:
            pass
        try:
            if soup.title and isinstance(soup.title.string, str) and soup.title.string.strip():
                return " ".join(soup.title.string.split())[:200]
        except Exception:
            pass
        return None

    def extract_text(node: Any) -> str | None:
        try:
            for tag in node.find_all(["script", "style", "noscript"]):
                tag.decompose()
        except Exception:
            pass
        try:
            # Preserve newlines to help later de-noising/selection.
            text = node.get_text("\n", strip=True)
        except Exception:
            return None
        text = _clean_text(text)
        return text if text else None

    soup = BeautifulSoup(html_text or "", "html.parser")
    title = extract_title(soup)
    for sel in ("article", "main", "body"):
        node = soup.find(sel)
        if not node:
            continue
        txt = extract_text(node)
        if not txt or len(txt) < 400:
            continue
        return ExtractResult(ok=True, extractor="bs4", title=title, text=txt, error=None)

    # Last attempt: soup-wide.
    txt = extract_text(soup)
    if not txt or len(txt) < 400:
        return ExtractResult(ok=False, extractor="bs4", title=title, text=None, error="too_short")
    return ExtractResult(ok=True, extractor="bs4", title=title, text=txt, error=None)


def _quality_score(text: str | None, *, html_text: str | None = None) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    t = text.strip()
    n = len(t)
    # Simple length-based baseline.
    score = 0
    if n >= 400:
        score = 30
    if n >= 800:
        score = 45
    if n >= 1500:
        score = 60
    if n >= 2500:
        score = 72
    if n >= 4000:
        score = 80
    if n >= 8000:
        score = 85

    low = t.lower()
    if any(h in low for h in _PAYWALL_HINTS):
        score -= 30
    if html_text and "paywall" in (html_text or "").lower():
        score -= 10

    return int(max(0, min(100, score)))


def _select_relevant_text(full_text: str, *, focus_terms: set[str] | None, max_chars: int) -> str:
    txt = _clean_text(full_text)
    if not txt:
        return ""
    max_chars = int(max(400, min(50_000, max_chars)))

    # If we have no focus terms, return an early “mostly-clean” slice by paragraph,
    # avoiding mid-paragraph truncation.
    paras = [p.strip() for p in re.split(r"\n{1,}", txt) if p.strip()]
    if not paras:
        return ""

    def para_tokens(p: str) -> set[str]:
        toks = tokenize_text(p, None)
        return choose_key_tokens(toks, drop_high_df=set())

    scored: list[tuple[int, int, str]] = []
    for idx, p in enumerate(paras):
        if len(p) < 60:
            continue
        score = 0
        if focus_terms:
            try:
                kt = para_tokens(p)
                score += 4 * len(kt.intersection(focus_terms))
            except Exception:
                pass
        # Digits often correlate with concrete anchors.
        score += min(6, len(re.findall(r"\d", p)))
        scored.append((score, idx, p))

    # Always keep the opening paragraph if available.
    keep_idx: set[int] = {0}
    # Keep up to 14 best-scoring paragraphs (deterministic).
    best = sorted(scored, key=lambda t: (-t[0], t[1]))
    for score, idx, _p in best[:14]:
        # If focus terms exist, we prefer on-topic paragraphs, but we still need
        # enough material for the reporter to avoid hallucination.
        if focus_terms and score <= 0:
            continue
        keep_idx.add(idx)

    # If selection is too thin, fall back to the first few substantive paragraphs.
    if len(keep_idx) <= 1:
        for i, p in enumerate(paras[:8]):
            if len(p) >= 70:
                keep_idx.add(i)
            if len(keep_idx) >= 4:
                break

    selected = [paras[i] for i in sorted(keep_idx) if 0 <= i < len(paras)]
    out_parts: list[str] = []
    used = 0
    for p in selected:
        # Bound growth without mid-paragraph cuts.
        extra = (2 if out_parts else 0) + len(p)
        if used + extra > max_chars:
            break
        if out_parts:
            out_parts.append("")
            used += 2
        out_parts.append(p)
        used += len(p)

    out = "\n\n".join(out_parts).strip()
    # If still too short, append additional paragraphs by length until we reach a
    # minimum evidence budget (without exceeding max_chars).
    if len(out) < 700:
        # Live blogs sometimes consist of many short updates that individually don't meet
        # our length thresholds. In that case, append in-order to reach a minimum budget.
        remaining = [(i, paras[i]) for i in range(len(paras)) if i not in keep_idx and paras[i]]
        for idx, p in remaining[:12]:
            if used + 2 + len(p) > max_chars:
                break
            out_parts.append("")
            out_parts.append(p)
            keep_idx.add(idx)
            used += 2 + len(p)
            out = "\n\n".join(out_parts).strip()
            if len(out) >= 700:
                break
    return out


def _extract_with_cascade(html_text: str, *, base_url: str | None) -> ExtractResult:
    # Order is important: deterministic and progressively more heuristic.
    for fn in (
        lambda: _extract_jsonld(html_text),
        lambda: _extract_trafilatura(html_text, base_url=base_url),
        lambda: _extract_readability(html_text),
        lambda: _extract_bs4(html_text),
    ):
        r = fn()
        if r.ok and isinstance(r.text, str) and r.text.strip():
            return r
    # If everything failed, return the last attempt error if any.
    return ExtractResult(ok=False, extractor=None, title=_extract_title_from_html(html_text), text=None, error="extraction_failed")


def build_source_pack(
    *,
    story_id: str,
    urls: list[str],
    openclaw_home: Path | None = None,
    cache_db_path: Path | None = None,
    cache_hours: int = 48,
    focus_text: str | None = None,
    focus_terms: set[str] | None = None,
    max_selected_chars: int = 4500,
    timeout_seconds: int = 12,
    max_bytes: int = 1_200_000,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    story_id = (story_id or "").strip()
    urls = [str(u).strip() for u in urls if isinstance(u, str) and str(u).strip()]
    urls = list(dict.fromkeys(urls))

    max_selected_chars = int(max(600, min(50_000, max_selected_chars)))
    timeout_seconds = int(max(3, min(60, timeout_seconds)))
    max_bytes = int(max(50_000, min(3_000_000, max_bytes)))
    user_agent = (user_agent or "").strip() or DEFAULT_USER_AGENT

    if focus_terms is None and isinstance(focus_text, str) and focus_text.strip():
        try:
            toks = tokenize_text(focus_text.strip(), None)
            focus_terms = choose_key_tokens(toks, drop_high_df=set())
        except Exception:
            focus_terms = None
    focus_anchor_terms = _focus_anchor_terms(focus_text if isinstance(focus_text, str) else None)

    now_ts = int(time.time())
    cutoff_ts = now_ts - int(max(1, cache_hours)) * 3600

    # Resolve cache DB location.
    if cache_db_path is None and openclaw_home is not None:
        cache_db_path = Path(openclaw_home) / "data" / "newsroom" / "news_pool.sqlite3"

    cache_enabled = bool(cache_db_path is not None)
    db: NewsPoolDB | None = None
    if cache_enabled:
        db = NewsPoolDB(path=Path(cache_db_path))  # WAL mode for concurrent reads/writes.

    out: dict[str, Any] = {
        "ok": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts)),
        "story_id": story_id,
        "cache": {"enabled": bool(cache_enabled), "db": (str(cache_db_path) if cache_db_path else None), "cutoff_ts": int(cutoff_ts)},
        "stats": {},
        "sources": [],
        "errors": [],
    }

    try:
        for url in urls:
            norm = normalize_url(url)
            cached_row = db.get_article_cache(norm_url=norm) if db else None
            use_cached = False
            if isinstance(cached_row, dict):
                fetched_at_ts = int(cached_row.get("fetched_at_ts") or 0)
                quality = int(cached_row.get("quality_score") or 0)
                text = cached_row.get("extracted_text")
                if fetched_at_ts >= cutoff_ts and isinstance(text, str) and text.strip() and quality >= 25:
                    use_cached = True

            final_url = url
            status: int | None = None
            extracted_title: str | None = None
            extracted_text: str | None = None
            extractor: str | None = None
            quality_score = 0
            error: str | None = None
            cached = False
            og_image_url: str | None = None

            if use_cached and cached_row:
                cached = True
                final_url = str(cached_row.get("final_url") or cached_row.get("url") or url)
                status = int(cached_row.get("http_status")) if cached_row.get("http_status") is not None else None
                extracted_title = cached_row.get("extracted_title") if isinstance(cached_row.get("extracted_title"), str) else None
                extracted_text = cached_row.get("extracted_text") if isinstance(cached_row.get("extracted_text"), str) else None
                extractor = cached_row.get("extractor") if isinstance(cached_row.get("extractor"), str) else None
                quality_score = int(cached_row.get("quality_score") or 0)
                error = cached_row.get("error") if isinstance(cached_row.get("error"), str) else None
            else:
                fetched = _fetch_html(url=url, timeout_seconds=timeout_seconds, max_bytes=max_bytes, user_agent=user_agent)
                final_url = fetched.final_url
                status = fetched.status
                if not fetched.ok or not isinstance(fetched.html, str):
                    error = fetched.error or "fetch_failed"
                else:
                    try:
                        og_image_url = extract_og_image_url(fetched.html, base_url=final_url)
                    except Exception:
                        og_image_url = None
                    extracted_title = _extract_title_from_html(fetched.html)
                    extracted = _extract_with_cascade(fetched.html, base_url=final_url)
                    if extracted.ok and isinstance(extracted.text, str):
                        extracted_text = extracted.text
                        extractor = extracted.extractor
                        extracted_title = extracted.title or extracted_title
                        quality_score = _quality_score(extracted_text, html_text=fetched.html)
                    else:
                        extractor = extracted.extractor
                        error = extracted.error or "extract_failed"
                        quality_score = _quality_score(extracted_title or "", html_text=fetched.html)

                # Store in SQLite cache (even failures) so retries can be rate-limited by time.
                if db:
                    try:
                        db.upsert_article_cache(
                            norm_url=norm,
                            url=url,
                            final_url=final_url,
                            domain=_domain(final_url),
                            http_status=status,
                            extracted_title=extracted_title,
                            extracted_text=((extracted_text[:_MAX_CACHED_TEXT_CHARS]) if extracted_text else None),
                            extractor=extractor,
                            fetched_at_ts=now_ts,
                            quality_score=int(quality_score),
                            error=error,
                        )
                    except Exception as e:
                        out["errors"].append({"url": url, "stage": "cache_write", "error": str(e)})

            selected_text = None
            if isinstance(extracted_text, str) and extracted_text.strip():
                selected_text = _select_relevant_text(extracted_text, focus_terms=focus_terms, max_chars=max_selected_chars)
                if not selected_text:
                    selected_text = None

            focus_hits = 0
            anchor_hits = 0
            on_topic = False
            selected_chars = len(selected_text.strip()) if isinstance(selected_text, str) else 0

            if isinstance(selected_text, str) and selected_text.strip():
                # Deterministic on-topic scoring: overlap with focus key terms + anchor term hits.
                hay = " ".join([str(extracted_title or "").strip(), selected_text.strip()]).strip()
                if focus_terms:
                    try:
                        focus_hits = len(tokenize_text(hay, None).intersection(focus_terms))
                    except Exception:
                        focus_hits = 0
                if focus_anchor_terms:
                    anchor_hits = _count_anchor_hits(hay, focus_anchor_terms)

                # Conservative: require at least 2 anchor hits, or 1 anchor + 1 focus hit.
                # This prevents generic “market/discount” glue from making irrelevant pages look usable.
                if focus_anchor_terms:
                    # Tighten: a single anchor hit is easy to get from locations/outlet boilerplate.
                    if anchor_hits >= 2 or (anchor_hits >= 1 and focus_hits >= 2) or focus_hits >= 4:
                        on_topic = True
                else:
                    # Without anchors, require higher lexical overlap.
                    if focus_hits >= 3:
                        on_topic = True

            if error:
                out["errors"].append({"url": url, "stage": "fetch/extract", "error": error, "http_status": status, "extractor": extractor})

            out["sources"].append(
                {
                    "url": url,
                    "norm_url": norm,
                    "final_url": final_url,
                    "domain": _domain(final_url),
                    "http_status": status,
                    "title": extracted_title,
                    "og_image_url": og_image_url,
                    "extractor": extractor,
                    "quality_score": int(quality_score),
                    "cached": bool(cached),
                    # Selected/denoised evidence for the reporter (bounded).
                    # Keep only on-topic evidence in the pack to avoid wasting LLM tokens.
                    "text": (selected_text if on_topic else None),
                    "selected_chars": int(selected_chars),
                    "on_topic": bool(on_topic),
                    "focus_hits": int(focus_hits),
                    "anchor_hits": int(anchor_hits),
                    # Full text is kept in SQLite cache when enabled.
                    "full_text_cached": bool(cache_enabled),
                }
            )

        if not out["sources"]:
            out["ok"] = False
            out["errors"].append({"stage": "final", "error": "no_sources"})

        # Reorder sources so usable evidence comes first. This makes downstream
        # ">=2 usable sources" gates more reliable (models often don't scan long lists).
        def is_usable(src: dict[str, Any]) -> bool:
            try:
                return int(src.get("selected_chars") or 0) >= 400
            except Exception:
                return False

        def is_on_topic(src: dict[str, Any]) -> bool:
            return bool(src.get("on_topic") is True)

        def is_usable_on_topic(src: dict[str, Any]) -> bool:
            return is_usable(src) and is_on_topic(src)

        def text_len(src: dict[str, Any]) -> int:
            try:
                return int(src.get("selected_chars") or 0)
            except Exception:
                return 0

        def score(src: dict[str, Any]) -> tuple[int, int, int, str, str]:
            usable_on_topic_rank = 0 if is_usable_on_topic(src) else 1
            usable_rank = 0 if is_usable(src) else 1
            q = int(src.get("quality_score") or 0)
            tl = text_len(src)
            dom = str(src.get("domain") or "")
            url = str(src.get("final_url") or src.get("url") or "")
            return (usable_on_topic_rank, usable_rank, -q, -tl, dom, url)

        out["sources"] = sorted(out["sources"], key=score)

        usable = [s for s in out["sources"] if is_usable(s)]
        usable_on_topic = [s for s in out["sources"] if is_usable_on_topic(s)]
        out["stats"] = {
            "total_sources": len(out["sources"]),
            "usable_min_chars": 400,
            "usable_sources_count": len(usable),
            "on_topic_sources_count": len(usable_on_topic),
            "usable_domains": sorted({str(s.get("domain") or "") for s in usable_on_topic if s.get("domain")}),
            "focus_terms_count": len(focus_terms or []),
            "focus_anchor_terms_count": len(focus_anchor_terms or []),
        }
        return out
    finally:
        if db is not None:
            db.close()
