from __future__ import annotations

import email.utils
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import yaml
from lxml import etree

from newsroom.brave_news import normalize_url

_USER_AGENT = "OpenClaw-Newsroom/1.0"


@dataclass(frozen=True)
class RssFeed:
    key: str        # e.g. "bbc-world"
    url: str        # feed URL
    label: str      # e.g. "BBC World"
    region: str     # "global" | "uk" | "hk"


@dataclass(frozen=True)
class RssArticle:
    url: str
    title: str
    description: str | None
    published: str | None    # ISO 8601 (for PoolLink.page_age)
    domain: str | None


@dataclass(frozen=True)
class RssFetch:
    ok: bool
    feed_key: str
    requests_made: int
    fetched_at: str
    articles: list[RssArticle]
    error: str | None = None


DEFAULT_FEEDS: list[dict[str, str]] = [
    # UK
    {"key": "bbc-world", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "label": "BBC World", "region": "uk"},
    {"key": "bbc-uk", "url": "https://feeds.bbci.co.uk/news/uk/rss.xml", "label": "BBC UK", "region": "uk"},
    {"key": "bbc-tech", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml", "label": "BBC Technology", "region": "uk"},
    {"key": "guardian-world", "url": "https://www.theguardian.com/world/rss", "label": "Guardian World", "region": "uk"},
    {"key": "sky-news", "url": "https://feeds.skynews.com/feeds/rss/home.xml", "label": "Sky News", "region": "uk"},
    {"key": "dailymail-news", "url": "https://www.dailymail.co.uk/articles.rss", "label": "Daily Mail", "region": "uk"},
    {"key": "telegraph-news", "url": "https://www.telegraph.co.uk/rss.xml", "label": "The Telegraph", "region": "uk"},
    # HK
    {"key": "rthk-en", "url": "https://rthk.hk/rthk/news/rss/e_expressnews_elocal.xml", "label": "RTHK English", "region": "hk"},
    {"key": "scmp-hk", "url": "https://www.scmp.com/rss/91/feed", "label": "SCMP HK", "region": "hk"},
    # Global
    {"key": "aljazeera-news", "url": "https://www.aljazeera.com/xml/rss/all.xml", "label": "Al Jazeera", "region": "global"},
    {"key": "bbc-business", "url": "https://feeds.bbci.co.uk/news/business/rss.xml", "label": "BBC Business", "region": "global"},
]


def load_feeds(config_path: Path | None = None) -> list[RssFeed]:
    """Load feeds from rss_feeds.yaml, fallback to DEFAULT_FEEDS."""
    if config_path is not None and config_path.exists():
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("feeds"), list):
                feeds: list[RssFeed] = []
                for f in data["feeds"]:
                    if not isinstance(f, dict):
                        continue
                    key = f.get("key")
                    url = f.get("url")
                    label = f.get("label", key)
                    region = f.get("region", "global")
                    if isinstance(key, str) and isinstance(url, str) and key.strip() and url.strip():
                        feeds.append(RssFeed(key=key.strip(), url=url.strip(), label=str(label), region=str(region)))
                if feeds:
                    return feeds
        except Exception:
            pass

    return [
        RssFeed(key=f["key"], url=f["url"], label=f["label"], region=f["region"])
        for f in DEFAULT_FEEDS
    ]


def _parse_rss_date(s: str | None) -> str | None:
    """Parse RSS/Atom date to ISO 8601 string.

    Tries RFC 822 (email.utils) first, then ISO 8601 fromisoformat.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()

    # RFC 822 (common in RSS 2.0).
    try:
        dt = email.utils.parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat(timespec="seconds")
    except (ValueError, TypeError, OverflowError):
        pass

    # ISO 8601 (common in Atom).
    raw = s
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat(timespec="seconds")
    except (ValueError, OverflowError):
        pass

    return None


def _text_or_none(el: etree._Element | None) -> str | None:
    if el is None:
        return None
    t = el.text
    if not isinstance(t, str) or not t.strip():
        return None
    return " ".join(t.split())


def _domain_from_url(url: str) -> str | None:
    try:
        h = urlsplit(url).hostname
        return h.lower() if h else None
    except Exception:
        return None


def parse_rss_xml(xml_bytes: bytes) -> list[RssArticle]:
    """Parse RSS 2.0 (<item>) or Atom (<entry>) from raw XML bytes."""
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return []

    articles: list[RssArticle] = []

    # RSS 2.0: <rss><channel><item>
    items = root.findall(".//item")
    if items:
        for item in items:
            link_el = item.find("link")
            title_el = item.find("title")
            url = _text_or_none(link_el)
            title = _text_or_none(title_el)
            if not url or not title:
                continue
            norm = normalize_url(url)
            if not norm:
                continue

            desc_el = item.find("description")
            desc = _text_or_none(desc_el)
            if desc and len(desc) > 280:
                desc = desc[:277] + "..."

            pub_el = item.find("pubDate")
            published = _parse_rss_date(_text_or_none(pub_el))

            articles.append(
                RssArticle(
                    url=norm,
                    title=title[:200],
                    description=desc,
                    published=published,
                    domain=_domain_from_url(url),
                )
            )
        return articles

    # Atom: <feed><entry>
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall(".//atom:entry", ns)
    if not entries:
        # Try without namespace (some feeds use default ns).
        entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    if not entries:
        entries = root.findall(".//entry")

    for entry in entries:
        # Atom <link href="..."/>
        link_el = entry.find("atom:link[@href]", ns)
        if link_el is None:
            link_el = entry.find("{http://www.w3.org/2005/Atom}link[@href]")
        if link_el is None:
            link_el = entry.find("link[@href]")

        url: str | None = None
        if link_el is not None:
            url = link_el.get("href")
        if not url:
            # Fallback: <link> with text content.
            link_text_el = entry.find("atom:link", ns)
            if link_text_el is None:
                link_text_el = entry.find("{http://www.w3.org/2005/Atom}link")
            if link_text_el is None:
                link_text_el = entry.find("link")
            url = _text_or_none(link_text_el)

        title_el = entry.find("atom:title", ns)
        if title_el is None:
            title_el = entry.find("{http://www.w3.org/2005/Atom}title")
        if title_el is None:
            title_el = entry.find("title")
        title = _text_or_none(title_el)

        if not url or not title:
            continue
        norm = normalize_url(url)
        if not norm:
            continue

        # Description from <summary> or <content>.
        desc = None
        for tag in ["atom:summary", "{http://www.w3.org/2005/Atom}summary", "summary",
                     "atom:content", "{http://www.w3.org/2005/Atom}content", "content"]:
            el = entry.find(tag, ns) if ":" in tag and not tag.startswith("{") else entry.find(tag)
            if el is not None:
                desc = _text_or_none(el)
                if desc:
                    break
        if desc and len(desc) > 280:
            desc = desc[:277] + "..."

        # Published date.
        pub_date = None
        for tag in ["atom:published", "{http://www.w3.org/2005/Atom}published", "published",
                     "atom:updated", "{http://www.w3.org/2005/Atom}updated", "updated"]:
            el = entry.find(tag, ns) if ":" in tag and not tag.startswith("{") else entry.find(tag)
            if el is not None:
                pub_date = _parse_rss_date(_text_or_none(el))
                if pub_date:
                    break

        articles.append(
            RssArticle(
                url=norm,
                title=title[:200],
                description=desc,
                published=pub_date,
                domain=_domain_from_url(url),
            )
        )

    return articles


def fetch_rss_feed(
    feed: RssFeed,
    *,
    timeout: int = 15,
    last_request_ts: float = 0.0,
) -> tuple[RssFetch, float]:
    """Fetch and parse one RSS/Atom feed.

    Returns (RssFetch, last_request_ts).
    0.5s politeness delay between calls.
    """
    # Politeness delay.
    now = time.time()
    delta = now - last_request_ts
    if delta < 0.5:
        time.sleep(0.5 - delta)
    last_request_ts = time.time()

    fetched_at = datetime.now(tz=UTC).isoformat(timespec="seconds")

    try:
        resp = requests.get(
            feed.url,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return (
            RssFetch(
                ok=False,
                feed_key=feed.key,
                requests_made=1,
                fetched_at=fetched_at,
                articles=[],
                error=str(e),
            ),
            last_request_ts,
        )

    articles = parse_rss_xml(resp.content)
    return (
        RssFetch(
            ok=True,
            feed_key=feed.key,
            requests_made=1,
            fetched_at=fetched_at,
            articles=articles,
        ),
        last_request_ts,
    )
