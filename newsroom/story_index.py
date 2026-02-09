from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable


_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'-]{1,}", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u4E00-\u9FFF]{2,}")

# Anchor-ish regexes (used for clustering across different angles/framing).
_ANCHOR_NUM_RE = re.compile(r"\b\d[\d,\.]{2,}\b")  # >=3 digits or has comma/dot
_ANCHOR_TICKER_RE = re.compile(r"\b[A-Z]{2,6}\b")
_ANCHOR_CAPWORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_ANCHOR_CAMEL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]{2,}\b")
_ANCHOR_CJK_RE = re.compile(r"[\u4E00-\u9FFF]{2,6}")


# Intentionally small: we only need to remove the most common glue words.
# We also filter "high-frequency" tokens dynamically per pool window.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "out",
    "says",
    "said",
    "say",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    # Glue words that cause over-clustering.
    "about",
    "could",
    "one",
    "than",
    "which",
    # News boilerplate.
    "after",
    "amid",
    "before",
    "during",
    "new",
    "news",
    "today",
    "yesterday",
    "tomorrow",
    "this",
    "these",
    "those",
    "more",
    "most",
    "breaking",
    "live",
    "latest",
    "update",
    "updates",
    "report",
    "reports",
    "reported",
    "according",
    # Common glue words that merge unrelated stories.
    "market",
    "markets",
    "price",
    "prices",
    "sale",
    "sales",
    "sell",
    "sells",
    "selling",
    "sold",
    "discount",
    "discounts",
    "deal",
    "deals",
    "acquire",
    "acquires",
    "acquired",
    "acquisition",
    "merge",
    "merger",
    "merges",
    "merged",
    "buy",
    "buys",
    "bought",
    "shares",
    "share",
    "stock",
    "stocks",
    "earnings",
    "revenue",
    "profit",
    "loss",
    "million",
    "billion",
    "trillion",
    "percent",
    "home",
    "house",
    "listed",
    "listing",
    "listings",
    # Directions / broad region words (often not the actual subject).
    "north",
    "south",
    "east",
    "west",
    "central",
}

_ANCHOR_GENERIC = {
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
    # Publisher/boilerplate terms (avoid glue).
    "the",
    "motley",
    "fool",
    "reuters",
    "bloomberg",
    "associated",
    "press",
    "ap",
    # Overly broad places.
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


_CJK_STOP = {
    "最新",
    "突發",
    "即時",
    "報道",
    "消息",
    "更新",
    "宣布",
    "表示",
}


def _now_ts() -> int:
    return int(time.time())


def parse_page_age_ts(page_age: str | None) -> int | None:
    if not isinstance(page_age, str) or not page_age.strip():
        return None
    s = page_age.strip()
    # Brave typically returns ISO without timezone.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def tokenize_text(title: str | None, description: str | None) -> set[str]:
    parts = []
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    text = " ".join(parts)
    if not text:
        return set()

    tokens: set[str] = set()
    low = text.lower()
    for w in _WORD_RE.findall(low):
        w = w.strip("-'")
        if not w or w in _STOPWORDS:
            continue
        # Drop years + pure digits (very noisy for clustering).
        if w.isdigit():
            continue
        if len(w) == 4 and w.isdigit():
            continue
        tokens.add(w)

    # CJK: use bigrams for rough segmentation; remove very common boilerplate bigrams.
    for seq in _CJK_RE.findall(text):
        seq = seq.strip()
        if len(seq) < 2:
            continue
        # Whole token helps when seq is a short proper noun like "英國" / "美國".
        if len(seq) <= 6 and seq not in _CJK_STOP:
            tokens.add(seq)
        for i in range(len(seq) - 1):
            bg = seq[i : i + 2]
            if bg in _CJK_STOP:
                continue
            tokens.add(bg)

    return tokens


def anchor_terms(title: str | None, description: str | None) -> set[str]:
    """Extract rough, stable anchor terms for clustering across different framing.

    These are intentionally crude but deterministic: numbers (>=3 digits), tickers,
    capitalized entity-ish words, and short CJK chunks.
    """
    parts = []
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    text = " ".join(parts).strip()
    if not text:
        return set()

    out: set[str] = set()

    for m in _ANCHOR_NUM_RE.findall(text):
        t = m.strip()
        if not t:
            continue
        # Drop probable years.
        if len(t) == 4 and t.isdigit():
            continue
        out.add(t)

    for m in _ANCHOR_TICKER_RE.findall(text):
        t = m.strip()
        if not t:
            continue
        if t in _GENERIC_TICKERS:
            continue
        out.add(t)

    for m in _ANCHOR_CAPWORD_RE.findall(text):
        t = m.strip()
        if not t:
            continue
        low = t.lower()
        if low in _ANCHOR_GENERIC or low in _STOPWORDS:
            continue
        out.add(low)

    # Mixed-case / camelcase entities (SpaceX, OpenAI, xAI, iPhone...). We keep
    # them lowercased so they match across wording variants.
    for m in _ANCHOR_CAMEL_RE.findall(text):
        t = m.strip()
        if not t:
            continue
        if t.isupper() or t.islower():
            # handled by ticker/capword/normal tokenization paths
            continue
        low = t.lower()
        if low in _ANCHOR_GENERIC or low in _STOPWORDS:
            continue
        out.add(low)

    for m in _ANCHOR_CJK_RE.findall(text):
        t = m.strip()
        if not t:
            continue
        if t in _CJK_STOP:
            continue
        out.add(t)

    if len(out) > 30:
        out = set(sorted(out)[:30])
    return out


def _anchor_is_signal(tok: str) -> bool:
    if not tok:
        return False
    if tok.isdigit() and len(tok) == 4:
        return False
    # Tickers.
    if tok.isupper() and 2 <= len(tok) <= 6:
        if tok in _GENERIC_TICKERS:
            return False
        return True
    # CJK entities.
    if any("\u4e00" <= ch <= "\u9fff" for ch in tok):
        return True
    low = tok.lower()
    if low in _ANCHOR_GENERIC:
        return False
    # Numbers (>=3 digits or comma/dot).
    if any(ch.isdigit() for ch in tok):
        return len(tok) >= 3
    # Entity-ish ASCII words.
    return tok.isascii() and len(tok) >= 4


def _token_is_signal(tok: str) -> bool:
    if not tok:
        return False
    if tok in _STOPWORDS:
        return False
    if tok.isdigit():
        return False
    # Prefer longer Latin words and any CJK token (which is already >=2 chars).
    if any("a" <= ch <= "z" for ch in tok):
        if len(tok) < 4:
            return False
    return True


def choose_key_tokens(tokens: set[str], *, drop_high_df: set[str]) -> set[str]:
    out = {t for t in tokens if _token_is_signal(t) and t not in drop_high_df}
    # Keep it bounded for speed and determinism.
    if len(out) > 40:
        out = set(sorted(out)[:40])
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a.intersection(b)
    if not inter:
        return 0.0
    union = a.union(b)
    return float(len(inter)) / float(len(union))


@dataclass(frozen=True)
class LinkDoc:
    norm_url: str
    url: str
    domain: str | None
    title: str | None
    description: str | None
    published_ts: int | None
    last_seen_ts: int
    seen_count: int
    tokens: set[str]
    key_tokens: set[str]
    anchor_tokens: set[str]


@dataclass
class Cluster:
    event_key: str
    docs: list[LinkDoc]
    seed_tokens: set[str]
    key_union: set[str]
    anchor_seed_tokens: set[str]
    anchor_union: set[str]
    best_published_ts: int | None
    last_seen_ts: int


def _event_key_from_terms(terms: list[str]) -> str:
    sig = "|".join(terms)
    digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:20]
    return f"event:{digest}"


def _cluster_signature_terms(cluster: Cluster) -> list[str]:
    freq: dict[str, int] = {}
    for d in cluster.docs:
        for t in d.key_tokens:
            freq[t] = freq.get(t, 0) + 1
    # Sort by (freq desc, token asc) for determinism.
    ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    terms = [t for t, _ in ordered if _token_is_signal(t)]
    return terms[:10]


def _cluster_core_terms(cluster: Cluster) -> list[str]:
    """Return stable-ish "core" terms for event_key generation.

    We prefer terms that appear in a majority of sources for the cluster so that
    adding a new source with extra wording doesn't flip the event_key.
    """
    freq: dict[str, int] = {}
    for d in cluster.docs:
        for t in d.key_tokens:
            freq[t] = freq.get(t, 0) + 1
    if not freq:
        return []

    size = len(cluster.docs)
    # Majority threshold (but never less than 2 so single-source clusters don't
    # generate brittle keys that collide).
    min_freq = max(2, int(math.ceil(size * 0.6)))

    ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    core = [t for t, c in ordered if c >= min_freq and _token_is_signal(t)]
    return core[:12]


def cluster_signature_terms(cluster: Cluster) -> list[str]:
    """Return a small deterministic set of terms that explains the cluster.

    This is used for human/audit readability and planner grounding; it is not
    intended to be a perfect "entity extractor".
    """
    return _cluster_signature_terms(cluster)


def cluster_links(links: Iterable[dict[str, Any]], *, now_ts: int | None = None) -> list[Cluster]:
    now_ts = _now_ts() if now_ts is None else int(now_ts)

    raw_docs: list[LinkDoc] = []
    all_tokens: list[set[str]] = []
    all_anchors: list[set[str]] = []
    for it in links:
        if not isinstance(it, dict):
            continue
        url = it.get("url")
        norm_url = it.get("norm_url") or it.get("normUrl") or it.get("norm")
        if not isinstance(url, str) or not url.strip():
            continue
        if not isinstance(norm_url, str) or not norm_url.strip():
            norm_url = url.strip()
        title = it.get("title") if isinstance(it.get("title"), str) else None
        desc = it.get("description") if isinstance(it.get("description"), str) else None
        domain = it.get("domain") if isinstance(it.get("domain"), str) else None
        last_seen_ts = int(it.get("last_seen_ts") or it.get("lastSeenTs") or it.get("last_seen") or now_ts)
        seen_count = int(it.get("seen_count") or it.get("seenCount") or 1)
        published_ts = parse_page_age_ts(it.get("page_age") if isinstance(it.get("page_age"), str) else None)

        tokens = tokenize_text(title, desc)
        anchors = anchor_terms(title, desc)
        all_tokens.append(tokens)
        all_anchors.append(anchors)
        # key_tokens filled after DF pass
        raw_docs.append(
            LinkDoc(
                norm_url=norm_url.strip(),
                url=url.strip(),
                domain=domain.lower().strip() if isinstance(domain, str) and domain else None,
                title=title,
                description=desc,
                published_ts=published_ts,
                last_seen_ts=last_seen_ts,
                seen_count=seen_count,
                tokens=tokens,
                key_tokens=set(),
                anchor_tokens=anchors,
            )
        )

    if not raw_docs:
        return []

    # Compute DF and drop very common tokens so "hong kong"/"government"/etc don't
    # glue unrelated stories together.
    df: dict[str, int] = {}
    for toks in all_tokens:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n = len(raw_docs)
    # If a token appears in >= ~5% of docs (min 8), treat as non-signal for clustering.
    # (The pool window is small; we want to aggressively drop glue tokens like "market".)
    df_cap = max(8, int(n * 0.05))
    drop_high_df = {t for t, c in df.items() if c >= df_cap}

    anchor_df: dict[str, int] = {}
    for toks in all_anchors:
        for t in toks:
            anchor_df[t] = anchor_df.get(t, 0) + 1
    anchor_df_cap = max(6, int(n * 0.04))
    drop_anchor_high_df = {t for t, c in anchor_df.items() if c >= anchor_df_cap}

    docs: list[LinkDoc] = []
    for d in raw_docs:
        kt = choose_key_tokens(d.tokens, drop_high_df=drop_high_df)
        at = {t for t in d.anchor_tokens if t not in drop_anchor_high_df and _anchor_is_signal(t)}
        if len(at) > 25:
            at = set(sorted(at)[:25])
        docs.append(
            LinkDoc(
                norm_url=d.norm_url,
                url=d.url,
                domain=d.domain,
                title=d.title,
                description=d.description,
                published_ts=d.published_ts,
                last_seen_ts=d.last_seen_ts,
                seen_count=d.seen_count,
                tokens=d.tokens,
                key_tokens=kt,
                anchor_tokens=at,
            )
        )

    # Sort by recency to make clustering deterministic.
    def sort_key(d: LinkDoc) -> tuple[int, int, str]:
        pub = d.published_ts or 0
        return (pub, d.last_seen_ts, d.norm_url)

    docs.sort(key=sort_key, reverse=True)

    clusters: list[Cluster] = []
    key_to_clusters: dict[str, set[int]] = {}
    anchor_to_clusters: dict[str, set[int]] = {}

    def add_cluster(doc: LinkDoc) -> int:
        idx = len(clusters)
        cluster = Cluster(
            event_key="",
            docs=[doc],
            seed_tokens=set(doc.key_tokens),
            key_union=set(doc.key_tokens),
            anchor_seed_tokens=set(doc.anchor_tokens),
            anchor_union=set(doc.anchor_tokens),
            best_published_ts=doc.published_ts,
            last_seen_ts=doc.last_seen_ts,
        )
        clusters.append(cluster)
        for t in doc.key_tokens:
            key_to_clusters.setdefault(t, set()).add(idx)
        for t in doc.anchor_tokens:
            anchor_to_clusters.setdefault(t, set()).add(idx)
        return idx

    def update_cluster(idx: int, doc: LinkDoc) -> None:
        c = clusters[idx]
        c.docs.append(doc)
        c.key_union |= doc.key_tokens
        c.anchor_union |= doc.anchor_tokens
        c.last_seen_ts = max(c.last_seen_ts, doc.last_seen_ts)
        if doc.published_ts:
            c.best_published_ts = max(c.best_published_ts or 0, doc.published_ts)
        for t in doc.key_tokens:
            key_to_clusters.setdefault(t, set()).add(idx)
        for t in doc.anchor_tokens:
            anchor_to_clusters.setdefault(t, set()).add(idx)

    for doc in docs:
        candidates: set[int] = set()
        for t in doc.key_tokens:
            candidates |= key_to_clusters.get(t, set())
        for t in doc.anchor_tokens:
            candidates |= anchor_to_clusters.get(t, set())

        best_idx = None
        best_sim = 0.0
        best_anchor_ok = False
        # Limit comparisons for speed.
        for idx in sorted(candidates)[:60]:
            c = clusters[idx]
            sim_key = max(
                jaccard(doc.key_tokens, c.seed_tokens),
                jaccard(doc.key_tokens, c.key_union),
            )
            sim_anchor = 0.0
            if doc.anchor_tokens and (c.anchor_union or c.anchor_seed_tokens):
                sim_anchor = max(
                    jaccard(doc.anchor_tokens, c.anchor_seed_tokens),
                    jaccard(doc.anchor_tokens, c.anchor_union),
                )

            inter = doc.anchor_tokens.intersection(c.anchor_union)
            anchor_ok = False
            if inter:
                if len(inter) >= 2:
                    anchor_ok = True
                else:
                    # Single-term anchors are only trusted if the term is strong (ticker/number/CJK).
                    t = next(iter(inter))
                    if _anchor_is_signal(t) and (t.isupper() or any(ch.isdigit() for ch in t) or any("\u4e00" <= ch <= "\u9fff" for ch in t)):
                        anchor_ok = True

            sim = sim_key
            if sim_anchor > 0 and anchor_ok:
                sim = max(sim, sim_anchor * 1.35)
            elif sim_anchor > 0:
                sim = max(sim, sim_anchor * 0.9)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
                best_anchor_ok = anchor_ok

        # Tuned conservative: avoid over-merging.
        if best_idx is None:
            add_cluster(doc)
            continue
        # If we don't have strong anchor alignment, require a higher lexical overlap.
        if not best_anchor_ok and best_sim < 0.35:
            add_cluster(doc)
        else:
            if best_sim < 0.30:
                add_cluster(doc)
            else:
                update_cluster(best_idx, doc)

    # Finalize event_key.
    for c in clusters:
        core = _cluster_core_terms(c)
        terms = core or _cluster_signature_terms(c)
        if not terms:
            # Fallback: stable hash from representative URL.
            terms = [c.docs[0].norm_url[:80]]
        c.event_key = _event_key_from_terms(terms)

    return clusters


def rank_clusters(clusters: list[Cluster], *, now_ts: int | None = None) -> list[Cluster]:
    now_ts = _now_ts() if now_ts is None else int(now_ts)

    def score(c: Cluster) -> tuple[float, int, int, str]:
        # Primary: best published time (newer first). Fallback: last_seen_ts.
        pub = c.best_published_ts or c.last_seen_ts
        age_s = max(0, now_ts - int(pub))
        age_hours = age_s / 3600.0
        recency = -age_hours
        # Secondary: cluster size and total seen_count (proxy for "importance").
        size = len(c.docs)
        seen = sum(d.seen_count for d in c.docs)
        return (recency, size, seen, c.event_key)

    return sorted(clusters, key=score, reverse=True)


def suggest_category(title: str | None, description: str | None) -> str:
    text = " ".join([t.strip() for t in [title or "", description or ""] if t.strip()])
    low = text.lower()
    words = set(_WORD_RE.findall(low))

    # Order aligned to planner rules (rough).
    if any(k in low for k in ("parliament", "westminster", "downing street", "uk government")):
        return "UK Parliament / Politics"
    if any(k in text for k in ("MIRROR", "叱咤", "演唱會", "港產片", "香港歌手", "藝人", "男團")):
        return "Hong Kong Entertainment"
    if any(k in low for k in ("transfer", "injury", "tournament", "match", "nba", "nfl", "premier league", "fifa")) or any(
        k in text for k in ("轉會", "球會", "聯賽", "賽事", "受傷")
    ):
        return "Sports"
    if any(k in low for k in ("movie", "film", "tv", "netflix", "disney", "hollywood", "celebrity")):
        return "Entertainment"
    if " ai " in f" {low} " or any(k in low for k in ("artificial intelligence", "grok", "openai", "anthropic", "model", "llm")) or any(
        k in text for k in ("人工智能", "大模型", "模型", "算力")
    ):
        return "AI"
    if (
        any(
            w in words
            for w in (
                "politics",
                "election",
                "elections",
                "congress",
                "senate",
                "government",
                "lawmakers",
                "bill",
                "bills",
                "legislation",
                "cabinet",
                "president",
                "parliamentary",
                "policy",
                "minister",
                "ministers",
            )
        )
        or "white house" in low
        or ("prime" in words and "minister" in words)
        or any(k in text for k in ("政治", "選舉", "國會", "參議院", "眾議院", "法案", "立法", "議員", "內閣", "總統", "首相", "政府"))
    ):
        return "Politics"
    if any(k in low for k in ("earnings", "shares", "stock", "nasdaq", "nyse", "sec", "ipo")):
        return "US Stocks"
    if any(k in low for k in ("bitcoin", "ethereum", "crypto", "token", "etf")):
        return "Crypto"
    if any(k in low for k in ("gold", "silver", "platinum", "palladium")):
        return "Precious Metals"
    if any(k in text for k in ("香港", "港府", "中環", "立法會")):
        return "Hong Kong News"
    if any(k in text for k in ("英國", "倫敦", "蘇格蘭", "威爾斯")) or " uk " in f" {low} ":
        return "UK News"
    return "Global News"
