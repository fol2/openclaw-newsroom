"""Event-centric clustering manager for the newsroom.

Clusters links into events using one-link-per-prompt LLM calls (Gemini Flash).
Replaces the old token-based clustering (story_index.py) and per-URL
semantic fingerprinting (semantic_cluster.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from newsroom.story_index import (
    anchor_terms as _si_anchor_terms,
    choose_key_tokens,
    jaccard,
    tokenize_text as _si_tokenize_text,
)

logger = logging.getLogger(__name__)

# Category list used in clustering prompts.
CATEGORY_LIST = [
    "UK Parliament / Politics",
    "UK News",
    "Politics",
    "AI",
    "Sports",
    "Hong Kong Entertainment",
    "Entertainment",
    "US Stocks",
    "Crypto",
    "Precious Metals",
    "Global News",
    "Hong Kong News",
]

_ASSIGNMENT_MIN_CONFIDENCE = 0.70
_LINK_FLAGS_ALLOWED = {"roundup", "opinion", "live_updates", "multi_topic"}
_LINK_FLAGS_FORCE_NEW_EVENT = {"roundup", "multi_topic"}

_CANONICAL_CATEGORY_BY_LOWER = {c.lower(): c for c in CATEGORY_LIST}
_CATEGORY_ALIASES = {
    "us news": "Global News",
    "world news": "Global News",
    "technology": "AI",
    "tech": "AI",
    "technology / tech": "AI",
}


def _normalise_category(raw: str | None) -> str:
    """Map a category label onto the canonical category set."""
    if raw is None:
        return "Global News"
    s = str(raw).strip()
    if not s:
        return "Global News"
    low = s.lower()
    # Normalise slash spacing so labels like "UK Parliament/Politics" match our
    # canonical "UK Parliament / Politics" category strings.
    low = re.sub(r"\s*/\s*", " / ", low)
    direct = _CANONICAL_CATEGORY_BY_LOWER.get(low)
    if direct:
        return direct
    alias = _CATEGORY_ALIASES.get(low)
    if alias:
        return alias
    return "Global News"


# ---------------------------------------------------------------------------
# Retrieve-then-decide helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventTokens:
    key_tokens: frozenset[str]
    anchor_tokens: frozenset[str]

_TOKEN_CACHE_DF_SIG_KEY = "__nr_drop_high_df_sig__"


def _drop_high_df_signature(drop_high_df: set[str]) -> str:
    if not drop_high_df:
        return "df0"
    h = hashlib.sha1()
    for t in sorted(drop_high_df):
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def _compute_drop_high_df(events: list[dict[str, Any]]) -> set[str]:
    """Compute a deterministic set of high-document-frequency tokens to drop.

    We drop very common tokens across the retrieval universe to avoid glue terms
    like "government" causing unrelated matches.
    """
    if not events:
        return set()

    df: dict[str, int] = {}
    for ev in events:
        summary = str(ev.get("summary_en") or "").strip()
        title = str(ev.get("title") or "").strip()
        toks = _si_tokenize_text(summary or title, title if summary else None)
        # Use "signal" key tokens before DF filtering so DF reflects what would
        # actually influence retrieval scoring.
        kt = choose_key_tokens(toks, drop_high_df=set())
        for t in kt:
            df[t] = df.get(t, 0) + 1

    # If a token appears in >= ~5% of events (min 8), treat it as non-signal.
    n = len(events)
    df_cap = max(8, int(n * 0.05))
    return {t for t, c in df.items() if c >= df_cap}


def _tokenize_event(event: dict[str, Any], *, drop_high_df: set[str] | None = None) -> EventTokens:
    """Tokenize event summary_en + title for retrieval matching."""
    drop_high_df = drop_high_df or set()
    summary = str(event.get("summary_en") or "").strip()
    title = str(event.get("title") or "").strip()
    tokens = _si_tokenize_text(summary or title, title if summary else None)
    anchors = _si_anchor_terms(summary or title, title if summary else None)
    key = choose_key_tokens(tokens, drop_high_df=drop_high_df)
    return EventTokens(key_tokens=frozenset(key), anchor_tokens=frozenset(anchors))


def _tokenize_link(link: dict[str, Any], *, drop_high_df: set[str] | None = None) -> EventTokens:
    """Tokenize a link's title + description for retrieval matching."""
    drop_high_df = drop_high_df or set()
    title = str(link.get("title") or "").strip()
    desc = str(link.get("description") or "").strip()
    tokens = _si_tokenize_text(title, desc)
    anchors = _si_anchor_terms(title, desc)
    key = choose_key_tokens(tokens, drop_high_df=drop_high_df)
    return EventTokens(key_tokens=frozenset(key), anchor_tokens=frozenset(anchors))


_SKIP_CLUSTER_REGEX_RULES: list[tuple[str, list[str]]] = [
    (
        "live_updates",
        [
            r"\blive updates?\b",
            r"\blatest updates?\b",
            r"\bliveblog\b",
            r"\bas it happened\b",
            r"\brolling coverage\b",
            r"\bminute by minute\b",
            r"\bbreaking news\b",
            r"即時",
            r"持續更新",
            r"最新",
        ],
    ),
    (
        "multi_topic",
        [
            r"\btop stories\b",
            r"\bthings to watch\b",
            r"\bweek in review\b",
            r"\bweek ahead\b",
            r"\bmorning briefing\b",
            r"\bevening briefing\b",
            r"\bdaily briefing\b",
            r"\btransfer gossip\b",
            r"懶人包",
            r"盤點",
        ],
    ),
    (
        "roundup",
        [
            r"\broundup\b",
            r"\brecap\b",
            r"\bhighlights\b",
            r"\bwhat we know\b",
            r"\beverything you need to know\b",
            r"\bexplainer\b",
            r"\bexplained\b",
            r"\bq&a\b",
        ],
    ),
    (
        "opinion",
        [
            r"\bopinion\b",
            r"\bcommentary\b",
            r"\beditorial\b",
            r"評論",
        ],
    ),
    (
        "analysis",
        [
            r"\banalysis\b",
            r"\banalyst\b",
            r"\bin depth\b",
            r"分析",
        ],
    ),
]

_SKIP_CLUSTER_REGEXES: list[tuple[str, re.Pattern[str]]] = [
    (reason, re.compile("|".join(patterns), re.IGNORECASE))
    for reason, patterns in _SKIP_CLUSTER_REGEX_RULES
]



_NOISY_DOMAINS = {
    # Aggregators / social shorteners tend to produce noisy titles and mixed-topic pages.
    "dailymail.co.uk",
    "mailonline.co.uk",
    "news.google.com",
    "t.co",
    "x.com",
    "twitter.com",
}


def _skip_cluster_reason(link: dict[str, Any]) -> str | None:
    """Return a skip-clustering reason if this link looks multi-topic/roundup-ish."""
    title = str(link.get("title") or "")
    desc = str(link.get("description") or "")
    text = f"{title}\n{desc}".strip()
    if text:
        for reason, rx in _SKIP_CLUSTER_REGEXES:
            if rx.search(text):
                return reason

    return None


def _is_noisy_or_roundup_link(link: dict[str, Any]) -> bool:
    if _skip_cluster_reason(link) is not None:
        return True

    domain = link.get("domain")
    if isinstance(domain, str) and domain.strip():
        d = domain.strip().lower()
        if d.startswith("www."):
            d = d[4:]
        if d in _NOISY_DOMAINS:
            return True
    return False


def _link_published_ts(link: dict[str, Any]) -> int | None:
    for k in ("published_at_ts", "published_ts"):
        v = link.get(k)
        if isinstance(v, int) and v > 0:
            return v
    v = link.get("first_seen_ts")
    if isinstance(v, int) and v > 0:
        return v
    return None


def _event_best_ts(ev: dict[str, Any]) -> int | None:
    for k in ("best_published_ts", "updated_at_ts", "created_at_ts"):
        v = ev.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return None


def _recency_bonus(link_ts: int | None, event_ts: int | None) -> float:
    if not link_ts or not event_ts:
        return 0.0
    delta = abs(int(link_ts) - int(event_ts))
    window = 72 * 3600
    if delta >= window:
        return 0.0
    # 0h -> +0.06, 72h -> +0.00
    return 0.06 * (1.0 - (float(delta) / float(window)))


def _adjusted_min_score(
    base_min_score: float,
    *,
    link_key_token_count: int,
    link_anchor_token_count: int,
    noisy_or_roundup: bool,
) -> float:
    bump = 0.0

    # Specificity: links with very few tokens/anchors are more ambiguous.
    if link_key_token_count <= 2:
        bump += 0.08
    elif link_key_token_count <= 4:
        bump += 0.04

    if link_anchor_token_count == 0:
        bump += 0.08
    elif link_anchor_token_count == 1:
        bump += 0.04

    if link_key_token_count <= 2 and link_anchor_token_count <= 1:
        bump += 0.04

    if noisy_or_roundup:
        bump += 0.03

    # Never reduce the caller's threshold. Cap to 1.0 so min_score remains meaningful.
    return min(1.0, float(base_min_score) + bump)


def retrieve_candidates(
    link: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    top_k: int = 5,
    min_score: float = 0.18,
    token_cache: dict[int, EventTokens] | None = None,
    drop_high_df: set[str] | None = None,
    drop_high_df_sig: str | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """Deterministic token/anchor similarity retrieval for candidate events.

    Returns up to *top_k* (event, score) pairs sorted by score descending,
    filtering those below *min_score*.  Cross-category, status-agnostic.
    """
    if not events:
        return []

    drop_high_df = drop_high_df or set()

    if token_cache is None:
        token_cache = {}

    if drop_high_df_sig is None:
        drop_high_df_sig = _drop_high_df_signature(drop_high_df)
    prev_sig = token_cache.get(_TOKEN_CACHE_DF_SIG_KEY)  # type: ignore[assignment]
    if prev_sig != drop_high_df_sig:
        token_cache.clear()
        token_cache[_TOKEN_CACHE_DF_SIG_KEY] = drop_high_df_sig  # type: ignore[index]

    link_tok = _tokenize_link(link, drop_high_df=drop_high_df)
    if not link_tok.key_tokens and not link_tok.anchor_tokens:
        return []

    noisy_or_roundup = _is_noisy_or_roundup_link(link)
    min_score_adj = _adjusted_min_score(
        min_score,
        link_key_token_count=len(link_tok.key_tokens),
        link_anchor_token_count=len(link_tok.anchor_tokens),
        noisy_or_roundup=noisy_or_roundup,
    )
    link_ts = _link_published_ts(link)

    scored: list[tuple[dict[str, Any], float]] = []
    for ev in events:
        eid = ev.get("id", 0)
        et: EventTokens | None = None
        if isinstance(eid, int) and eid > 0:
            if eid not in token_cache:
                token_cache[eid] = _tokenize_event(ev, drop_high_df=drop_high_df)
            et = token_cache.get(eid)
        if et is None:
            et = _tokenize_event(ev, drop_high_df=drop_high_df)

        key_sim = jaccard(link_tok.key_tokens, et.key_tokens)
        anchor_sim = jaccard(link_tok.anchor_tokens, et.anchor_tokens)

        score = 0.55 * key_sim + 0.45 * anchor_sim

        # Bonus for numeric/entity anchor overlap.
        anchor_overlap = frozenset()
        if link_tok.anchor_tokens and et.anchor_tokens:
            overlap = link_tok.anchor_tokens & et.anchor_tokens
            anchor_overlap = overlap
            numeric_overlap = sum(1 for t in overlap if any(c.isdigit() for c in t))
            entity_overlap = sum(
                1 for t in overlap
                if any("\u4e00" <= c <= "\u9fff" for c in t) or (t.isupper() and len(t) >= 2)
            )
            if numeric_overlap:
                score += 0.05 * numeric_overlap
            if entity_overlap:
                score += 0.05 * entity_overlap

        # Recency weighting: small bonus for events close in time to the link.
        score += _recency_bonus(link_ts, _event_best_ts(ev))

        # Noisy/roundup links are prone to false matches; require stronger anchors or a high score.
        if noisy_or_roundup and len(anchor_overlap) < 2 and score < 0.35:
            continue

        if score >= min_score_adj:
            scored.append((ev, score))

    scored.sort(
        key=lambda x: (
            -x[1],
            -(int(x[0].get("best_published_ts") or 0)),
            -(int(x[0].get("link_count") or 0)),
            int(x[0].get("id") or 0),
        )
    )
    if not scored:
        return []

    best = scored[0][1]
    dynamic_min = max(min_score_adj, best - 0.08)
    pruned = [it for it in scored if it[1] >= dynamic_min]
    return pruned[:top_k]


def build_focused_clustering_prompt(
    link: dict[str, Any],
    candidates: list[tuple[dict[str, Any], float]],
) -> str:
    """Build a focused clustering prompt showing only retrieved candidates.

    If *candidates* is empty, offers only NEW_EVENT.
    """
    title = str(link.get("title") or "").strip() or "(no title)"
    desc = str(link.get("description") or "").strip()

    link_section = f'Title: "{title}"'
    if desc:
        link_section += f'\nDescription: "{desc[:300]}"'

    categories_str = ", ".join(f'"{c}"' for c in CATEGORY_LIST)

    if not candidates:
        return f"""You are a news event classifier.

This is a news link:
{link_section}

No candidate events matched this link. Create a NEW EVENT.

Return STRICT JSON:
{{"action":"new_event",
  "confidence":<0.0-1.0>,
  "summary_en":"<one-sentence English summary>",
  "category":"<category>","jurisdiction":"<jurisdiction>",
  "link_flags":[],
  "match_basis":[]}}

Categories: {categories_str}
Jurisdictions: "US", "UK", "HK", "CN", "EU", "JP", "KR", "GLOBAL", or other 2-letter code.
Return STRICT JSON only, no explanation."""

    # Build candidate list with scores and children.
    cand_lines: list[str] = []
    for i, (ev, score) in enumerate(candidates, start=1):
        eid = ev["id"]
        cat = ev.get("category") or "?"
        jur = ev.get("jurisdiction") or "?"
        summary = ev.get("summary_en") or ev.get("title") or "?"
        status = ev.get("status") or "?"
        lc = ev.get("link_count", 0)
        cand_lines.append(
            f'E{i}. [id={eid}] "{summary}" | {cat} | {jur} | status={status} | {lc} links | score={score:.2f}'
        )

    cand_section = "\n".join(cand_lines)

    return f"""You are a news event classifier.

This is a news link:
{link_section}

These candidate events share significant vocabulary with this link:
{cand_section}

Choose ONE action:

A) ASSIGN to existing event (same incident, different source, no new material)
   → {{"action":"assign","event_id":<id>,
      "confidence":<0.0-1.0>,
      "summary_en":"<one-sentence English summary>",
      "category":"<category>","jurisdiction":"<jurisdiction>",
      "link_flags":[],
      "match_basis":["<entity|number|location|time|other>"]}}
   ASSIGN is likely correct if a candidate covers the same story from a different source.

B) NEW DEVELOPMENT of existing event (significant escalation, verdict, arrest, resignation)
   → {{"action":"development","parent_event_id":<id>,
      "confidence":<0.0-1.0>,
      "summary_en":"<one-sentence English summary>",
      "development":"<short label>",
      "category":"<category>","jurisdiction":"<jurisdiction>",
      "link_flags":[],
      "match_basis":["<entity|number|location|time|other>"]}}
   DEVELOPMENT = new phase only. Same facts from different source = ASSIGN, not DEVELOPMENT.

C) NEW EVENT (completely new story, none of the candidates match)
   → {{"action":"new_event",
      "confidence":<0.0-1.0>,
      "summary_en":"<one-sentence English summary>",
      "category":"<category>","jurisdiction":"<jurisdiction>",
      "link_flags":[],
      "match_basis":[]}}

Categories: {categories_str}
Jurisdictions: "US", "UK", "HK", "CN", "EU", "JP", "KR", "GLOBAL", or other 2-letter code.

Rules:
- Same event = same incident/announcement, different source or angle → A
- Significant new development (result, escalation, arrest, resignation) → B
- Completely unrelated story → C
- Match across languages (Chinese headline about same event = same event)
- Only use ASSIGN if a matching event exists in the candidates above
- Always include: confidence (0.0-1.0), summary_en, category, jurisdiction, link_flags (list), match_basis (list; can be empty)
- link_flags may include: "roundup", "opinion", "live_updates", "multi_topic"
- If you are NOT at least 70% sure for ASSIGN or DEVELOPMENT, choose NEW EVENT instead
- If the link is a roundup/briefing, opinion, live updates, or clearly multi-topic, set link_flags accordingly and choose NEW EVENT

Return STRICT JSON only, no explanation."""


def build_clustering_prompt(
    link: dict[str, Any],
    fresh_events: list[dict[str, Any]],
) -> str:
    """Build a one-link-per-prompt clustering prompt for Gemini."""
    title = str(link.get("title") or "").strip() or "(no title)"
    desc = str(link.get("description") or "").strip()

    link_section = f'Title: "{title}"'
    if desc:
        link_section += f'\nDescription: "{desc[:300]}"'

    # Build event list with parent-child structure.
    events_section_lines: list[str] = []
    if fresh_events:
        # Group children under parents.
        parent_map: dict[int, list[dict[str, Any]]] = {}
        root_events: list[dict[str, Any]] = []
        for ev in fresh_events:
            pid = ev.get("parent_event_id")
            if pid is not None:
                parent_map.setdefault(pid, []).append(ev)
            else:
                root_events.append(ev)

        for i, ev in enumerate(root_events[:50], start=1):
            eid = ev["id"]
            cat = ev.get("category") or "?"
            jur = ev.get("jurisdiction") or "?"
            summary = ev.get("summary_en") or ev.get("title") or "?"
            status = ev.get("status") or "?"
            events_section_lines.append(
                f'E{i}. [id={eid}] "{summary}" | {cat} | {jur} | status={status}'
            )
            # Show children (developments).
            children = parent_map.get(eid, [])
            for child in children[:5]:
                cid = child["id"]
                dev = child.get("development") or child.get("summary_en") or "?"
                events_section_lines.append(
                    f'  └─ [id={cid}] "{dev}" (development)'
                )

    events_section = "\n".join(events_section_lines) if events_section_lines else "(no fresh events)"

    categories_str = ", ".join(f'"{c}"' for c in CATEGORY_LIST)

    prompt = f"""You are a news event classifier.

This is a news link:
{link_section}

Here are the current FRESH EVENTS (last 48h):
{events_section}

Choose ONE action:

A) ASSIGN to existing event (same incident, different source, no new material)
   → {{"action":"assign","event_id":<id>,
      "confidence":<0.0-1.0>,
      "summary_en":"<one-sentence English summary>",
      "category":"<category>","jurisdiction":"<jurisdiction>",
      "link_flags":[],
      "match_basis":["<entity|number|location|time|other>"]}}

B) NEW DEVELOPMENT of existing event (significant escalation, result, reversal)
   → {{"action":"development","parent_event_id":<id>,
      "confidence":<0.0-1.0>,
      "summary_en":"<one-sentence English summary>",
      "development":"<short label>",
      "category":"<category>","jurisdiction":"<jurisdiction>",
      "link_flags":[],
      "match_basis":["<entity|number|location|time|other>"]}}

C) NEW EVENT (completely new story)
   → {{"action":"new_event",
      "confidence":<0.0-1.0>,
      "summary_en":"<one-sentence English summary>",
      "category":"<category>","jurisdiction":"<jurisdiction>",
      "link_flags":[],
      "match_basis":[]}}

Categories: {categories_str}
Jurisdictions: "US", "UK", "HK", "CN", "EU", "JP", "KR", "GLOBAL", or other 2-letter code.

Rules:
- Same event = same incident/announcement, different source or angle → A
- Significant new development (result, escalation, arrest, resignation) → B
- Completely unrelated story → C
- Match across languages (Chinese headline about same event = same event)
- Only use ASSIGN if a matching event exists in the list above
- Do NOT assign to events with status=posted unless it's truly the same event
- Always include: confidence (0.0-1.0), summary_en, category, jurisdiction, link_flags (list), match_basis (list; can be empty)
- link_flags may include: "roundup", "opinion", "live_updates", "multi_topic"
- If you are NOT at least 70% sure for ASSIGN or DEVELOPMENT, choose NEW EVENT instead
- If the link is a roundup/briefing, opinion, live updates, or clearly multi-topic, set link_flags accordingly and choose NEW EVENT

Return STRICT JSON only, no explanation."""

    return prompt


def parse_clustering_response(
    response: dict[str, Any],
    link: dict[str, Any],
    fresh_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate and normalize LLM clustering response.

    Returns a dict containing:
    - validated: the model's validated action (no policy overrides)
    - enforced: the action after deterministic policy enforcement

    Returns None if the response is structurally invalid.
    """
    def _parse_confidence(v: Any) -> float | None:
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            f = float(v)
        elif isinstance(v, str) and v.strip():
            try:
                f = float(v.strip())
            except ValueError:
                return None
        else:
            return None
        if 0.0 <= f <= 1.0:
            return f
        return None

    def _parse_str_list(v: Any) -> list[str]:
        if v is None:
            return []
        items: list[str]
        if isinstance(v, list):
            items = [str(x) for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
        elif isinstance(v, str) and v.strip():
            items = [v.strip()]
        else:
            return []

        out: list[str] = []
        seen: set[str] = set()
        for s in items:
            t = str(s).strip()
            if not t:
                continue
            low = t.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(low)
        return out

    def _parse_link_flags(v: Any) -> list[str]:
        flags = _parse_str_list(v)
        # Keep only allowed flags and preserve the returned order.
        out: list[str] = []
        seen: set[str] = set()
        for f in flags:
            if f not in _LINK_FLAGS_ALLOWED:
                continue
            if f in seen:
                continue
            seen.add(f)
            out.append(f)
        return out

    action = str(response.get("action") or "").strip().lower()
    confidence = _parse_confidence(response.get("confidence"))
    confidence_missing = confidence is None
    if confidence is None:
        logger.warning("clustering: missing/invalid confidence, treating as 0.0")
        confidence = 0.0

    match_basis = _parse_str_list(response.get("match_basis"))
    link_flags = _parse_link_flags(response.get("link_flags"))

    raw_category = str(response.get("category") or "").strip() or None
    category = _normalise_category(raw_category)
    jurisdiction = str(response.get("jurisdiction") or "").strip() or None

    summary_en = str(response.get("summary_en") or "").strip()
    if not summary_en:
        # Deterministic fallback so we can still create an event when policy overrides trigger.
        title = str(link.get("title") or "").strip()
        if title:
            summary_en = title
        else:
            desc = str(link.get("description") or "").strip()
            summary_en = desc[:200] if desc else ""

    valid_ids = {ev["id"] for ev in fresh_events if isinstance(ev.get("id"), int)}

    validated: dict[str, Any]

    if action == "assign":
        event_id = response.get("event_id")
        if not isinstance(event_id, int):
            logger.warning("assign: event_id not int: %s", response)
            return None
        if event_id not in valid_ids:
            logger.warning("assign: event_id %d not in candidate set", event_id)
            return None
        validated = {
            "action": "assign",
            "event_id": event_id,
            "confidence": confidence,
            "match_basis": match_basis,
            "link_flags": link_flags,
            "summary_en": summary_en,
            "category": category,
            "jurisdiction": jurisdiction,
        }

    elif action == "development":
        parent_id = response.get("parent_event_id")
        if not isinstance(parent_id, int):
            logger.warning("development: parent_event_id not int: %s", response)
            return None
        if parent_id not in valid_ids:
            logger.warning("development: parent_event_id %d not in candidate set", parent_id)
            return None
        if not summary_en:
            logger.warning("development: empty summary_en")
            return None
        validated = {
            "action": "development",
            "parent_event_id": parent_id,
            "summary_en": summary_en,
            "development": str(response.get("development") or "").strip() or None,
            "category": category,
            "jurisdiction": jurisdiction,
            "confidence": confidence,
            "match_basis": match_basis,
            "link_flags": link_flags,
        }

    elif action in ("new_event", "new"):
        if not summary_en:
            logger.warning("new_event: empty summary_en")
            return None
        validated = {
            "action": "new_event",
            "summary_en": summary_en,
            "category": category,
            "jurisdiction": jurisdiction,
            "confidence": confidence,
            "match_basis": match_basis,
            "link_flags": link_flags,
        }

    else:
        logger.warning("Unknown clustering action: %s", action)
        return None

    enforced = dict(validated)
    override_reasons: list[str] = []

    if validated["action"] in ("assign", "development"):
        if confidence_missing:
            override_reasons.append("missing_confidence")

        if float(confidence) < _ASSIGNMENT_MIN_CONFIDENCE:
            override_reasons.append(f"low_confidence:{float(confidence):.2f}")

        for f in sorted(set(link_flags) & _LINK_FLAGS_FORCE_NEW_EVENT):
            override_reasons.append(f"link_flag:{f}")

    if override_reasons:
        logger.info(
            "Clustering policy override: %s -> new_event (confidence=%.2f flags=%s reasons=%s)",
            validated["action"],
            float(confidence),
            ",".join(link_flags) if link_flags else "-",
            ",".join(override_reasons),
        )
        enforced = {
            "action": "new_event",
            "summary_en": str(validated.get("summary_en") or "").strip(),
            "category": validated.get("category"),
            "jurisdiction": validated.get("jurisdiction"),
            "confidence": confidence,
            "match_basis": match_basis,
            "link_flags": link_flags,
            "enforcement": {
                "original_action": str(validated.get("action") or ""),
                "reasons": override_reasons,
            },
        }

    return {"validated": validated, "enforced": enforced}


def cluster_link(
    *,
    link: dict[str, Any],
    all_events: list[dict[str, Any]] | None = None,
    fresh_events: list[dict[str, Any]] | None = None,
    gemini: Any,
    db: Any,
    token_cache: dict[int, EventTokens] | None = None,
    drop_high_df: set[str] | None = None,
    drop_high_df_sig: str | None = None,
) -> dict[str, Any] | None:
    """Classify a single link using retrieve-then-decide.

    Uses *all_events* (preferred) or *fresh_events* (backward compat alias).
    Returns the action result dict or None on failure.
    """
    link_id = link.get("id")
    if not isinstance(link_id, int):
        logger.warning("link missing id")
        return None

    # Gate: deterministically skip clustering for roundup/multi-topic links so they
    # do not poison event clusters.
    skip_reason = _skip_cluster_reason(link)
    if skip_reason:
        mark_fn = getattr(db, "mark_link_skip_cluster", None)
        if callable(mark_fn):
            try:
                mark_fn(link_id=link_id, reason=skip_reason)
            except Exception:
                logger.warning("Failed to persist skip_cluster for link_id=%d", link_id, exc_info=True)
        logger.info("Skipping clustering for link %d: %s", link_id, skip_reason)
        return {"action": "skip", "skip_reason": skip_reason}

    events = all_events if all_events is not None else (fresh_events or [])

    # Stage 1: Retrieve candidates (deterministic).
    candidates = retrieve_candidates(
        link,
        events,
        top_k=5,
        token_cache=token_cache,
        drop_high_df=drop_high_df,
        drop_high_df_sig=drop_high_df_sig,
    )

    # Stage 2: Focused LLM prompt.
    prompt = build_focused_clustering_prompt(link, candidates)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    candidate_pairs: list[tuple[int, float]] = []
    for ev, score in candidates:
        try:
            candidate_pairs.append((int(ev["id"]), float(score)))
        except Exception:
            continue

    def _pick_model_name(obj: Any) -> str | None:
        for attr in ("last_model_name", "last_model", "model_name", "model"):
            v = getattr(obj, attr, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _log_decision(
        *,
        llm_response: Any | None,
        validated_action: dict[str, Any] | None,
        enforced_action: dict[str, Any] | None,
        llm_started_at_ts: int | None,
        llm_finished_at_ts: int | None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        # Logging is best-effort only; it must never change clustering behaviour.
        insert_fn = getattr(db, "insert_clustering_decision", None)
        if not callable(insert_fn):
            return
        try:
            insert_fn(
                link_id=link_id,
                candidates=candidate_pairs,
                prompt_sha256=prompt_sha256,
                model_name=_pick_model_name(gemini),
                llm_response=llm_response,
                validated_action=validated_action,
                enforced_action=enforced_action,
                llm_started_at_ts=llm_started_at_ts,
                llm_finished_at_ts=llm_finished_at_ts,
                error_type=error_type,
                error_message=error_message,
            )
        except Exception:
            logger.debug("Failed to log clustering decision for link_id=%d", link_id, exc_info=True)

    llm_started_at_ts = int(time.time())
    try:
        response = gemini.generate_json(prompt)
    except Exception as e:
        llm_finished_at_ts = int(time.time())
        _log_decision(
            llm_response=None,
            validated_action=None,
            enforced_action=None,
            llm_started_at_ts=llm_started_at_ts,
            llm_finished_at_ts=llm_finished_at_ts,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        logger.warning("Gemini call failed for link %s: %s", link.get("norm_url", "?"), e)
        return None
    llm_finished_at_ts = int(time.time())

    if not response:
        _log_decision(
            llm_response=response,
            validated_action=None,
            enforced_action=None,
            llm_started_at_ts=llm_started_at_ts,
            llm_finished_at_ts=llm_finished_at_ts,
        )
        logger.warning("Empty Gemini response for link %s", link.get("norm_url", "?"))
        return None

    # Validate against candidate events (not full list), then enforce policy.
    candidate_events = [ev for ev, _score in candidates]
    parsed = parse_clustering_response(response, link, candidate_events)
    if not parsed:
        _log_decision(
            llm_response=response,
            validated_action=None,
            enforced_action=None,
            llm_started_at_ts=llm_started_at_ts,
            llm_finished_at_ts=llm_finished_at_ts,
        )
        return None
    validated_action = parsed.get("validated")
    enforced_action = parsed.get("enforced")
    if not isinstance(validated_action, dict) or not isinstance(enforced_action, dict):
        _log_decision(
            llm_response=response,
            validated_action=None,
            enforced_action=None,
            llm_started_at_ts=llm_started_at_ts,
            llm_finished_at_ts=llm_finished_at_ts,
        )
        logger.warning("Invalid parsed clustering response shape: %s", type(parsed))
        return None

    validated_action_obj = dict(validated_action)
    enforced: dict[str, Any] = dict(enforced_action)

    link_title = str(link.get("title") or "").strip() or None
    link_url = str(link.get("url") or "").strip() or None

    action = str(enforced.get("action") or "").strip().lower()

    if action == "assign":
        event_id = enforced["event_id"]
        assigned_to_posted = False
        for ev in candidate_events:
            if ev["id"] == event_id and ev.get("status") == "posted":
                assigned_to_posted = True
                break
        db.assign_link_to_event(link_id=link_id, event_id=event_id)
        enforced["assigned_to_posted"] = assigned_to_posted
        logger.info("Assigned link %d to event %d%s", link_id, event_id, " (posted)" if assigned_to_posted else "")
        _log_decision(
            llm_response=response,
            validated_action=validated_action_obj,
            enforced_action=dict(enforced),
            llm_started_at_ts=llm_started_at_ts,
            llm_finished_at_ts=llm_finished_at_ts,
        )
        return enforced

    if action == "development":
        parent_id = enforced["parent_event_id"]
        model_name = _pick_model_name(gemini) or "gemini-flash"
        event_id = db.create_event(
            summary_en=enforced["summary_en"],
            category=enforced.get("category"),
            jurisdiction=enforced.get("jurisdiction"),
            title=link_title,
            primary_url=link_url,
            parent_event_id=parent_id,
            development=enforced.get("development"),
            model=model_name,
        )
        db.assign_link_to_event(link_id=link_id, event_id=event_id)
        enforced["event_id"] = event_id
        logger.info("Created development event %d (parent=%d) for link %d", event_id, parent_id, link_id)
        _log_decision(
            llm_response=response,
            validated_action=validated_action_obj,
            enforced_action=dict(enforced),
            llm_started_at_ts=llm_started_at_ts,
            llm_finished_at_ts=llm_finished_at_ts,
        )
        return enforced

    if action == "new_event":
        model_name = _pick_model_name(gemini) or "gemini-flash"
        event_id = db.create_event(
            summary_en=enforced["summary_en"],
            category=enforced.get("category"),
            jurisdiction=enforced.get("jurisdiction"),
            title=link_title,
            primary_url=link_url,
            model=model_name,
        )
        db.assign_link_to_event(link_id=link_id, event_id=event_id)
        enforced["event_id"] = event_id
        logger.info("Created new event %d for link %d", event_id, link_id)
        _log_decision(
            llm_response=response,
            validated_action=validated_action_obj,
            enforced_action=dict(enforced),
            llm_started_at_ts=llm_started_at_ts,
            llm_finished_at_ts=llm_finished_at_ts,
        )
        return enforced

    _log_decision(
        llm_response=response,
        validated_action=validated_action_obj,
        enforced_action=dict(enforced),
        llm_started_at_ts=llm_started_at_ts,
        llm_finished_at_ts=llm_finished_at_ts,
    )
    return None


def build_merge_prompt(events: list[dict[str, Any]], category: str) -> str:
    """Build a prompt asking the LLM to group duplicate events within a category."""
    lines: list[str] = []
    for ev in events:
        eid = ev["id"]
        summary = ev.get("summary_en") or ev.get("title") or "?"
        lc = ev.get("link_count", 0)
        jur = ev.get("jurisdiction") or "?"
        lines.append(f'{eid}. "{summary}" | {lc} links | {jur}')

    events_block = "\n".join(lines)

    return f"""You are a news deduplication engine.

Below are {len(events)} events from the "{category}" category.
Group events that cover the SAME underlying story (same incident, same announcement, same match).

Events:
{events_block}

Rules:
- SAME story = same specific incident/announcement reported by different sources → group them
- Do NOT merge: related-but-distinct stories, parent vs child developments, or different matches/incidents
- Only group events you are confident are duplicates

Return STRICT JSON with plain integer IDs (the number before each event's period):
{{"groups": [[42, 55], [78, 82]], "no_merge": [90, 91]}}

IMPORTANT: Use plain integers only. NOT strings, NOT "E42", just 42.
- "groups": list of lists, each inner list has 2+ event IDs that should be merged
- "no_merge": list of event IDs that should NOT be merged with anything
- Every event ID must appear exactly once across groups + no_merge"""


def parse_merge_response(
    response: dict[str, Any] | None,
    valid_ids: set[int],
) -> list[list[int]] | None:
    """Validate LLM merge response. Returns list of merge groups or None on failure."""
    if not response or not isinstance(response, dict):
        return None

    groups_raw = response.get("groups")
    if not isinstance(groups_raw, list):
        return None

    seen: set[int] = set()
    valid_groups: list[list[int]] = []

    for group in groups_raw:
        if not isinstance(group, list):
            continue
        # Filter to valid integer IDs not yet seen.
        clean: list[int] = []
        for item in group:
            if isinstance(item, int) and item in valid_ids and item not in seen:
                clean.append(item)
                seen.add(item)
        # Only keep groups with 2+ members.
        if len(clean) >= 2:
            valid_groups.append(clean)

    return valid_groups if valid_groups else None


def _cross_category_pairs_by_retrieval(
    events: list[dict[str, Any]],
    *,
    token_cache: dict[int, EventTokens] | None = None,
    drop_high_df: set[str] | None = None,
    min_score: float = 0.25,
    top_k: int = 10,
    max_pairs: int = 300,
) -> list[tuple[int, int, float]]:
    """Return likely duplicate event pairs across different categories via retrieval.

    This mirrors the retrieve-then-decide approach in scripts/recluster_events.py,
    but restricts candidate generation to cross-category pairs only.
    """
    # Return cross-category event pairs for LLM confirmation, sorted by score desc.
    #
    # Args:
    #   drop_high_df: Optional high-document-frequency token drop set for precision.
    #   min_score: Minimum retrieval score for a pair to be included.
    #   top_k: Maximum candidates to keep per event query.
    #   max_pairs: Global cap on pair count to avoid excessive LLM calls.
    if token_cache is None:
        token_cache = {}

    drop_high_df = drop_high_df or set()
    drop_high_df_sig = _drop_high_df_signature(drop_high_df)

    pairs: list[tuple[int, int, float]] = []

    for i, ev_a in enumerate(events):
        aid = ev_a.get("id", 0)
        if not isinstance(aid, int) or aid <= 0:
            continue

        cat_a = ev_a.get("category") or ""
        summary = str(ev_a.get("summary_en") or ev_a.get("title") or "").strip()
        if not summary:
            continue

        title = str(ev_a.get("title") or "").strip()
        link_like: dict[str, Any] = {
            "title": summary,
            "description": title,
        }

        # Dedup pairs by only comparing against later events.
        others = [
            ev for ev in events[i + 1 :]
            if (ev.get("category") or "") != cat_a
        ]
        if not others:
            continue

        candidates = retrieve_candidates(
            link_like,
            others,
            top_k=top_k,
            min_score=min_score,
            token_cache=token_cache,
            drop_high_df=drop_high_df,
            drop_high_df_sig=drop_high_df_sig,
        )
        for ev_b, score in candidates:
            bid = ev_b.get("id", 0)
            if not isinstance(bid, int) or bid <= 0:
                continue
            pairs.append((aid, bid, float(score)))

        # Keep the list bounded for large pools.
        if max_pairs > 0 and len(pairs) >= max_pairs * 3:
            pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
            pairs = pairs[:max_pairs]

    pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs


def _execute_merge_group(
    group_events: list[dict[str, Any]],
    *,
    db: Any,
    stats: dict[str, int],
    merged_ids: set[int],
) -> None:
    """Pick winner, handle posted status transfer, merge losers into winner."""
    if len(group_events) < 2:
        return

    posted_events = [e for e in group_events if e.get("status") == "posted"]
    if len(posted_events) > 1:
        # Safety: do not merge posted events together, as that would delete an audit trail
        # (thread_id/run_id are keyed to the posted event row).
        logger.info(
            "Skipping merge group with multiple posted events: %s",
            [int(e.get("id") or 0) for e in posted_events],
        )
        return

    if len(posted_events) == 1:
        # Prefer preserving the posted event row for audit safety.
        winner = posted_events[0]
    else:
        winner = max(
            group_events,
            key=lambda e: (e.get("link_count", 0), -(e.get("created_at_ts", 0))),
        )
    loser_ids = [e["id"] for e in group_events if e["id"] != winner["id"]]

    # Handle posted status transfer.
    winner_posted = winner.get("status") == "posted"
    for lid in loser_ids:
        loser = next((e for e in group_events if e["id"] == lid), None)
        if loser and loser.get("status") == "posted" and not winner_posted:
            try:
                db.mark_event_posted(
                    winner["id"],
                    thread_id=loser.get("thread_id"),
                    run_id=loser.get("run_id"),
                )
                winner_posted = True
            except Exception:
                pass

    try:
        links_moved = db.merge_events_into(
            winner_id=winner["id"], loser_ids=loser_ids
        )
        stats["groups_merged"] += 1
        stats["events_removed"] += len(loser_ids)
        stats["links_moved"] += links_moved
        merged_ids.update(loser_ids)
        logger.info(
            "Merged %d events into %d (winner), %d links moved",
            len(loser_ids), winner["id"], links_moved,
        )
    except Exception as e:
        logger.warning("Merge DB error for group %s: %s", [e["id"] for e in group_events], e)
        stats["errors"] += 1


def merge_events(
    *,
    db: Any,
    gemini: Any,
    delay_seconds: float = 3.0,
    batch_size: int = 50,
    include_expired: bool = False,
    max_consecutive_failures: int = 3,
) -> dict[str, int]:
    """Post-clustering merge pass: compare event summaries within each category,
    merge duplicates using LLM.  Includes posted events and cross-category pairs.

    Args:
        include_expired: If True, merge across ALL events (not just fresh).
                         Useful for one-off dedup of the entire table.

    Returns stats dict.
    """
    stats = {
        "categories_processed": 0,
        "cross_category_pairs": 0,
        "llm_calls": 0,
        "groups_merged": 0,
        "events_removed": 0,
        "links_moved": 0,
        "errors": 0,
    }
    consecutive_failures = 0

    try:
        fresh_events = db.get_all_fresh_events(max_age_hours=168)
    except (TypeError, AttributeError):
        fresh_events = db.get_fresh_events(now_ts=0 if include_expired else None)
    # Filter to root events only (includes posted).
    root_events = [
        ev for ev in fresh_events
        if ev.get("parent_event_id") is None
    ]

    if not root_events:
        return stats

    # --- Cross-category merge pass ---
    token_cache: dict[int, EventTokens] = {}
    drop_high_df = _compute_drop_high_df(root_events)
    cross_pairs = _cross_category_pairs_by_retrieval(
        root_events,
        token_cache=token_cache,
        drop_high_df=drop_high_df,
        min_score=0.25,
        top_k=10,
        max_pairs=300,
    )
    stats["cross_category_pairs"] = len(cross_pairs)
    ev_lookup_all = {ev["id"]: ev for ev in root_events}
    merged_ids: set[int] = set()

    for aid, bid, _score in cross_pairs:
        if consecutive_failures >= max_consecutive_failures:
            logger.warning(
                "Merge early exit: %d consecutive LLM failures, skipping remaining cross-category pairs",
                consecutive_failures,
            )
            break

        if aid in merged_ids or bid in merged_ids:
            continue
        ev_a = ev_lookup_all.get(aid)
        ev_b = ev_lookup_all.get(bid)
        if not ev_a or not ev_b:
            continue

        label = f"cross-category ({ev_a.get('category', '?')} + {ev_b.get('category', '?')})"
        prompt = build_merge_prompt([ev_a, ev_b], label)

        try:
            raw_text = gemini.generate(prompt)
        except Exception as e:
            logger.warning("Cross-category merge LLM failed: %s", e)
            stats["errors"] += 1
            consecutive_failures += 1
            continue

        stats["llm_calls"] += 1
        if not raw_text:
            stats["errors"] += 1
            consecutive_failures += 1
            continue

        consecutive_failures = 0
        sanitized = re.sub(r'\bE(\d+)\b', r'\1', raw_text)
        try:
            response = json.loads(
                sanitized[sanitized.find("{"):sanitized.rfind("}") + 1]
            )
        except (json.JSONDecodeError, ValueError):
            stats["errors"] += 1
            consecutive_failures += 1
            continue

        valid_ids = {aid, bid}
        groups = parse_merge_response(response, valid_ids)
        if not groups:
            continue

        for group in groups:
            group_events = [ev_lookup_all[eid] for eid in group if eid in ev_lookup_all]
            _execute_merge_group(group_events, db=db, stats=stats, merged_ids=merged_ids)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    # --- Per-category merge pass ---
    # Group by category (exclude already-merged).
    by_category: dict[str, list[dict[str, Any]]] = {}
    for ev in root_events:
        if ev["id"] in merged_ids:
            continue
        cat = ev.get("category") or "Other"
        by_category.setdefault(cat, []).append(ev)

    for cat, cat_events in by_category.items():
        if consecutive_failures >= max_consecutive_failures:
            logger.warning(
                "Merge early exit: %d consecutive LLM failures, skipping remaining categories",
                consecutive_failures,
            )
            break

        if len(cat_events) < 2:
            continue

        stats["categories_processed"] += 1

        # Sort alphabetically so similar summaries land in the same chunk.
        cat_events.sort(key=lambda e: (e.get("summary_en") or "").lower())

        # Sliding window: overlap adjacent batches so boundary pairs get compared.
        overlap = min(10, batch_size // 3)
        stride = max(batch_size - overlap, 1)

        chunk_start = 0
        while chunk_start < len(cat_events):
            chunk = cat_events[chunk_start : chunk_start + batch_size]
            # Skip events already merged as losers in earlier windows.
            chunk = [ev for ev in chunk if ev["id"] not in merged_ids]
            if len(chunk) < 2:
                chunk_start += stride
                continue

            prompt = build_merge_prompt(chunk, cat)

            try:
                raw_text = gemini.generate(prompt)
            except Exception as e:
                logger.warning("Merge LLM call failed for category %s: %s", cat, e)
                stats["errors"] += 1
                consecutive_failures += 1
                chunk_start += stride
                continue

            stats["llm_calls"] += 1

            if not raw_text:
                logger.warning("Empty merge response for category %s", cat)
                stats["errors"] += 1
                consecutive_failures += 1
                chunk_start += stride
                continue

            consecutive_failures = 0

            # Sanitize: LLM sometimes returns E123 instead of 123.
            sanitized = re.sub(r'\bE(\d+)\b', r'\1', raw_text)
            try:
                response = json.loads(
                    sanitized[sanitized.find("{"):sanitized.rfind("}") + 1]
                )
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Failed to parse merge JSON for category %s: %s...",
                    cat, raw_text[:200],
                )
                stats["errors"] += 1
                chunk_start += stride
                continue

            valid_ids = {ev["id"] for ev in chunk}
            groups = parse_merge_response(response, valid_ids)
            if not groups:
                chunk_start += stride
                continue

            ev_lookup = {ev["id"]: ev for ev in chunk}
            for group in groups:
                group_events = [ev_lookup[eid] for eid in group if eid in ev_lookup]
                _execute_merge_group(group_events, db=db, stats=stats, merged_ids=merged_ids)

            if delay_seconds > 0:
                time.sleep(delay_seconds)

            chunk_start += stride

    logger.info(
        "Merge pass complete: %d categories, %d cross-cat pairs, %d LLM calls, %d groups merged, "
        "%d events removed, %d links moved, %d errors",
        stats["categories_processed"], stats["cross_category_pairs"],
        stats["llm_calls"], stats["groups_merged"],
        stats["events_removed"], stats["links_moved"], stats["errors"],
    )
    return stats


def cluster_all_pending(
    *,
    db: Any,
    gemini: Any,
    max_links: int = 100,
    delay_seconds: float = 3.0,
    max_consecutive_failures: int = 3,
) -> dict[str, int]:
    """Process all unassigned links sequentially, oldest first.

    Uses retrieve-then-decide: loads all events once, refreshes every 10 links,
    and maintains a shared token cache across iterations.

    Returns stats dict.
    """
    links = db.get_unassigned_links()
    if not links:
        return {"processed": 0, "assigned": 0, "new_events": 0, "developments": 0,
                "assigned_to_posted": 0, "errors": 0, "skipped": 0}

    links = links[:max_links]

    stats = {"processed": 0, "assigned": 0, "new_events": 0, "developments": 0,
             "assigned_to_posted": 0, "errors": 0, "skipped": 0}
    consecutive_failures = 0
    token_cache: dict[int, EventTokens] = {}

    # Load all events once (cross-category, includes posted).
    try:
        all_events = db.get_all_fresh_events(max_age_hours=168)
    except (TypeError, AttributeError):
        all_events = db.get_fresh_events()
    drop_high_df = _compute_drop_high_df(all_events)
    drop_high_df_sig = _drop_high_df_signature(drop_high_df)

    for i, link in enumerate(links):
        # Early exit: if Gemini is consistently failing, stop wasting quota.
        if consecutive_failures >= max_consecutive_failures:
            stats["skipped"] = len(links) - stats["processed"]
            logger.warning(
                "Early exit: %d consecutive Gemini failures, skipping remaining %d links",
                consecutive_failures, stats["skipped"],
            )
            break

        # Refresh events every 10 links to pick up newly created events.
        if i > 0 and i % 10 == 0:
            try:
                all_events = db.get_all_fresh_events(max_age_hours=168)
            except (TypeError, AttributeError):
                all_events = db.get_fresh_events()
            drop_high_df = _compute_drop_high_df(all_events)
            drop_high_df_sig = _drop_high_df_signature(drop_high_df)

        result = cluster_link(
            link=link, all_events=all_events, gemini=gemini, db=db,
            token_cache=token_cache,
            drop_high_df=drop_high_df,
            drop_high_df_sig=drop_high_df_sig,
        )
        stats["processed"] += 1

        if result is None:
            stats["errors"] += 1
            consecutive_failures += 1
        else:
            consecutive_failures = 0
            if result["action"] == "assign":
                stats["assigned"] += 1
                if result.get("assigned_to_posted"):
                    stats["assigned_to_posted"] += 1
            elif result["action"] == "new_event":
                stats["new_events"] += 1
                # Add newly created event to the list for subsequent links.
                new_eid = result.get("event_id")
                if new_eid:
                    try:
                        new_ev = db.get_event(new_eid)
                        if new_ev:
                            all_events.append(new_ev)
                    except Exception:
                        pass
            elif result["action"] == "development":
                stats["developments"] += 1

        # Rate limiting between Gemini calls.
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    logger.info(
        "Clustering complete: %d processed, %d assigned (%d to posted), %d new, %d dev, %d errors, %d skipped",
        stats["processed"], stats["assigned"], stats["assigned_to_posted"],
        stats["new_events"], stats["developments"], stats["errors"], stats["skipped"],
    )
    return stats
