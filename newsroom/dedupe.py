from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ._util import count_cjk
from .story_index import anchor_terms, choose_key_tokens, jaccard, tokenize_text

# Keep module-private alias so call-sites stay unchanged.
_count_cjk = count_cjk


# --- Cross-lingual entity matching helpers ---
# Cantonese news titles almost always preserve English proper nouns unchanged
# (e.g. "Leicester 違反財務規則即時扣 6 分"). We exploit this to catch duplicates
# across languages where token-based overlap fails.

_CAPWORD_RE = re.compile(r"\b[A-Z][a-z]{3,}\b")

# Entities too generic for cross-lingual matching (common in many unrelated stories).
_XLANG_GENERIC = {
    "city", "council", "police", "court", "government", "minister", "party",
    "state", "national", "international", "world", "report", "breaking",
    "update", "latest", "live", "news", "says", "said", "premier", "league",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
}


def _is_english_title(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return False
    if _count_cjk(t) >= 4:
        return False
    ascii_letters = sum(1 for ch in t if ch.isascii() and ch.isalpha())
    return ascii_letters >= 10


def _is_cjk_title(title: str) -> bool:
    return _count_cjk(title or "") >= 4


def _extract_proper_nouns(english_title: str) -> list[str]:
    """Extract capitalized proper nouns (>= 4 chars, not generic) from English text."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _CAPWORD_RE.findall(english_title):
        if m.lower() in _XLANG_GENERIC or m.lower() in seen:
            continue
        seen.add(m.lower())
        out.append(m)
    return out


def cross_lingual_entity_duplicate(
    candidate_title: str,
    recent_titles: Iterable[str],
    *,
    cluster_terms: Iterable[str] | None = None,
) -> str | None:
    """Check if an English candidate duplicates a CJK recent title via shared proper nouns.

    Returns the matched recent title if duplicate, else None.

    To reduce false positives the matched proper noun must also appear in the
    candidate's ``cluster_terms`` (confirming it's a central entity, not a passing
    mention). If ``cluster_terms`` is not provided, the check is skipped and any
    proper noun match is accepted.
    """
    if not _is_english_title(candidate_title):
        return None

    nouns = _extract_proper_nouns(candidate_title)
    if not nouns:
        return None

    cluster_lower = None
    if cluster_terms is not None:
        cluster_lower = {str(t).lower() for t in cluster_terms if isinstance(t, str) and str(t).strip()}

    for rt in recent_titles:
        if not rt or not _is_cjk_title(rt):
            continue
        for noun in nouns:
            # Short names (Apple, Trump, 5 chars) appear in many unrelated stories
            # within a 24h window. Require >= 6 chars for cross-lingual confidence.
            # The runner's LLM-based title translation still backstops shorter names.
            if len(noun) < 6:
                continue
            if noun not in rt:
                continue
            # Noun found in both English title and CJK title.
            # If cluster_terms provided, require the noun to be a core cluster term.
            if cluster_lower is not None and noun.lower() not in cluster_lower:
                continue
            return rt

    return None


@dataclass(frozen=True)
class TitleFeatures:
    key_tokens: set[str]
    anchor_tokens: set[str]


@dataclass(frozen=True)
class DedupeMatch:
    is_duplicate: bool
    score: float
    key_jaccard: float
    anchor_jaccard: float
    anchor_overlap: list[str]


def title_features(title: str | None) -> TitleFeatures:
    """Compute deterministic title features for semantic dedupe.

    Notes:
    - This is intentionally lightweight (no LLM).
    - Uses the same tokenization primitives as clustering so results are stable.
    """
    t = (title or "").strip()
    toks = tokenize_text(t, None)
    key = choose_key_tokens(toks, drop_high_df=set())
    anchors = anchor_terms(t, None)
    return TitleFeatures(key_tokens=key, anchor_tokens=anchors)


def _count_numeric_overlap(overlap: Iterable[str]) -> int:
    n = 0
    for t in overlap:
        if any(ch.isdigit() for ch in t):
            n += 1
    return n


def semantic_match(a: TitleFeatures, b: TitleFeatures) -> DedupeMatch:
    """Return a match decision + score for two titles.

    Goal: catch "same event with different wording/framing" without relying on
    exact title/URL matches.
    """
    key_j = jaccard(a.key_tokens, b.key_tokens)
    anchor_j = jaccard(a.anchor_tokens, b.anchor_tokens)
    overlap = sorted(a.anchor_tokens.intersection(b.anchor_tokens))
    key_overlap = a.key_tokens.intersection(b.key_tokens)
    numeric_overlap = _count_numeric_overlap(overlap)
    entity_overlap = len([t for t in overlap if not any(ch.isdigit() for ch in t)])

    # Weighted similarity score with small boosts for shared numeric anchors.
    score = (0.65 * key_j) + (0.35 * anchor_j)
    if numeric_overlap >= 1:
        score += 0.15
    if numeric_overlap >= 2:
        score += 0.10
    if len(overlap) >= 3:
        score += 0.10
    score = float(max(0.0, min(1.0, score)))

    # Hard rules to reduce false negatives on short titles:
    # - numeric anchors are extremely strong signals.
    # - multiple anchor overlaps + a bit of lexical overlap is also strong.
    is_dup = False
    if numeric_overlap >= 1 and entity_overlap >= 1 and key_j >= 0.07:
        is_dup = True
    elif entity_overlap >= 2 and key_j >= 0.12:
        is_dup = True
    elif key_j >= 0.45:
        is_dup = True
    elif score >= 0.52:
        is_dup = True
    else:
        # Additional backstop: some Cantonese / CJK headlines share many key bigrams
        # but only overlap on 1 strong anchor token (names/places), so the jaccard-based
        # rules above can miss obvious duplicates.
        #
        # We keep this conservative by requiring:
        # - reasonably high key overlap
        # - at least one *strong* anchor overlap (len>=3 CJK chunk, ticker, or number)
        # - a minimum number of overlapping key tokens (scaled for shorter titles)
        strong_anchor_overlap = []
        for t in overlap:
            if not t:
                continue
            if any(ch.isdigit() for ch in t) and len(t) >= 3:
                strong_anchor_overlap.append(t)
                continue
            if t.isupper() and 2 <= len(t) <= 6:
                strong_anchor_overlap.append(t)
                continue
            if any("\u4e00" <= ch <= "\u9fff" for ch in t) and len(t) >= 3:
                strong_anchor_overlap.append(t)
                continue
            if t.isascii() and len(t) >= 5:
                strong_anchor_overlap.append(t)
                continue

        min_key_overlap = 8
        if min(len(a.key_tokens), len(b.key_tokens)) <= 16:
            min_key_overlap = 5

        if key_j >= 0.34 and len(key_overlap) >= min_key_overlap and len(strong_anchor_overlap) >= 1:
            is_dup = True

    return DedupeMatch(
        is_duplicate=is_dup,
        score=score,
        key_jaccard=key_j,
        anchor_jaccard=anchor_j,
        anchor_overlap=overlap[:8],
    )


def best_semantic_duplicate(candidate_title: str, recent_titles: Iterable[str]) -> tuple[DedupeMatch | None, str | None]:
    """Return the best *duplicate* match if any, else the best overall match.

    Important: `semantic_match()` can mark a pair as duplicate even when the
    aggregate `score` is modest (e.g. CJK titles with many shared bigrams but
    only one strong anchor overlap). If we picked the best match purely by
    score, we could miss those duplicates.
    """
    cand = title_features(candidate_title)
    best_any: DedupeMatch | None = None
    best_any_title: str | None = None
    best_dup: DedupeMatch | None = None
    best_dup_title: str | None = None
    for t in recent_titles:
        if not t or not t.strip():
            continue
        m = semantic_match(cand, title_features(t))
        if best_any is None or m.score > best_any.score:
            best_any = m
            best_any_title = t
        if m.is_duplicate:
            if best_dup is None or m.score > best_dup.score:
                best_dup = m
                best_dup_title = t

    if best_dup is not None:
        return best_dup, best_dup_title
    return best_any, best_any_title
