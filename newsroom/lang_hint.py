from __future__ import annotations

import unicodedata
from typing import Any, Literal

LangHint = Literal["en", "zh", "mixed"]

_VALID_LANG_HINTS: frozenset[str] = frozenset({"en", "zh", "mixed"})
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= code <= hi:
            return True
    return False


def _count_cjk_chars(text: str) -> int:
    return sum(1 for ch in (text or "") if _is_cjk_char(ch))


def _is_latin_letter(ch: str) -> bool:
    if not ch.isalpha():
        return False
    if ch.isascii():
        return True
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def _count_latin_letters(text: str) -> int:
    return sum(1 for ch in (text or "") if _is_latin_letter(ch))


def normalise_lang_hint(value: Any) -> LangHint | None:
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if s in _VALID_LANG_HINTS:
        return s  # type: ignore[return-value]
    return None


def detect_text_lang_hint(text: str | None) -> LangHint:
    cjk = _count_cjk_chars(text or "")
    latin = _count_latin_letters(text or "")

    if cjk == 0 and latin == 0:
        return "mixed"
    if cjk == 0:
        return "en"
    if latin == 0:
        return "zh"

    total = cjk + latin
    cjk_ratio = cjk / total
    latin_ratio = latin / total

    # "Mostly CJK" and "mostly Latin" with conservative thresholds.
    if cjk_ratio >= 0.72:
        return "zh"
    if latin_ratio >= 0.72:
        return "en"

    # Tie-breaker for longer strings where one script is still clearly dominant.
    if cjk >= 10 and cjk >= (latin * 1.2):
        return "zh"
    if latin >= 14 and latin >= (cjk * 1.2):
        return "en"

    return "mixed"


def detect_link_lang_hint(
    *,
    title: str | None,
    description: str | None,
    existing_hint: Any | None = None,
) -> LangHint:
    norm = normalise_lang_hint(existing_hint)
    if norm:
        return norm

    parts: list[str] = []
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    return detect_text_lang_hint(" ".join(parts))
