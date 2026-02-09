"""Shared utilities for the newsroom package.

Centralises small helpers and constants that were previously copy-pasted
across validators, runner, dedupe, and result_repair modules.
"""

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Allowed failure types -- used by every validator to gate error_type values.
# ---------------------------------------------------------------------------

ALLOWED_FAILURE_TYPES: frozenset[str] = frozenset({
    "discord_429",
    "discord_403",
    "discord_error",
    "http_429",
    "http_503",
    "timeout",
    "paywall",
    "missing_data",
    "validation_failed",
    "unknown",
})


# ---------------------------------------------------------------------------
# ValidationResult -- canonical definition used by all validators.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


# ---------------------------------------------------------------------------
# Validator helper functions
# ---------------------------------------------------------------------------

def is_non_empty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        try:
            return int(v.strip())
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# CJK character counting
# ---------------------------------------------------------------------------

def count_cjk(text: str) -> int:
    """Count CJK characters in *text*.

    Ranges counted:
    - CJK Unified Ideographs Extension A  (U+3400 .. U+4DBF)
    - CJK Unified Ideographs              (U+4E00 .. U+9FFF)
    - CJK Symbols and Punctuation          (U+3000 .. U+303F)
    - Halfwidth and Fullwidth Forms         (U+FF00 .. U+FFEF)
    """
    total = 0
    for ch in (text or ""):
        o = ord(ch)
        if 0x3400 <= o <= 0x4DBF:
            total += 1
        elif 0x4E00 <= o <= 0x9FFF:
            total += 1
        elif 0x3000 <= o <= 0x303F:
            total += 1
        elif 0xFF00 <= o <= 0xFFEF:
            total += 1
    return total
