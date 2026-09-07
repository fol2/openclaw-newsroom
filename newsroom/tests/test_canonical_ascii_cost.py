"""Keep the ASCII authority hot path cheap without relaxing Unicode validation."""
from __future__ import annotations

import json
import pytest
import newsroom.authority.canonical as canonical


@pytest.mark.parametrize("value", ("", "sha256:" + "a" * 64, "authority.projection", "x" * 65536, "".join(chr(i) for i in range(128))))
def test_ascii_validation_avoids_per_character_python_calls(monkeypatch, value):
    calls = []

    def observed_ord(character):
        calls.append(character)
        return ord(character)

    monkeypatch.setattr(canonical, "ord", observed_ord, raising=False)
    document = {"record": value, "n": 1, "ok": True, "missing": None}
    expected = json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    assert canonical.canonical_json_bytes(document) == expected
    assert calls == []


def test_all_surrogate_codepoints_are_still_rejected_at_the_same_path():
    for codepoint in range(0xD800, 0xE000):
        for value in (chr(codepoint), "ascii" + chr(codepoint)):
            with pytest.raises(canonical.CanonicalizationError, match=r"lone surrogate is unsupported at \$\.record\[0\]"):
                canonical.canonical_json_bytes({"record": [value]})


@pytest.mark.parametrize("value", ("香港新聞", "café", "🙂", "\ud7ff", "\ue000", "\U0010ffff", "a\u0301"))
def test_non_ascii_canonical_bytes_remain_identical(value):
    document = {"標題": value}
    expected = json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    assert canonical.canonical_json_bytes(document) == expected


def test_surrogate_key_and_explicit_utf16_pair_remain_rejected():
    with pytest.raises(canonical.CanonicalizationError, match=r"lone surrogate is unsupported at \$\.<key>"):
        canonical.canonical_json_bytes({"bad\ud800": "value"})
    with pytest.raises(canonical.CanonicalizationError, match="lone surrogate"):
        canonical.canonical_json_bytes("\ud83d\ude42")


def test_str_subclass_cannot_lie_about_ascii():
    class LyingString(str):
        def isascii(self):
            return True

    with pytest.raises(canonical.CanonicalizationError, match="lone surrogate"):
        canonical.canonical_json_bytes(LyingString("\ud800"))
    assert canonical.canonical_json_bytes(LyingString("香港")) == '"香港"'.encode()


@pytest.mark.parametrize("value", (1.0, float("nan"), float("inf"), 2**53, -(2**53), b"bytes", {1: "key"}))
def test_ascii_fast_path_does_not_relax_other_canonical_rules(value):
    with pytest.raises(canonical.CanonicalizationError):
        canonical.canonical_json_bytes(value)
