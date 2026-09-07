"""Canonical fast validation must preserve bytes and exact diagnostic behaviour."""
from __future__ import annotations

import json
import random
from collections import UserDict, UserList

import pytest
from newsroom.authority import canonical


def _outcome(value, *, diagnostic=False):
    try:
        if diagnostic:
            canonical._validate_restricted_value_with_path(value)
            try:
                text = json.dumps(value, ensure_ascii=False, allow_nan=False,
                                  sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError, UnicodeError) as exc:
                raise canonical.CanonicalizationError(
                    f"canonical JSON encoding failed: {exc}"
                ) from exc
            result = text.encode("utf-8", errors="strict")
        else:
            result = canonical.canonical_json_bytes(value)
        return ("ok", result)
    except Exception as exc:
        return (type(exc).__name__, str(exc))


def test_canonical_fast_builtin_path_does_not_construct_error_paths(monkeypatch):
    calls = []
    original = canonical._validate_restricted_value_with_path

    def observed(value, path="$"):
        calls.append(path)
        return original(value, path)

    monkeypatch.setattr(canonical, "_validate_restricted_value_with_path", observed)
    value = {"香港": ["新聞🙂" * 4000, (0, True, None)], "ascii": "a" * 4000}
    assert canonical.canonical_json_bytes(value) == json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert calls == []


@pytest.mark.parametrize("value", [
    {"path": [1.5]}, {"path": [float("nan")]}, {"path": [float("inf")]},
    {"path": [2**53]}, {"path": [-(2**53)]}, {"path": [b"bytes"]},
    {"path": [{1: "wrong key"}]}, {"path": [set()]},
    {"path": [{"bad\udfff": "value"}]}, {"path": ["bad\ud800"]},
    {"path": ["\ud83d\ude42"]}, UserDict({"custom": "value"}),
    UserList([1, "custom"]), range(3), {"deep": [UserDict({1: True})]},
])
def test_canonical_fast_failure_and_custom_collection_parity(value):
    assert _outcome(value) == _outcome(value, diagnostic=True)


def test_canonical_fast_exhaustive_surrogate_parity():
    for codepoint in range(0xD800, 0xE000):
        for value in ({"record": [chr(codepoint)]}, {"香港" + chr(codepoint): 1}):
            assert _outcome(value) == _outcome(value, diagnostic=True)
            assert _outcome(value)[0] == "CanonicalizationError"


def test_canonical_fast_builtin_subclasses_use_existing_checks():
    class CustomString(str):
        def isascii(self):
            return True

    class CustomInt(int):
        pass

    for value in [CustomString("\ud800"), CustomString("香港"), CustomInt(2**53),
                  CustomInt(42), {CustomString("香港"): CustomString("text")}]:
        assert _outcome(value) == _outcome(value, diagnostic=True)


def test_canonical_fast_generated_nested_value_parity():
    rng = random.Random(895)
    scalars = [None, True, False, 0, -1, 2**53 - 1, -(2**53 - 1),
               "", "ascii", "香港🙂", "\x00\n\t", "e\u0301", "\U0010ffff",
               float("inf"), 0.1, 2**53, "bad\ud800", b"bytes"]

    def value(depth=0):
        choice = rng.randrange(5) if depth < 5 else 0
        if choice == 0:
            return rng.choice(scalars)
        if choice in (1, 2):
            items = [value(depth+1) for _ in range(rng.randrange(5))]
            return items if choice == 1 else tuple(items)
        return {str(i): value(depth+1) for i in range(rng.randrange(5))}

    for _ in range(2000):
        item = value()
        assert _outcome(item) == _outcome(item, diagnostic=True)


def test_canonical_fast_deep_and_cyclic_inputs_terminate():
    value = "valid"
    for _ in range(80):
        value = [value]
    assert _outcome(value) == _outcome(value, diagnostic=True)
    cyclic = []; cyclic.append(cyclic)
    with pytest.raises(RecursionError):
        canonical.canonical_json_bytes(cyclic)
