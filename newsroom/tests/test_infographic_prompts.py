from __future__ import annotations


def test_infographic_prompts_use_traditional_chinese() -> None:
    # Keep this very lightweight: we only want to prevent regressions where the
    # runner accidentally instructs English section labels inside the infographic.
    from newsroom.runner import (
        _card_fallback_prefix,
        _card_final_prefix,
        _infographic_fallback_prefix,
        _infographic_final_prefix,
    )

    final = _infographic_final_prefix("2:3")
    assert "Traditional Chinese" in final
    assert "Do NOT use English" in final

    fallback = _infographic_fallback_prefix("2:3")
    assert "2:3" in fallback
    assert "發生咩事" in fallback
    assert "關鍵事實" in fallback
    assert "後續睇咩" in fallback
    assert "What happened" not in fallback

    card_final = _card_final_prefix("3:2")
    assert "Traditional Chinese" in card_final
    assert "Do NOT use English" in card_final
    card_fallback = _card_fallback_prefix("3:2")
    assert "3:2" in card_fallback
    assert "What happened" not in card_fallback
