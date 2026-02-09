from __future__ import annotations

from newsroom.prompt_policy import prompt_id_for_category


def test_prompt_policy_finance() -> None:
    assert prompt_id_for_category("US Stocks") == "news_reporter_script_finance_v1"
    assert prompt_id_for_category("Crypto") == "news_reporter_script_finance_v1"
    assert prompt_id_for_category("Precious Metals") == "news_reporter_script_finance_v1"


def test_prompt_policy_mappings() -> None:
    assert prompt_id_for_category("UK Parliament / Politics") == "news_reporter_script_uk_parliament_v1"
    assert prompt_id_for_category("UK News") == "news_reporter_script_uk_news_v1"
    assert prompt_id_for_category("Hong Kong News") == "news_reporter_script_hk_news_v1"
    assert prompt_id_for_category("Hong Kong Entertainment") == "news_reporter_script_hk_entertainment_v1"
    assert prompt_id_for_category("Entertainment") == "news_reporter_script_entertainment_v1"
    assert prompt_id_for_category("Sports") == "news_reporter_script_sports_v1"
    assert prompt_id_for_category("AI") == "news_reporter_script_ai_v1"


def test_prompt_policy_default_fallback() -> None:
    assert prompt_id_for_category("Global News") == "news_reporter_script_default_v1"
    assert prompt_id_for_category("Politics") == "news_reporter_script_politics_v1"
    assert prompt_id_for_category(None) == "news_reporter_script_default_v1"
    assert prompt_id_for_category("Unknown Category") == "news_reporter_script_default_v1"
