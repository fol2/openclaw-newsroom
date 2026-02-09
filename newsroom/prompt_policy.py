from __future__ import annotations


def prompt_id_for_category(category: str | None) -> str:
    """Return the script_posts worker prompt_id for a given story category.

    This must be deterministic and safe: unknown categories fall back to the
    default prompt rather than raising.
    """

    cat = str(category or "").strip()

    # Finance-ish.
    if cat in {"US Stocks", "Crypto", "Precious Metals"}:
        return "news_reporter_script_finance_v1"

    mapping = {
        # Default coverage.
        "Global News": "news_reporter_script_default_v1",
        "Politics": "news_reporter_script_politics_v1",
        # UK.
        "UK Parliament / Politics": "news_reporter_script_uk_parliament_v1",
        "UK News": "news_reporter_script_uk_news_v1",
        # HK.
        "Hong Kong News": "news_reporter_script_hk_news_v1",
        "Hong Kong Entertainment": "news_reporter_script_hk_entertainment_v1",
        # Other.
        "Entertainment": "news_reporter_script_entertainment_v1",
        "Sports": "news_reporter_script_sports_v1",
        "AI": "news_reporter_script_ai_v1",
    }
    return mapping.get(cat, "news_reporter_script_default_v1")
