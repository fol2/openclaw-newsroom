from __future__ import annotations

from newsroom.market_data import extract_market_symbols, normalize_hk_stooq_symbol, normalize_us_stooq_symbol


def test_normalize_hk_stooq_symbol_strips_leading_zeros() -> None:
    assert normalize_hk_stooq_symbol("0700.HK") == "700.hk"
    assert normalize_hk_stooq_symbol("0005.HK") == "5.hk"
    assert normalize_hk_stooq_symbol("5") == "5.hk"
    assert normalize_hk_stooq_symbol("00000.HK") is None


def test_normalize_us_stooq_symbol() -> None:
    assert normalize_us_stooq_symbol("AAPL") == "aapl.us"
    assert normalize_us_stooq_symbol("TSLA.US") == "tsla.us"
    assert normalize_us_stooq_symbol("BRK.B") is None


def test_extract_market_symbols_prefers_explicit_hk_suffix() -> None:
    syms = extract_market_symbols(
        category="US Stocks",
        story_title="港交所 (0388.HK) 宣布新措施",
        concrete_anchor="0388.HK",
        source_pack=None,
    )
    assert any(s.kind == "hk_stock" and s.symbol_provider == "388.hk" and s.symbol_display == "0388.HK" for s in syms)

