from __future__ import annotations

from newsroom.market_data import _parse_stooq_history_csv, _parse_stooq_quote_csv


def test_parse_stooq_quote_csv_basic() -> None:
    text = (
        "Symbol,Date,Time,Open,High,Low,Close,Volume\n"
        "aapl.us,2026-02-04,22:00:00,190.0,195.0,189.0,194.0,123456\n"
    )
    q = _parse_stooq_quote_csv(text)
    assert isinstance(q, dict)
    assert q["symbol"] == "aapl.us"
    assert q["close"] == 194.0
    assert q["volume"] == 123456


def test_parse_stooq_quote_csv_empty_or_no_rows() -> None:
    assert _parse_stooq_quote_csv("") is None
    assert _parse_stooq_quote_csv("Symbol,Date\n") is None


def test_parse_stooq_history_csv_basic() -> None:
    text = "Date,Open,High,Low,Close,Volume\n2026-02-01,1,2,0.5,1.5,100\n2026-02-02,1,2,0.5,1.7,120\n"
    rows = _parse_stooq_history_csv(text)
    assert rows == [{"date": "2026-02-01", "close": 1.5}, {"date": "2026-02-02", "close": 1.7}]


def test_parse_stooq_history_csv_no_data() -> None:
    assert _parse_stooq_history_csv("No data\n") == []

