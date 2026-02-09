from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
from dataclasses import dataclass
from typing import Any

import requests


_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
_TICKER_PAREN_RE = re.compile(r"\(([A-Z]{1,5})\)")
_HK_SUFFIX_RE = re.compile(r"\b(\d{1,5})\.HK\b", re.IGNORECASE)

_IGNORE_TICKERS = {
    "A",
    "AN",
    "AND",
    "AS",
    "AT",
    "BE",
    "BY",
    "FOR",
    "FROM",
    "IN",
    "INTO",
    "IS",
    "IT",
    "ITS",
    "OF",
    "ON",
    "OR",
    "OUT",
    "SAYS",
    "SAID",
    "THE",
    "TO",
    "WAS",
    "WERE",
    "WITH",
    # Common non-tickers in our feeds.
    "US",
    "UK",
    "EU",
    "UAE",
    "USD",
    "HK",
    "HKSAR",
    "AI",
    "PM",
    "PMQS",
    "PMQ",
}


def _now_utc() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


def _yyyymmdd(d: dt.date) -> str:
    return f"{d.year:04d}{d.month:02d}{d.day:02d}"


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s.upper() in {"N/D", "NA", "N/A", "NULL"}:
        return None
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    s = str(v).strip()
    if not s or s.upper() in {"N/D", "NA", "N/A", "NULL"}:
        return None
    try:
        return int(float(s.replace(",", "")))
    except Exception:
        return None


def normalize_hk_stooq_symbol(raw: str) -> str | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    m = _HK_SUFFIX_RE.search(raw)
    digits = m.group(1) if m else None
    if digits is None:
        # Accept bare numeric.
        m2 = re.search(r"\b(\d{1,5})\b", raw)
        digits = m2.group(1) if m2 else None
    if digits is None:
        return None
    try:
        n = int(digits)
    except Exception:
        return None
    if n <= 0:
        return None
    return f"{n}.hk"


def normalize_us_stooq_symbol(raw: str) -> str | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    # Drop .US suffix if present.
    raw = re.sub(r"\.US$", "", raw, flags=re.IGNORECASE).strip()
    if not raw or not raw.isascii():
        return None
    if not re.fullmatch(r"[A-Za-z]{1,5}", raw):
        return None
    return f"{raw.lower()}.us"


def _extract_text_for_symbols(story_title: str, concrete_anchor: str, source_pack: dict[str, Any] | None) -> str:
    parts = [str(story_title or "").strip(), str(concrete_anchor or "").strip()]
    if isinstance(source_pack, dict):
        sources = source_pack.get("sources")
        if isinstance(sources, list):
            for s in sources[:8]:
                if not isinstance(s, dict):
                    continue
                t = s.get("title")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
                u = s.get("final_url") or s.get("url")
                if isinstance(u, str) and u.strip():
                    parts.append(u.strip())
    return "\n".join([p for p in parts if p]).strip()


def _pick_us_ticker(text: str) -> str | None:
    if not text:
        return None
    # Prefer (TICKER) patterns.
    for t in _TICKER_PAREN_RE.findall(text):
        t = t.strip().upper()
        if not t or t in _IGNORE_TICKERS:
            continue
        return t
    tickers = []
    for t in _TICKER_RE.findall(text):
        t = t.strip().upper()
        if not t or t in _IGNORE_TICKERS:
            continue
        # Avoid single-letter noise.
        if len(t) == 1:
            continue
        tickers.append(t)
    # Deterministic: longest first, then alpha.
    tickers = sorted(set(tickers), key=lambda s: (-len(s), s))
    return tickers[0] if tickers else None


def _pick_hk_ticker(text: str) -> str | None:
    if not text:
        return None
    m = _HK_SUFFIX_RE.search(text)
    if m:
        return m.group(1).zfill(4) + ".HK"

    # If we see "HK" markers, allow bare numeric.
    low = text.lower()
    if any(k in low for k in ("hkex", "hksi", "hang seng", "hong kong", "港股", "港交所", "恒生")):
        m2 = re.search(r"\b(\d{3,4})\b", text)
        if m2:
            return m2.group(1).zfill(4) + ".HK"
    return None


def _pick_crypto(text: str) -> str | None:
    if not text:
        return None
    low = text.lower()
    # Prefer explicit tickers.
    if re.search(r"\bBTC\b", text):
        return "BTC"
    if re.search(r"\bETH\b", text):
        return "ETH"
    if "bitcoin" in low or "比特幣" in text:
        return "BTC"
    if "ethereum" in low or "以太" in text:
        return "ETH"
    return None


def _metals_from_text(text: str) -> list[str]:
    low = (text or "").lower()
    out: list[str] = []
    if any(k in low for k in ("gold",)) or any(k in text for k in ("黃金", "金價")):
        out.append("XAUUSD")
    if any(k in low for k in ("silver",)) or any(k in text for k in ("白銀", "銀價")):
        out.append("XAGUSD")
    if "platinum" in low or "鉑金" in text:
        out.append("XPTUSD")
    if "palladium" in low or "鈀" in text:
        out.append("XPDUSD")
    return list(dict.fromkeys(out))


@dataclass(frozen=True)
class MarketSymbol:
    kind: str  # us_stock|hk_stock|crypto|metal
    symbol_raw: str
    symbol_provider: str
    symbol_display: str


def extract_market_symbols(*, category: str, story_title: str, concrete_anchor: str, source_pack: dict[str, Any] | None) -> list[MarketSymbol]:
    cat = str(category or "").strip()
    text = _extract_text_for_symbols(story_title, concrete_anchor, source_pack)

    # Best-effort extraction: we may still want HK tickers in a "US Stocks" story
    # (or vice versa) if the ticker is explicitly present in the sources/title.
    us = _pick_us_ticker(text)
    hk = _pick_hk_ticker(text)
    crypto = _pick_crypto(text)
    metals = _metals_from_text(text)

    priorities: list[str] = []
    if cat == "Crypto":
        priorities = ["crypto", "us", "hk", "metals"]
    elif cat == "Precious Metals":
        priorities = ["metals", "us", "hk", "crypto"]
    else:
        priorities = ["us", "hk", "crypto", "metals"]

    symbols: list[MarketSymbol] = []
    for key in priorities:
        if key == "us" and us:
            prov = normalize_us_stooq_symbol(us)
            if prov:
                symbols.append(MarketSymbol(kind="us_stock", symbol_raw=us, symbol_provider=prov, symbol_display=us))
        elif key == "hk" and hk:
            prov = normalize_hk_stooq_symbol(hk)
            if prov:
                # Display as canonical 4-digit .HK.
                digits = normalize_hk_stooq_symbol(hk)
                # digits is like "700.hk" -> "0700.HK"
                disp = None
                if digits:
                    m = re.match(r"^(\d+)\.hk$", digits)
                    if m:
                        disp = m.group(1).zfill(4) + ".HK"
                symbols.append(MarketSymbol(kind="hk_stock", symbol_raw=hk, symbol_provider=prov, symbol_display=disp or hk))
        elif key == "crypto" and crypto:
            symbols.append(MarketSymbol(kind="crypto", symbol_raw=crypto, symbol_provider=f"{crypto}-USD", symbol_display=crypto))
        elif key == "metals":
            ms = metals or (["XAUUSD"] if cat == "Precious Metals" else [])
            for m in ms[:2]:
                symbols.append(MarketSymbol(kind="metal", symbol_raw=m, symbol_provider=m.lower(), symbol_display=m))

        if len(symbols) >= 2:
            break

    return symbols


def _stooq_quote_url(symbol: str) -> str:
    return f"https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv"


def _stooq_history_url(symbol: str, *, interval: str, d1: dt.date, d2: dt.date) -> str:
    return f"https://stooq.com/q/d/l/?s={symbol}&i={interval}&d1={_yyyymmdd(d1)}&d2={_yyyymmdd(d2)}"


def _fetch_text(session: requests.Session, url: str, *, timeout_seconds: int) -> str:
    resp = session.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    return resp.text


def _parse_stooq_quote_csv(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return None
    row = rows[0] or {}
    sym = str(row.get("Symbol") or "").strip() or None
    if not sym or sym.upper().endswith(",N/D"):
        return None
    date = str(row.get("Date") or "").strip() or None
    time = str(row.get("Time") or "").strip() or None
    return {
        "symbol": sym,
        "date": date,
        "time": time,
        "open": _safe_float(row.get("Open")),
        "high": _safe_float(row.get("High")),
        "low": _safe_float(row.get("Low")),
        "close": _safe_float(row.get("Close")),
        "volume": _safe_int(row.get("Volume")),
    }


def _parse_stooq_history_csv(text: str) -> list[dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return []
    if text.strip().startswith("No data"):
        return []
    rows = list(csv.DictReader(io.StringIO(text)))
    out: list[dict[str, Any]] = []
    for r in rows:
        d = str(r.get("Date") or "").strip()
        c = _safe_float(r.get("Close"))
        if not d or c is None:
            continue
        out.append({"date": d, "close": c})
    return out


def _coinbase_spot(session: requests.Session, product: str, *, timeout_seconds: int) -> float | None:
    url = f"https://api.coinbase.com/v2/prices/{product}/spot"
    resp = session.get(url, timeout=timeout_seconds)
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    try:
        amt = data.get("data", {}).get("amount")
    except Exception:
        amt = None
    return _safe_float(amt)


def _coinbase_candles(
    session: requests.Session,
    product: str,
    *,
    start: dt.datetime,
    end: dt.datetime,
    granularity_seconds: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    url = "https://api.exchange.coinbase.com/products/{product}/candles".format(product=product)
    params = {
        "granularity": int(granularity_seconds),
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
    }
    resp = session.get(url, params=params, timeout=timeout_seconds)
    if resp.status_code != 200:
        return []
    try:
        payload = resp.json()
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 6:
            continue
        t = _safe_int(row[0])
        close = _safe_float(row[4])
        if t is None or close is None:
            continue
        d = dt.datetime.fromtimestamp(int(t), tz=dt.UTC).date().isoformat()
        out.append({"date": d, "close": float(close)})
    # Coinbase returns most-recent first.
    out.sort(key=lambda r: r["date"])
    return out


def build_market_assets(
    *,
    category: str,
    story_title: str,
    concrete_anchor: str,
    source_pack: dict[str, Any] | None,
    run_time_uk: str | None,
    timeout_seconds: int = 6,
) -> dict[str, Any]:
    """Build a deterministic market snapshot for finance stories (0 LLM tokens)."""
    cat = str(category or "").strip()
    symbols = extract_market_symbols(category=cat, story_title=story_title, concrete_anchor=concrete_anchor, source_pack=source_pack)

    now = _now_utc()
    end_d = now.date()
    start_d = (end_d - dt.timedelta(days=140))

    out: dict[str, Any] = {
        "asof_utc": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "asof_uk": str(run_time_uk or "").strip() or None,
        "category": cat,
        "items": [],
        "series": None,
        "errors": [],
        "symbols": [s.symbol_raw for s in symbols],
    }

    if not symbols:
        return out

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (OpenClaw Newsroom)"})

    series_for_chart: list[dict[str, Any]] | None = None
    series_symbol: str | None = None

    for idx, sym in enumerate(symbols[:2]):
        try:
            if sym.kind in {"us_stock", "hk_stock", "metal"}:
                quote_txt = _fetch_text(session, _stooq_quote_url(sym.symbol_provider), timeout_seconds=timeout_seconds)
                quote = _parse_stooq_quote_csv(quote_txt)
                hist_txt = _fetch_text(
                    session,
                    _stooq_history_url(sym.symbol_provider, interval="d", d1=start_d, d2=end_d),
                    timeout_seconds=timeout_seconds,
                )
                hist = _parse_stooq_history_csv(hist_txt)
                last = hist[-1]["close"] if hist else None
                prev = hist[-2]["close"] if len(hist) >= 2 else None
                change_pct = None
                if isinstance(last, (int, float)) and isinstance(prev, (int, float)) and prev:
                    change_pct = ((float(last) - float(prev)) / float(prev)) * 100.0

                currency = "USD"
                if sym.kind == "hk_stock":
                    currency = "HKD"

                item = {
                    "kind": sym.kind,
                    "symbol_raw": sym.symbol_raw,
                    "symbol_provider": sym.symbol_provider,
                    "symbol_display": sym.symbol_display,
                    "price": (quote.get("close") if isinstance(quote, dict) else None) or last,
                    "currency": currency,
                    "change_1d_pct": change_pct,
                    "open": quote.get("open") if isinstance(quote, dict) else None,
                    "high": quote.get("high") if isinstance(quote, dict) else None,
                    "low": quote.get("low") if isinstance(quote, dict) else None,
                    "close": quote.get("close") if isinstance(quote, dict) else last,
                    "source": "stooq",
                }
                out["items"].append(item)

                if idx == 0 and hist:
                    series_for_chart = hist[-60:]
                    series_symbol = sym.symbol_display

            elif sym.kind == "crypto":
                product = sym.symbol_provider
                spot = _coinbase_spot(session, product, timeout_seconds=timeout_seconds)
                candles = _coinbase_candles(
                    session,
                    product,
                    start=now - dt.timedelta(days=70),
                    end=now,
                    granularity_seconds=86400,
                    timeout_seconds=timeout_seconds,
                )
                last = candles[-1]["close"] if candles else spot
                prev = candles[-2]["close"] if len(candles) >= 2 else None
                change_pct = None
                if isinstance(last, (int, float)) and isinstance(prev, (int, float)) and prev:
                    change_pct = ((float(last) - float(prev)) / float(prev)) * 100.0
                item = {
                    "kind": sym.kind,
                    "symbol_raw": sym.symbol_raw,
                    "symbol_provider": product,
                    "symbol_display": sym.symbol_display,
                    "price": spot or last,
                    "currency": "USD",
                    "change_1d_pct": change_pct,
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": last,
                    "source": "coinbase",
                }
                out["items"].append(item)
                if idx == 0 and candles:
                    series_for_chart = candles[-60:]
                    series_symbol = sym.symbol_display
        except Exception as e:
            out["errors"].append({"symbol": sym.symbol_raw, "error": str(e)[:200]})

    if series_for_chart and series_symbol:
        out["series"] = {"symbol_display": series_symbol, "points": series_for_chart, "window": "60d"}

    # Ensure JSON serializable.
    json.dumps(out, ensure_ascii=False)
    return out
