"""Historical data loaders for equities (yfinance) and crypto (Binance REST).

Primary interface:
    load_historical(ticker, start, end, interval, source) → polars DataFrame

Legacy interface (used by training pipeline):
    load_data(ticker, start, end) → pandas DataFrame  (daily, yfinance)

Binance REST pagination handles arbitrarily long date ranges automatically —
each request fetches up to 1 000 klines; requests repeat until the full range is covered.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import polars as pl
import requests
import yfinance as yf

# yfinance interval strings that map to our canonical interval labels
_YF_INTERVAL_MAP: dict[str, str] = {
    "1m":  "1m",
    "2m":  "2m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "60m",
    "4h":  "1h",   # yfinance has no 4h; caller should resample from 1h
    "1d":  "1d",
    "1wk": "1wk",
}

_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_BINANCE_MAX_LIMIT  = 1_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_historical(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    source: str = "auto",
) -> pl.DataFrame:
    """Fetch OHLCV history as a polars DataFrame.

    Args:
        ticker:   Equity symbol (e.g. "SPY") or Binance pair (e.g. "BTCUSDT").
        start:    Start date string, e.g. "2020-01-01".
        end:      End date string (inclusive). Defaults to today.
        interval: One of "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk".
        source:   "auto" | "yfinance" | "binance".
                  "auto" routes to Binance when ticker looks like a crypto pair
                  (ends with USDT, BTC, ETH, BNB, or BUSD).

    Returns:
        polars DataFrame with columns: timestamp (Datetime UTC), open, high, low, close, volume.
    """
    resolved = _resolve_source(ticker, source)
    if resolved == "binance":
        return _load_binance(ticker, interval, start, end)
    return _load_yfinance_polars(ticker, interval, start, end)


def load_macro_data(
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """B1: Fetch cross-asset macro features aligned to daily trading days.

    Downloads four secondary tickers and engineers five features:
      vix_pct         — VIX close as expanding percentile [0, 1] (high = fear)
      vix_ret         — VIX daily log return (positive = fear spike)
      credit_spread   — log(HYG / LQD) credit-risk proxy (negative = risk-off)
      credit_spread_ret — daily change in credit_spread
      dxy_ret         — UUP (USD ETF) log return (positive = USD strengthening)

    Missing data (different holiday calendars, stale prices) is forward-filled.
    If a ticker fails to download its features are quietly set to 0.
    """
    import warnings
    _TICKERS = ["^VIX", "HYG", "LQD", "UUP"]

    try:
        raw = yf.download(
            _TICKERS, start=start, end=end,
            auto_adjust=True, progress=False,
        )
    except Exception as exc:
        warnings.warn(f"load_macro_data: download failed ({exc})", stacklevel=2)
        return pd.DataFrame()

    def _close(ticker: str) -> pd.Series:
        """Extract the Close series for one ticker, handling MultiIndex columns."""
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = raw.columns.get_level_values(0).unique().tolist()
                lvl1 = raw.columns.get_level_values(1).unique().tolist()
                if "Close" in lvl0 and ticker in lvl1:
                    return raw["Close"][ticker].ffill()
                if ticker in lvl0 and "Close" in lvl1:
                    return raw[ticker]["Close"].ffill()
            if "Close" in raw.columns:
                return raw["Close"].ffill()
        except Exception:
            pass
        return pd.Series(dtype=float, name=ticker)

    vix = _close("^VIX")
    hyg = _close("HYG")
    lqd = _close("LQD")
    uup = _close("UUP")

    # ── VIX features ──────────────────────────────────────────────────────────
    if not vix.empty:
        vix_pct = vix.expanding(min_periods=20).rank(pct=True).fillna(0.5)
        vix_ret  = np.log(vix / vix.shift(1)).fillna(0.0)
    else:
        vix_pct = vix_ret = pd.Series(dtype=float)

    # ── Credit spread: log(HYG / LQD) — widens (goes more negative) in stress ─
    common_idx = hyg.index.intersection(lqd.index) if not hyg.empty and not lqd.empty else pd.Index([])
    if len(common_idx) > 0:
        credit_spread     = np.log(hyg.reindex(common_idx) / lqd.reindex(common_idx))
        credit_spread_ret = credit_spread.diff().fillna(0.0)
    else:
        credit_spread = credit_spread_ret = pd.Series(dtype=float)

    # ── Dollar strength ────────────────────────────────────────────────────────
    dxy_ret = np.log(uup / uup.shift(1)).fillna(0.0) if not uup.empty else pd.Series(dtype=float)

    macro = pd.DataFrame({
        "vix_pct":          vix_pct,
        "vix_ret":          vix_ret,
        "credit_spread":    credit_spread,
        "credit_spread_ret":credit_spread_ret,
        "dxy_ret":          dxy_ret,
    })
    return macro


def load_data(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Legacy daily loader used by the training pipeline.

    Returns a pandas DataFrame indexed by date with columns: open, high, low, close, volume.

    Uses yf.Ticker().history() rather than yf.download() so that each call
    gets its own isolated session — safe for concurrent ThreadPoolExecutor use.
    yf.download() is not thread-safe and returns MultiIndex DataFrames with
    duplicate column names when called in parallel, breaking downstream feature
    engineering that expects single-column Series.
    """
    _end = end or datetime.now().strftime("%Y-%m-%d")
    df = yf.Ticker(ticker).history(start=start, end=_end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({start} – {_end})")

    # Ticker.history() always returns flat CamelCase columns — lowercase them.
    df.columns = [c.lower() for c in df.columns]

    # Keep only standard OHLCV; drop dividends, stock splits, etc.
    available = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[available].dropna()

    # Ticker.history() returns a tz-aware (UTC) DatetimeIndex.  Strip the
    # timezone so that all feature DataFrames share a tz-naive index, which
    # is required for clean date intersection in the portfolio engine.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# yfinance loader
# ---------------------------------------------------------------------------

def _load_yfinance_polars(
    ticker: str,
    interval: str,
    start: str,
    end: str | None,
) -> pl.DataFrame:
    yf_interval = _YF_INTERVAL_MAP.get(interval, interval)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=yf_interval,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return pl.DataFrame(schema=_OHLCV_SCHEMA)

    # Flatten MultiIndex columns (yfinance ≥1.0) — same logic as load_data()
    if isinstance(df.columns, pd.MultiIndex):
        _ohlcv = {"Close", "High", "Low", "Open", "Volume"}
        _lvl0  = set(df.columns.get_level_values(0).tolist())
        _lvl1  = set(df.columns.get_level_values(1).tolist())
        if _ohlcv & _lvl0:
            df.columns = df.columns.get_level_values(0)
        elif _ohlcv & _lvl1:
            df.columns = df.columns.get_level_values(1)
        else:
            try:
                df = df.droplevel(1, axis=1)
            except Exception:
                df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index.name = "timestamp"
    df = df.reset_index()

    pdf = pl.from_pandas(df).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )
    return pdf.rename({c: c for c in pdf.columns})  # normalise column names


# ---------------------------------------------------------------------------
# Binance REST loader
# ---------------------------------------------------------------------------

def _load_binance(
    symbol: str,
    interval: str,
    start: str,
    end: str | None,
) -> pl.DataFrame:
    """Paginate Binance klines endpoint until the full range is fetched."""
    start_ms = _to_ms(start)
    end_ms   = _to_ms(end) if end else int(time.time() * 1000)

    rows: list[dict] = []
    while start_ms < end_ms:
        params = {
            "symbol":    symbol.upper(),
            "interval":  interval,
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     _BINANCE_MAX_LIMIT,
        }
        resp = requests.get(_BINANCE_KLINES_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        for k in data:
            rows.append({
                "timestamp": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                "open":      float(k[1]),
                "high":      float(k[2]),
                "low":       float(k[3]),
                "close":     float(k[4]),
                "volume":    float(k[5]),
            })

        if len(data) < _BINANCE_MAX_LIMIT:
            break
        start_ms = data[-1][0] + 1  # advance past the last returned candle

    if not rows:
        return pl.DataFrame(schema=_OHLCV_SCHEMA)

    return pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CRYPTO_SUFFIXES = ("USDT", "BTC", "ETH", "BNB", "BUSD", "USDC")

_OHLCV_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "open":      pl.Float64,
    "high":      pl.Float64,
    "low":       pl.Float64,
    "close":     pl.Float64,
    "volume":    pl.Float64,
}


def _resolve_source(ticker: str, source: str) -> str:
    if source != "auto":
        return source
    upper = ticker.upper()
    return "binance" if any(upper.endswith(s) for s in _CRYPTO_SUFFIXES) else "yfinance"


def _to_ms(date_str: str) -> int:
    return int(pd.Timestamp(date_str).timestamp() * 1000)
