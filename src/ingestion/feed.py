"""DataFeed — unified interface for historical data, live streaming, and resampling.

DataFeed ties together the three ingestion layers:
    1. Historical  — load_historical() via yfinance or Binance REST
    2. Live        — BinanceStream / AlpacaStream / YFinancePoller, all running as daemon threads
    3. Buffer      — thread-safe deque per (symbol, interval) key; queryable as polars DataFrames

Quick start:
    from src.ingestion.feed import DataFeed

    feed = DataFeed()

    # Historical (returns polars DataFrame)
    hist = feed.history("BTCUSDT", start="2024-01-01", interval="1h", source="binance")
    hist = feed.history("SPY",     start="2023-01-01", interval="1d")

    # Live streaming
    feed.subscribe_crypto("BTCUSDT", interval="1m")
    feed.subscribe_equity("SPY")                         # yfinance polling (no keys)
    feed.subscribe_equity("AAPL", use_alpaca=True)       # Alpaca WebSocket (needs env vars)

    # Query the live buffer
    recent = feed.latest("BTCUSDT", interval="1m", n=60)  # last 60 bars → polars DataFrame
    hourly = feed.resample("BTCUSDT", from_interval="1m", to_interval="1h")

    # Stop all streams when done
    feed.stop()
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Sequence

import polars as pl

from src.ingestion.historical import load_historical
from src.ingestion.resample   import resample_ohlcv
from src.ingestion.stream     import (
    AlpacaStream,
    BinanceStream,
    OHLCVBar,
    YFinancePoller,
)

_OHLCV_SCHEMA = {
    "symbol":    pl.Utf8,
    "timestamp": pl.Datetime("us", "UTC"),
    "open":      pl.Float64,
    "high":      pl.Float64,
    "low":       pl.Float64,
    "close":     pl.Float64,
    "volume":    pl.Float64,
    "interval":  pl.Utf8,
}


class DataFeed:
    """Unified historical + live data feed with an in-memory bar buffer.

    Args:
        buffer_size: Maximum number of bars held per (symbol, interval) key.
                     Older bars are evicted automatically (deque with maxlen).
    """

    def __init__(self, buffer_size: int = 1_000):
        # key: "{SYMBOL}_{interval}"  →  thread-safe bar buffer
        self._buffers: dict[str, deque[OHLCVBar]] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        self._lock    = threading.Lock()
        self._streams: list[BinanceStream | AlpacaStream | YFinancePoller] = []

    # ------------------------------------------------------------------
    # Historical
    # ------------------------------------------------------------------

    def history(
        self,
        ticker:   str,
        start:    str,
        end:      str | None = None,
        interval: str = "1d",
        source:   str = "auto",
    ) -> pl.DataFrame:
        """Fetch historical OHLCV bars as a polars DataFrame.

        source="auto" routes crypto pairs (ending in USDT, BTC, …) to Binance
        and everything else to yfinance.
        """
        return load_historical(ticker, start=start, end=end, interval=interval, source=source)

    # ------------------------------------------------------------------
    # Live subscriptions
    # ------------------------------------------------------------------

    def subscribe_crypto(self, symbol: str, interval: str = "1m") -> None:
        """Stream closed klines from Binance WebSocket (no credentials needed).

        Bars are stored in the internal buffer and can be read via `latest()`.
        """
        stream = BinanceStream(on_bar=self._on_bar)
        stream.start(symbol, interval=interval)
        self._streams.append(stream)

    def subscribe_equity(
        self,
        symbol:      str,
        use_alpaca:  bool = False,
        alpaca_feed: str  = "iex",
        api_key:     str | None = None,
        secret_key:  str | None = None,
    ) -> None:
        """Stream 1-minute equity bars.

        With use_alpaca=False (default): polls yfinance every 60 s.
            No credentials required; data lags by up to one minute.

        With use_alpaca=True: connects to Alpaca's WebSocket.
            Requires ALPACA_API_KEY + ALPACA_SECRET_KEY env vars (or pass explicitly).
            feed="iex" is free; feed="sip" requires a paid subscription.
        """
        if use_alpaca:
            stream: AlpacaStream | YFinancePoller = AlpacaStream(
                on_bar     = self._on_bar,
                feed       = alpaca_feed,
                api_key    = api_key,
                secret_key = secret_key,
            )
            stream.start(symbol)
        else:
            stream = YFinancePoller(on_bar=self._on_bar)
            stream.start(symbol)
        self._streams.append(stream)

    def subscribe_many_crypto(self, symbols: Sequence[str], interval: str = "1m") -> None:
        """Subscribe to multiple crypto symbols at once (one stream per symbol)."""
        for sym in symbols:
            self.subscribe_crypto(sym, interval=interval)

    # ------------------------------------------------------------------
    # Buffer queries
    # ------------------------------------------------------------------

    def latest(
        self,
        symbol:   str,
        interval: str = "1m",
        n:        int | None = None,
    ) -> pl.DataFrame:
        """Return the most recent bars from the live buffer as a polars DataFrame.

        Args:
            symbol:   Ticker/pair (e.g. "BTCUSDT", "SPY").
            interval: Interval the subscription was started with (e.g. "1m").
            n:        Number of bars to return. None returns the entire buffer.
        """
        key = _buf_key(symbol, interval)
        with self._lock:
            bars = list(self._buffers[key])
        if n is not None:
            bars = bars[-n:]
        return _bars_to_polars(bars)

    def resample(
        self,
        symbol:        str,
        from_interval: str = "1m",
        to_interval:   str = "5m",
    ) -> pl.DataFrame:
        """Resample buffered bars from a finer to a coarser interval.

        Example: convert 1-min bars to 5-min OHLCV in one call.
        """
        df = self.latest(symbol, interval=from_interval)
        if df.is_empty():
            return df
        return resample_ohlcv(df, to_interval)

    def symbols(self) -> list[str]:
        """Return all (symbol, interval) keys currently in the buffer."""
        with self._lock:
            return list(self._buffers.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop all active streams."""
        for s in self._streams:
            s.stop()
        self._streams.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_bar(self, bar: OHLCVBar) -> None:
        key = _buf_key(bar.symbol, bar.interval)
        with self._lock:
            self._buffers[key].append(bar)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buf_key(symbol: str, interval: str) -> str:
    return f"{symbol.upper()}_{interval}"


def _bars_to_polars(bars: list[OHLCVBar]) -> pl.DataFrame:
    if not bars:
        return pl.DataFrame(schema=_OHLCV_SCHEMA)
    return pl.DataFrame({
        "symbol":    [b.symbol    for b in bars],
        "timestamp": [b.timestamp for b in bars],
        "open":      [b.open      for b in bars],
        "high":      [b.high      for b in bars],
        "low":       [b.low       for b in bars],
        "close":     [b.close     for b in bars],
        "volume":    [b.volume    for b in bars],
        "interval":  [b.interval  for b in bars],
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
