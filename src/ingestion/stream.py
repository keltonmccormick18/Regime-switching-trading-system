"""Live data streaming via WebSocket (Binance, Alpaca) and polling fallback (yfinance).

Three stream implementations share a common OHLCVBar datatype and BarCallback protocol:

    BinanceStream   — Binance kline WebSocket (crypto, no auth needed).
    AlpacaStream    — Alpaca bar WebSocket (equities, requires API keys).
    YFinancePoller  — Polls yfinance every ~60 s (equities fallback, no keys needed).

Each stream runs in its own daemon thread with a dedicated asyncio event loop.
Reconnect with exponential back-off is built in to both WebSocket implementations.

Usage:
    def on_bar(bar: OHLCVBar) -> None:
        print(bar.symbol, bar.close)

    s = BinanceStream(on_bar)
    s.start("BTCUSDT", interval="1m")

    s2 = YFinancePoller(on_bar)
    s2.start("SPY")
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
import yfinance as yf

try:
    import websockets
    _HAS_WS = True
except ImportError:
    _HAS_WS = False


# ---------------------------------------------------------------------------
# Shared data type
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class OHLCVBar:
    symbol:    str
    timestamp: datetime   # UTC, candle-close time
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    interval:  str        # "1m", "5m", "1d", …

    def __repr__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M")
        return f"OHLCVBar({self.symbol} {ts} C={self.close:.4f} V={self.volume:.0f} [{self.interval}])"


BarCallback = Callable[[OHLCVBar], None]


# ---------------------------------------------------------------------------
# Binance WebSocket stream (crypto)
# ---------------------------------------------------------------------------

class BinanceStream:
    """Stream closed klines from Binance WebSocket (no API key required).

    Fires `on_bar` once per *closed* candle (i.e. `k.x == true` in the payload).
    Automatically reconnects with exponential back-off on disconnect or error.

    Args:
        on_bar:     Callback invoked with each completed OHLCVBar.
        max_delay:  Maximum reconnect wait in seconds (default 60).
    """

    _WS_BASE = "wss://stream.binance.com:9443/ws"

    def __init__(self, on_bar: BarCallback, max_delay: float = 60.0):
        if not _HAS_WS:
            raise ImportError("websockets package is required: pip install websockets")
        self.on_bar    = on_bar
        self.max_delay = max_delay
        self._running  = False
        self._thread:  threading.Thread | None = None

    # --- public ---

    def start(self, symbol: str, interval: str = "1m") -> threading.Thread:
        """Start streaming in a background daemon thread. Returns the thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._thread_main,
            args=(symbol, interval),
            daemon=True,
            name=f"binance-{symbol}-{interval}",
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._running = False

    # --- internals ---

    def _thread_main(self, symbol: str, interval: str) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._run_with_reconnect(symbol, interval))
        loop.close()

    async def _run_with_reconnect(self, symbol: str, interval: str) -> None:
        delay = 1.0
        while self._running:
            try:
                await self._connect(symbol, interval)
                delay = 1.0  # reset on clean disconnect
            except Exception:
                if not self._running:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_delay)

    async def _connect(self, symbol: str, interval: str) -> None:
        stream = f"{symbol.lower()}@kline_{interval}"
        url    = f"{self._WS_BASE}/{stream}"
        async with websockets.connect(url, ping_interval=20) as ws:
            async for raw in ws:
                if not self._running:
                    return
                self._handle(json.loads(raw), symbol, interval)

    def _handle(self, msg: dict, symbol: str, interval: str) -> None:
        k = msg.get("k", {})
        if not k.get("x"):  # candle not yet closed
            return
        bar = OHLCVBar(
            symbol    = symbol.upper(),
            timestamp = datetime.fromtimestamp(k["T"] / 1000, tz=timezone.utc),
            open      = float(k["o"]),
            high      = float(k["h"]),
            low       = float(k["l"]),
            close     = float(k["c"]),
            volume    = float(k["v"]),
            interval  = interval,
        )
        self.on_bar(bar)


# ---------------------------------------------------------------------------
# Alpaca WebSocket stream (equities)
# ---------------------------------------------------------------------------

class AlpacaStream:
    """Stream 1-minute bars from Alpaca's market data WebSocket.

    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables
    (or pass them explicitly). Free tier uses the IEX feed; paid uses SIP.

    Args:
        on_bar:     Callback invoked with each completed OHLCVBar.
        feed:       "iex" (free) or "sip" (paid, real-time NBBO).
        api_key:    Alpaca API key (falls back to env var ALPACA_API_KEY).
        secret_key: Alpaca secret key (falls back to env var ALPACA_SECRET_KEY).
    """

    _WS_URLS = {
        "iex": "wss://stream.data.alpaca.markets/v2/iex",
        "sip": "wss://stream.data.alpaca.markets/v2/sip",
    }

    def __init__(
        self,
        on_bar:     BarCallback,
        feed:       str = "iex",
        api_key:    str | None = None,
        secret_key: str | None = None,
        max_delay:  float = 60.0,
    ):
        if not _HAS_WS:
            raise ImportError("websockets package is required: pip install websockets")
        self.on_bar    = on_bar
        self.url       = self._WS_URLS[feed]
        self.api_key   = api_key    or os.environ.get("ALPACA_API_KEY", "")
        self.secret    = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.max_delay = max_delay
        self._running  = False
        self._thread:  threading.Thread | None = None

        if not self.api_key or not self.secret:
            raise ValueError(
                "Alpaca API keys required. Set ALPACA_API_KEY / ALPACA_SECRET_KEY env vars "
                "or pass api_key and secret_key to AlpacaStream()."
            )

    # --- public ---

    def start(self, *symbols: str) -> threading.Thread:
        """Start streaming bars for one or more symbols in a background thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._thread_main,
            args=(list(symbols),),
            daemon=True,
            name=f"alpaca-{','.join(symbols)}",
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._running = False

    # --- internals ---

    def _thread_main(self, symbols: list[str]) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._run_with_reconnect(symbols))
        loop.close()

    async def _run_with_reconnect(self, symbols: list[str]) -> None:
        delay = 1.0
        while self._running:
            try:
                await self._connect(symbols)
                delay = 1.0
            except Exception:
                if not self._running:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_delay)

    async def _connect(self, symbols: list[str]) -> None:
        async with websockets.connect(self.url, ping_interval=20) as ws:
            # Authenticate
            await ws.send(json.dumps({"action": "auth", "key": self.api_key, "secret": self.secret}))
            await ws.recv()  # auth response

            # Subscribe to 1-min bars
            await ws.send(json.dumps({"action": "subscribe", "bars": symbols}))

            async for raw in ws:
                if not self._running:
                    return
                for msg in json.loads(raw):
                    if msg.get("T") == "b":
                        self._handle(msg)

    def _handle(self, msg: dict) -> None:
        bar = OHLCVBar(
            symbol    = msg["S"],
            timestamp = pd.Timestamp(msg["t"]).to_pydatetime().replace(tzinfo=timezone.utc),
            open      = float(msg["o"]),
            high      = float(msg["h"]),
            low       = float(msg["l"]),
            close     = float(msg["c"]),
            volume    = float(msg["v"]),
            interval  = "1m",
        )
        self.on_bar(bar)


# ---------------------------------------------------------------------------
# yfinance polling fallback (equities, no API key)
# ---------------------------------------------------------------------------

class YFinancePoller:
    """Poll yfinance for the latest 1-minute bars every `poll_interval` seconds.

    Deduplicates on timestamp so each bar is delivered exactly once.
    Only the most recent trading session is available (yfinance 1-min data
    is limited to the last 7 days).

    Args:
        on_bar:        Callback invoked with each new OHLCVBar.
        poll_interval: Seconds between polls (default 60 — one candle length).
    """

    def __init__(self, on_bar: BarCallback, poll_interval: float = 60.0):
        self.on_bar        = on_bar
        self.poll_interval = poll_interval
        self._seen:    set[datetime] = set()
        self._running: bool = False
        self._thread:  threading.Thread | None = None

    def start(self, ticker: str) -> threading.Thread:
        """Start polling in a background daemon thread. Returns the thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop,
            args=(ticker,),
            daemon=True,
            name=f"yf-poller-{ticker}",
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._running = False

    def _poll_loop(self, ticker: str) -> None:
        while self._running:
            try:
                df = yf.download(ticker, period="1d", interval="1m", progress=False)
                df.columns = [c.lower() for c in df.columns]
                for ts, row in df.iterrows():
                    ts_dt = ts.to_pydatetime().replace(tzinfo=timezone.utc)
                    if ts_dt not in self._seen:
                        self._seen.add(ts_dt)
                        self.on_bar(OHLCVBar(
                            symbol    = ticker.upper(),
                            timestamp = ts_dt,
                            open      = float(row["open"]),
                            high      = float(row["high"]),
                            low       = float(row["low"]),
                            close     = float(row["close"]),
                            volume    = float(row["volume"]),
                            interval  = "1m",
                        ))
            except Exception:
                pass  # retry on next poll cycle
            time.sleep(self.poll_interval)
