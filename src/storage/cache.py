"""Redis storage for real-time trading signals.

Three access patterns are supported for each symbol:

  1. Latest signal   — `SET signal:latest:{symbol}` with a configurable TTL.
                       Fast O(1) read for the current signal value.

  2. Signal history  — `LPUSH signal:history:{symbol}` — a capped list (LTRIM)
                       of the N most recent signal JSON blobs. Useful for
                       short-term replay and dashboard sparklines.

  3. Pub/Sub         — `PUBLISH signals {json}` — fire-and-forget broadcast to
                       any subscribers (dashboard, paper trader, alerting).

The `Signal` dataclass is the canonical in-process representation.
JSON serialisation/deserialisation happens at the Redis boundary.

Usage:
    cache = SignalCache()
    cache.ping()            # verify connectivity

    sig = Signal(symbol="SPY", timestamp="2024-01-15T10:30:00Z",
                 signal=1, regime="low_vol_bull", model_used="TCNModel",
                 prediction=[450.1, 451.3], confidence=0.68)

    cache.set_signal(sig)                      # store latest + push to history
    cache.publish(sig)                         # broadcast to pub/sub channel

    latest = cache.get_signal("SPY")           # → Signal | None
    hist   = cache.get_signal_history("SPY", n=20)  # → list[Signal]

    # Subscribe to all signals in another thread/process:
    for sig in cache.listen():
        print(sig)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generator

import redis

_DEFAULT_URL     = "redis://localhost:6379/0"
_LATEST_KEY      = "signal:latest:{symbol}"
_HISTORY_KEY     = "signal:history:{symbol}"
_CHANNEL         = "signals"
_DEFAULT_TTL     = 300   # seconds — latest signal expires after 5 minutes by default
_DEFAULT_MAXHIST = 500   # max bars kept per symbol history list


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    symbol:     str
    signal:     int              # -1 (short) | 0 (flat) | 1 (long)
    timestamp:  str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    regime:     str = ""
    model_used: str = ""
    prediction: list[float] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol":     self.symbol,
            "signal":     self.signal,
            "timestamp":  self.timestamp,
            "regime":     self.regime,
            "model_used": self.model_used,
            "prediction": self.prediction,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Signal:
        return cls(
            symbol     = d["symbol"],
            signal     = int(d["signal"]),
            timestamp  = d.get("timestamp", ""),
            regime     = d.get("regime", ""),
            model_used = d.get("model_used", ""),
            prediction = d.get("prediction", []),
            confidence = float(d.get("confidence", 0.0)),
        )


# ---------------------------------------------------------------------------
# SignalCache
# ---------------------------------------------------------------------------

class SignalCache:
    """Redis-backed store for real-time trading signals.

    Args:
        url:      Redis connection URL. Reads from REDIS_URL env var if not given.
        ttl:      Seconds before a 'latest' key expires. Default 300 (5 min).
        max_hist: Maximum number of signals retained per symbol in history. Default 500.
    """

    def __init__(
        self,
        url:      str | None = None,
        ttl:      int = _DEFAULT_TTL,
        max_hist: int = _DEFAULT_MAXHIST,
    ):
        self._url      = url or os.environ.get("REDIS_URL", _DEFAULT_URL)
        self._ttl      = ttl
        self._max_hist = max_hist
        self._client   = redis.from_url(self._url, decode_responses=True)

    # --- Connectivity ---

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            return self._client.ping()
        except redis.RedisError:
            return False

    # --- Write ---

    def set_signal(self, sig: Signal, ttl: int | None = None) -> None:
        """Store `sig` as the current latest signal for its symbol.

        Sets a TTL so stale signals expire automatically.
        """
        key  = _LATEST_KEY.format(symbol=sig.symbol.upper())
        blob = json.dumps(sig.to_dict())
        self._client.set(key, blob, ex=ttl or self._ttl)

    def push_signal(self, sig: Signal) -> None:
        """Prepend `sig` to the per-symbol history list and trim to max_hist."""
        key  = _HISTORY_KEY.format(symbol=sig.symbol.upper())
        blob = json.dumps(sig.to_dict())
        pipe = self._client.pipeline()
        pipe.lpush(key, blob)
        pipe.ltrim(key, 0, self._max_hist - 1)
        pipe.execute()

    def record(self, sig: Signal, ttl: int | None = None) -> None:
        """Convenience: set_signal + push_signal in a single call."""
        self.set_signal(sig, ttl=ttl)
        self.push_signal(sig)

    def publish(self, sig: Signal) -> int:
        """Publish `sig` to the global signals channel.

        Returns the number of subscribers that received the message.
        """
        return self._client.publish(_CHANNEL, json.dumps(sig.to_dict()))

    def broadcast(self, sig: Signal, ttl: int | None = None) -> None:
        """record() + publish() in one call — the typical hot path."""
        self.record(sig, ttl=ttl)
        self.publish(sig)

    # --- Read ---

    def get_signal(self, symbol: str) -> Signal | None:
        """Return the current latest signal for `symbol`, or None if expired/absent."""
        key  = _LATEST_KEY.format(symbol=symbol.upper())
        blob = self._client.get(key)
        return Signal.from_dict(json.loads(blob)) if blob else None

    def get_signal_history(self, symbol: str, n: int = 100) -> list[Signal]:
        """Return the `n` most recent signals for `symbol` (newest first)."""
        key   = _HISTORY_KEY.format(symbol=symbol.upper())
        blobs = self._client.lrange(key, 0, n - 1)
        return [Signal.from_dict(json.loads(b)) for b in blobs]

    def ttl(self, symbol: str) -> int:
        """Return remaining TTL (seconds) of the latest-signal key, or -2 if absent."""
        return self._client.ttl(_LATEST_KEY.format(symbol=symbol.upper()))

    # --- Pub/Sub ---

    def listen(self, channel: str = _CHANNEL) -> Generator[Signal, None, None]:
        """Block and yield Signal objects as they arrive on the pub/sub channel.

        Designed to run in a dedicated thread:
            for sig in cache.listen():
                process(sig)
        """
        ps = self._client.pubsub()
        ps.subscribe(channel)
        for msg in ps.listen():
            if msg["type"] != "message":
                continue
            try:
                yield Signal.from_dict(json.loads(msg["data"]))
            except (json.JSONDecodeError, KeyError):
                continue

    def get_pubsub(self, channel: str = _CHANNEL) -> redis.client.PubSub:
        """Return a raw PubSub handle for manual subscription management."""
        ps = self._client.pubsub()
        ps.subscribe(channel)
        return ps

    # --- Cleanup ---

    def clear_symbol(self, symbol: str) -> None:
        """Delete all signal keys for a symbol (latest + history)."""
        upper = symbol.upper()
        self._client.delete(
            _LATEST_KEY.format(symbol=upper),
            _HISTORY_KEY.format(symbol=upper),
        )

    def flush(self) -> None:
        """Delete ALL signal keys (latest + history) across all symbols.

        Does NOT use FLUSHDB — only removes keys matching the signal: prefix.
        """
        for pattern in ("signal:latest:*", "signal:history:*"):
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
