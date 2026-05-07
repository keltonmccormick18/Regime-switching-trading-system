"""Signal generation: convert model price predictions → directional trade signals.

The central idea is that a model outputs a horizon-length array of predicted
future prices.  The SignalGenerator distils that into three numbers:

  direction   — int   -1 (short) | 0 (flat) | 1 (long)
  magnitude   — float  |expected_return| as a fraction of current price
  confidence  — float  [0, 1]  signal-to-noise ratio of the prediction

A signal is suppressed (direction = 0) when:
  • The expected return is smaller than `min_return_threshold`   (noise filter)
  • The confidence is below `min_confidence`                     (uncertainty filter)
  • The regime is in the caller's blacklist                      (regime filter)

Usage::

    gen = SignalGenerator()
    sig = gen.from_price_prediction(
        symbol      = "AAPL",
        prediction  = model.predict(X)[-1],   # (horizon,) array
        current_price = 175.30,
        regime      = "HIGH_VOL_BEAR",
    )
    if sig.direction != 0:
        order = engine.on_signal(sig)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


# ─────────────────────────── Output dataclass ──────────────────────────────

@dataclass
class GeneratedSignal:
    symbol: str
    direction: int             # -1, 0, 1
    magnitude: float           # |expected_return| (fraction, e.g. 0.02 = 2 %)
    confidence: float          # [0, 1]
    expected_return: float     # signed expected return (negative = bearish)
    prediction: list[float]    # raw prediction array from model
    current_price: float
    horizon: int               # number of bars in prediction
    regime: str = ""
    model_used: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    # Convenience ──────────────────────────────────────────────────────────

    @property
    def is_long(self) -> bool:
        return self.direction == 1

    @property
    def is_short(self) -> bool:
        return self.direction == -1

    @property
    def is_flat(self) -> bool:
        return self.direction == 0

    def to_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "direction":       self.direction,
            "magnitude":       round(self.magnitude, 6),
            "confidence":      round(self.confidence, 4),
            "expected_return": round(self.expected_return, 6),
            "current_price":   self.current_price,
            "horizon":         self.horizon,
            "regime":          self.regime,
            "model_used":      self.model_used,
            "timestamp":       self.timestamp,
        }


# ─────────────────────────── Generator ────────────────────────────────────

class SignalGenerator:
    """Convert a model price-prediction array into a trade signal.

    Parameters
    ----------
    min_return_threshold : float
        Minimum |expected_return| (as a fraction) required to emit a non-flat
        signal.  Default 0.005 = 0.5 %.
    min_confidence : float
        Minimum confidence score required.  Signals below this are set flat.
        Default 0.3.
    use_horizon_slope : bool
        If True, weight the expected return by the slope of the prediction
        curve (rising = more bullish) in addition to endpoint vs. current.
        Default True.
    regime_blacklist : list[str]
        Regime names for which all signals are forced flat.
        E.g. ["HIGH_VOL_BEAR"] to stay out of bear markets.
    """

    def __init__(
        self,
        min_return_threshold: float = 0.005,
        min_confidence: float = 0.30,
        use_horizon_slope: bool = True,
        regime_blacklist: Optional[list[str]] = None,
    ) -> None:
        self.min_return_threshold = min_return_threshold
        self.min_confidence = min_confidence
        self.use_horizon_slope = use_horizon_slope
        self.regime_blacklist: list[str] = regime_blacklist or []

    # ── Main entry point ──────────────────────────────────────────────────

    def from_price_prediction(
        self,
        symbol: str,
        prediction: np.ndarray,   # shape (horizon,) — predicted future prices
        current_price: float,
        regime: str = "",
        model_used: str = "",
        confidence_override: float | None = None,
    ) -> GeneratedSignal:
        """Generate a signal from a model price-prediction array.

        The expected return is computed as the mean predicted price vs. the
        current price.  Confidence is a signal-to-noise ratio: large return
        relative to prediction spread → high confidence.

        Args:
            symbol:        Ticker symbol.
            prediction:    1-D array of predicted future prices (horizon length).
            current_price: Current (last known) close price.
            regime:        Current market regime name.
            model_used:    Name of the model that produced the prediction.

        Returns:
            GeneratedSignal with direction, magnitude, and confidence filled in.
        """
        pred = np.asarray(prediction, dtype=float)
        horizon = len(pred)

        if current_price <= 0 or horizon == 0:
            return self._flat(symbol, prediction, current_price, horizon, regime, model_used)

        # ── Expected return: mean predicted vs current ─────────────────
        pred_mean   = float(pred.mean())
        pred_end    = float(pred[-1])
        pred_std    = float(pred.std()) if horizon > 1 else 0.0

        mean_return = (pred_mean - current_price) / current_price
        end_return  = (pred_end  - current_price) / current_price

        # ── Optional slope component ───────────────────────────────────
        if self.use_horizon_slope and horizon > 1:
            # Normalised OLS slope over the prediction horizon
            x = np.arange(horizon, dtype=float)
            slope = float(np.polyfit(x, pred, 1)[0])            # price units / bar
            slope_return = slope * horizon / current_price       # normalise to return scale
            # Blend: 50% endpoint, 30% mean, 20% slope direction
            expected_return = 0.50 * end_return + 0.30 * mean_return + 0.20 * slope_return
        else:
            expected_return = 0.60 * end_return + 0.40 * mean_return

        # ── Confidence: SNR or conformal override ─────────────────────
        abs_return  = abs(expected_return)
        noise_level = pred_std / current_price if current_price > 0 else 1.0
        snr_conf    = _sigmoid((abs_return - noise_level) / max(noise_level, 1e-8))
        # ConformalWrapper provides a calibrated alternative; use it when given
        confidence  = float(confidence_override) if confidence_override is not None else snr_conf

        # ── Direction + filters ────────────────────────────────────────
        if (
            regime in self.regime_blacklist
            or abs_return < self.min_return_threshold
            or confidence < self.min_confidence
        ):
            direction = 0
        else:
            direction = int(math.copysign(1, expected_return))

        return GeneratedSignal(
            symbol          = symbol,
            direction       = direction,
            magnitude       = abs_return,
            confidence      = round(confidence, 4),
            expected_return = expected_return,
            prediction      = pred.tolist(),
            current_price   = current_price,
            horizon         = horizon,
            regime          = regime,
            model_used      = model_used,
        )

    def from_return_prediction(
        self,
        symbol: str,
        predicted_return: float,   # single predicted return (e.g. from regression head)
        confidence: float = 0.5,
        current_price: float = 0.0,
        regime: str = "",
        model_used: str = "",
    ) -> GeneratedSignal:
        """Simpler entry point for models that output a single return estimate."""
        abs_return = abs(predicted_return)
        if (
            regime in self.regime_blacklist
            or abs_return < self.min_return_threshold
            or confidence < self.min_confidence
        ):
            direction = 0
        else:
            direction = int(math.copysign(1, predicted_return))

        return GeneratedSignal(
            symbol          = symbol,
            direction       = direction,
            magnitude       = abs_return,
            confidence      = confidence,
            expected_return = predicted_return,
            prediction      = [current_price * (1 + predicted_return)],
            current_price   = current_price,
            horizon         = 1,
            regime          = regime,
            model_used      = model_used,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _flat(
        self,
        symbol: str,
        prediction,
        current_price: float,
        horizon: int,
        regime: str,
        model_used: str,
    ) -> GeneratedSignal:
        pred = np.asarray(prediction, dtype=float)
        return GeneratedSignal(
            symbol=symbol, direction=0, magnitude=0.0, confidence=0.0,
            expected_return=0.0, prediction=pred.tolist(),
            current_price=current_price, horizon=horizon,
            regime=regime, model_used=model_used,
        )


# ─────────────────────────── Utility ──────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid, clipped to [0, 1]."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
