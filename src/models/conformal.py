"""Conformal Prediction Wrapper — applied to all regime models.

Conformal prediction provides distribution-free coverage guarantees: at
coverage level (1-alpha), the true future price will fall within the
returned interval on at least (1-alpha) fraction of test windows,
regardless of the base model or data distribution (Vovk et al. 2005).

Implementation
--------------
Split conformal prediction (inductive conformal):
  1. fit() splits training data: 80% train → base model, 20% calibration.
  2. Nonconformity scores: |y_cal - ŷ_cal| per (sample, horizon_step).
  3. Per-horizon quantile q_h stored at level (1-alpha)(1 + 1/n_cal)
     (finite-sample correction for exact coverage).
  4. predict() returns the base model's point prediction unchanged.
  5. predict_interval() returns [ŷ - q, ŷ + q] per horizon step.
  6. conformal_confidence() maps interval width to [0,1]: narrow → high
     confidence → larger position size via the signal generator.

predict() is fully backward compatible — all existing callers see no change.
The enhanced confidence flows into position sizing only in the paper engine
and predict endpoint, which check for the conformal_confidence() method.
"""
from __future__ import annotations

import numpy as np

from src.models.base import BaseModel


class ConformalWrapper(BaseModel):
    """Wraps any BaseModel with split conformal prediction intervals.

    Parameters
    ----------
    base_model : BaseModel
        Any fitted or unfitted model implementing fit() / predict().
    alpha : float
        Miscoverage level.  Default 0.10 gives 90% coverage intervals.
    cal_ratio : float
        Fraction of training data reserved for calibration.  Default 0.20.
    """

    def __init__(
        self,
        base_model: BaseModel,
        alpha:      float = 0.10,
        cal_ratio:  float = 0.20,
    ):
        self.base_model = base_model
        self.alpha      = alpha
        self.cal_ratio  = cal_ratio
        self._q: np.ndarray | None = None   # (horizon,) conformal quantiles

    # ── BaseModel interface ────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train base model on (1-cal_ratio) of data; calibrate on remainder."""
        n         = len(X)
        cal_start = int(n * (1.0 - self.cal_ratio))

        # Guard: need at least a few calibration samples
        if cal_start >= n - 5:
            self.base_model.fit(X, y)
            self._q = None
            return

        X_train, X_cal = X[:cal_start], X[cal_start:]
        y_train, y_cal = y[:cal_start], y[cal_start:]

        self.base_model.fit(X_train, y_train)

        y_hat   = self.base_model.predict(X_cal)         # (n_cal, horizon)
        scores  = np.abs(y_cal - y_hat)                  # (n_cal, horizon)
        n_cal   = len(X_cal)

        # Finite-sample corrected quantile level (Tibshirani et al.)
        level   = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n_cal))
        self._q = np.quantile(scores, level, axis=0)     # (horizon,)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction from base model — unchanged for callers."""
        return self.base_model.predict(X)

    # ── Extended interface ─────────────────────────────────────────────────

    def predict_interval(
        self,
        X:     np.ndarray,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction intervals with coverage (1-alpha).

        Shape of each: (n_samples, horizon).  Falls back to a zero-width
        interval if the wrapper has not been calibrated yet.
        """
        preds = self.base_model.predict(X)               # (n, horizon)

        if self._q is None:
            return preds.copy(), preds.copy()

        if alpha is not None and alpha != self.alpha:
            # Recompute is not possible without stored calibration data;
            # scale stored quantile proportionally as a best-effort estimate.
            scale = (1.0 - alpha) / max(1.0 - self.alpha, 1e-8)
            q = self._q * scale
        else:
            q = self._q

        lower = preds - q                                 # (n, horizon)
        upper = preds + q
        return lower, upper

    def conformal_confidence(self, X: np.ndarray) -> np.ndarray:
        """Map interval width to a [0, 1] confidence score per sample.

        Narrow interval relative to the predicted price level → high
        confidence → the signal generator emits a larger position size.

        Shape: (n_samples,)
        """
        if self._q is None:
            return np.full(len(X), 0.5, dtype=np.float32)

        preds       = self.base_model.predict(X)                  # (n, h)
        lower, upper = self.predict_interval(X)

        mid_price   = np.abs(preds).mean(axis=1, keepdims=True)   # (n, 1)
        mid_price   = np.maximum(mid_price, 1.0)

        rel_width   = (upper - lower) / mid_price                 # (n, h)
        avg_width   = rel_width.mean(axis=1)                      # (n,)

        # Exponential decay: avg_width ≈ 0 → conf ≈ 1; wide → conf → 0
        confidence  = np.exp(-avg_width * 15.0)
        return np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def direction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Direction certainty — delegates to base model if it has a direction head.

        Falls back to conformal_confidence() (interval-width-based) when the
        base model does not expose direction_confidence().
        """
        if hasattr(self.base_model, "direction_confidence"):
            return self.base_model.direction_confidence(X)
        return self.conformal_confidence(X)

    def direction_prob(self, X: np.ndarray) -> np.ndarray:
        """Raw direction probability — delegates to base model if available.

        When the base model has no direction head, returns 0.5 (uncertain)
        for all samples so the caller falls back to SNR-based confidence.
        """
        if hasattr(self.base_model, "direction_prob"):
            return self.base_model.direction_prob(X)
        return np.full(len(X), 0.5, dtype=np.float32)

    # ── Delegation (pass-through TFT quantile access) ─────────────────────

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        """Delegate to base model if it supports quantile prediction (TFT)."""
        if hasattr(self.base_model, "predict_quantiles"):
            return self.base_model.predict_quantiles(X)
        raise AttributeError(
            f"{type(self.base_model).__name__} does not support predict_quantiles()"
        )
