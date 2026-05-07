"""Online / Adaptive Learning model — regime-transition hedge.

Uses River's Adaptive Random Forest (ARF) regressor, which internally tracks
concept drift (ADWIN detector) and replaces stale trees when a drift is
detected.  This makes it the only model in the system that genuinely adapts
in real-time without a full retraining cycle.

Design choices
--------------
* One ARFRegressor per horizon step (horizon models total) — River has no
  native multi-output support, but this is lightweight since ARF trees are
  small.
* fit() trains sequentially through all historical samples (learn_one per
  sample) — River models must see data in chronological order.
* update(X, y) provides incremental learning on new samples without
  resetting the model state.  The paper engine calls this each tick so the
  model continuously adapts to live market conditions.
* Features: last time-step values + per-channel rolling statistics.

Input shape:  (n_samples, n_features, lookback)
Output shape: (n_samples, horizon)
"""
from __future__ import annotations

import numpy as np

from src.models.base import BaseModel

try:
    from river import forest as _river_forest
    _RIVER_AVAILABLE = True
except ImportError:
    _RIVER_AVAILABLE = False


def _check_river() -> None:
    if not _RIVER_AVAILABLE:
        raise ImportError("river is required: pip install river")


# ---------------------------------------------------------------------------
# Feature extraction (single sample)
# ---------------------------------------------------------------------------

def _featurize_one(x: np.ndarray) -> np.ndarray:
    """Extract a flat feature vector from one sequence window.

    Input:  x — (n_features, lookback)
    Output: 1-D array of length n_features * 5
            [last, mean, std, momentum, z_score] per feature channel
    """
    last     = x[:, -1]
    mean     = x.mean(axis=1)
    std      = x.std(axis=1) + 1e-8
    momentum = last - x[:, 0]
    z_score  = (last - mean) / std
    return np.concatenate([last, mean, std, momentum, z_score])


def _to_river_dict(vec: np.ndarray) -> dict[str, float]:
    return {f"f{i}": float(v) for i, v in enumerate(vec)}


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class OnlineModel(BaseModel):
    """River ARF online adaptive model.

    Unlike all other models in this system, OnlineModel state persists across
    fit() calls — each call continues learning rather than resetting.  This
    is the intended River behaviour for streaming / concept-drift settings.

    The paper engine calls update() on each tick with the latest data window,
    giving the model a rolling view of the most recent market dynamics.
    """

    def __init__(
        self,
        n_features:   int = 7,
        lookback:     int = 64,
        horizon:      int = 16,
        n_models:     int = 15,    # trees per ARF ensemble; 15 balances variance vs speed
                                   # (25 trees × 16 horizon = 400 learn_one calls/sample
                                   # at ~2ms each → 50ms/sample → 400 samples ≈ 20s;
                                   # 15 trees → 30ms/sample → 400 samples ≈ 12s)
        max_features: str = "sqrt",
        drift_detector: str = "ADWIN",
        grace_period: int  = 50,   # Hoeffding split threshold (default 200 is too high
                                   # for regime-filtered datasets of ~50–200 windows)
        max_depth:    int  = 10,   # prevent overfitting on small datasets
        # Accepted for API compatibility with train_model() kwargs
        epochs:       int = 0,
    ):
        _check_river()
        self.n_features   = n_features
        self.lookback     = lookback
        self.horizon      = horizon
        self.n_models     = n_models
        self.max_features = max_features
        self.grace_period = grace_period
        self.max_depth    = max_depth

        # One ARF per forecast horizon step
        self.models: list = self._build_models()
        self._fitted = False

    def _build_models(self) -> list:
        return [
            _river_forest.ARFRegressor(
                n_models     = self.n_models,
                max_features = self.max_features,
                seed         = h,
                grace_period = self.grace_period,
                max_depth    = self.max_depth,
            )
            for h in range(self.horizon)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Sequential training pass through all historical samples.

        Intentionally does NOT reset model state — subsequent calls continue
        learning from where the previous call left off (online behaviour).
        """
        for i in range(len(X)):
            x_dict = _to_river_dict(_featurize_one(X[i]))
            for h, model in enumerate(self.models):
                model.learn_one(x_dict, float(y[i, h]))
        self._fitted = True

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incremental update on new samples — called by paper engine each tick."""
        self.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for a batch of samples. Returns (n_samples, horizon)."""
        results = np.zeros((len(X), self.horizon), dtype=np.float32)
        for i in range(len(X)):
            x_dict = _to_river_dict(_featurize_one(X[i]))
            for h, model in enumerate(self.models):
                pred = model.predict_one(x_dict)
                results[i, h] = float(pred) if pred is not None else 0.0
        return results

    def direction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Direction certainty — fraction of horizon steps with consistent sign.

        Returns |frac_pos - 0.5| * 2: high when all steps agree (all up or
        all down), low when steps are mixed.  Does NOT encode direction —
        use direction_prob() for that.
        """
        preds    = self.predict(X)                        # (n, horizon)
        frac_pos = (preds > 0).mean(axis=1)               # fraction of steps positive
        return (np.abs(frac_pos - 0.5) * 2).astype(np.float32)

    def direction_prob(self, X: np.ndarray) -> np.ndarray:
        """Raw direction probability, shape (n,).

        Returns the fraction of horizon steps that are predicted positive
        (> 0).  > 0.5 means the ARF ensemble expects price to be HIGHER at
        horizon end on average; < 0.5 means LOWER.
        """
        preds    = self.predict(X)                        # (n, horizon)
        frac_pos = (preds > 0).mean(axis=1)               # (n,) in [0, 1]
        return frac_pos.astype(np.float32)
