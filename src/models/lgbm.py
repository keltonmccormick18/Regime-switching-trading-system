"""LightGBM model — HIGH_VOL_BEAR regime.

Gradient-boosted trees handle fast, non-stationary bear-market dynamics better
than neural nets: no smoothness assumption, robust to outliers, trains in
seconds (enabling frequent retraining), and exposes feature importances.

The sequence input (n_samples, n_features, lookback) is converted to a flat
feature vector enriched with per-feature summary statistics before passing to
a MultiOutputRegressor wrapping LGBMRegressor.

Input shape:  (n_samples, n_features, lookback)
Output shape: (n_samples, horizon)
"""
from __future__ import annotations

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False


def _check_lgb() -> None:
    if not _LGB_AVAILABLE:
        raise ImportError("lightgbm is required: pip install lightgbm")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _featurize(X: np.ndarray) -> np.ndarray:
    """Flatten sequence + add summary statistics per feature channel.

    Input:  (n, n_features, lookback)
    Output: (n, n_features * lookback + n_features * 5)
             = flat sequence + [mean, std, last, first, momentum] per feature
    """
    n, f, l = X.shape
    flat     = X.reshape(n, f * l)                        # raw sequence
    means    = X.mean(axis=2)                             # (n, f)
    stds     = X.std(axis=2)                              # (n, f)
    last     = X[:, :, -1]                                # (n, f)
    first    = X[:, :, 0]                                 # (n, f)
    momentum = last - first                               # (n, f) total change
    return np.concatenate([flat, means, stds, last, first, momentum], axis=1)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class LightGBMModel(BaseModel):
    """LightGBM multi-output price forecaster — used for HIGH_VOL_BEAR regime.

    Uses MultiOutputRegressor so each horizon step gets its own estimator,
    capturing the distinct dynamics at different forecast horizons.

    Attributes
    ----------
    feature_importances_ : np.ndarray | None
        Per-feature importance scores (mean across horizon estimators) after
        fitting.  Shape: (n_features * lookback + n_features * 5,).
    """

    def __init__(
        self,
        n_features:    int   = 7,
        lookback:      int   = 64,
        horizon:       int   = 16,
        n_estimators:  int   = 300,
        learning_rate: float = 0.05,
        num_leaves:    int   = 31,
        max_depth:     int   = -1,
        min_child_samples: int = 20,
        subsample:     float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha:     float = 0.1,
        reg_lambda:    float = 0.1,
        n_jobs:        int   = -1,
        # Accepted for API compatibility with train_model() kwargs
        epochs:        int   = 0,
    ):
        _check_lgb()
        self.n_features       = n_features
        self.lookback         = lookback
        self.horizon          = horizon
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.num_leaves       = num_leaves
        self.max_depth        = max_depth
        self.min_child_samples = min_child_samples
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha        = reg_alpha
        self.reg_lambda       = reg_lambda
        self.n_jobs           = n_jobs

        self.scaler = StandardScaler()
        self.model: MultiOutputRegressor | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_flat = np.asarray(_featurize(X), dtype=np.float32)
        X_flat = np.asarray(self.scaler.fit_transform(X_flat), dtype=np.float32)
        # Scale min_child_samples to dataset size: at least 2, at most the constructor value.
        # Prevents over-restriction on small regime-filtered training sets.
        adaptive_min_child = max(2, min(self.min_child_samples, len(X_flat) // 10))
        estimator = lgb.LGBMRegressor(
            n_estimators      = self.n_estimators,
            learning_rate     = self.learning_rate,
            num_leaves        = self.num_leaves,
            max_depth         = self.max_depth,
            min_child_samples = adaptive_min_child,
            subsample         = self.subsample,
            colsample_bytree  = self.colsample_bytree,
            reg_alpha         = self.reg_alpha,
            reg_lambda        = self.reg_lambda,
            n_jobs            = self.n_jobs,
            verbose           = -1,
        )
        self.model = MultiOutputRegressor(estimator, n_jobs=1)
        self.model.fit(X_flat, y)
        # Aggregate feature importances across horizon estimators
        imps = np.array([e.feature_importances_ for e in self.model.estimators_])
        self.feature_importances_ = imps.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = np.asarray(_featurize(X), dtype=np.float32)
        X_flat = np.asarray(self.scaler.transform(X_flat), dtype=np.float32)
        return self.model.predict(X_flat).astype(np.float32)
