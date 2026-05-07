from __future__ import annotations

import numpy as np

from src.models.base import BaseModel
from src.models.conformal import ConformalWrapper


def train_model(
    model_class: type[BaseModel],
    X: np.ndarray,
    y: np.ndarray,
    conformal: bool = True,
    conformal_alpha: float = 0.10,
    conformal_cal_ratio: float = 0.20,
    **model_kwargs,
) -> BaseModel:
    """Instantiate, fit, and optionally wrap a model with conformal prediction.

    Args:
        model_class:         A BaseModel subclass (TFTModel, LightGBMModel, TCNModel, …).
        X:                   Training input  (n_samples, n_features, lookback).
        y:                   Training target (n_samples, horizon) — future close prices.
        conformal:           Wrap result in ConformalWrapper (default True).
        conformal_alpha:     Miscoverage level for prediction intervals (default 0.10 = 90%).
        conformal_cal_ratio: Fraction of training data used for calibration (default 0.20).
        **model_kwargs:      Forwarded to model_class.__init__ (n_features, lookback, horizon, …).

    Returns:
        Fitted model — ConformalWrapper(base_model) when conformal=True, else raw base model.
    """
    model = model_class(**model_kwargs)

    if conformal:
        wrapped = ConformalWrapper(model, alpha=conformal_alpha, cal_ratio=conformal_cal_ratio)
        wrapped.fit(X, y)
        return wrapped

    model.fit(X, y)
    return model
