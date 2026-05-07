from __future__ import annotations

import numpy as np

from src.models.base import BaseModel


def predict(model: BaseModel, X: np.ndarray) -> np.ndarray:
    """Run inference and return the target array.

    Args:
        model: A fitted BaseModel instance.
        X:     Input array of shape (n_samples, n_features, lookback).

    Returns:
        target — numpy array of shape (n_samples, horizon) containing predicted future close prices.
    """
    return model.predict(X)
