from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Common interface for all regime-specific forecasting models.

    X shape: (n_samples, n_features, lookback)  — channels-first (Conv1d convention)
    y shape: (n_samples, horizon)               — future close prices
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted target array of shape (n_samples, horizon)."""
        ...
