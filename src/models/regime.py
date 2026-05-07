"""Market regime detection and model routing.

Regime classification uses two independent signals:
  - Volatility: TDA L1 norm (when available) or 20-day rolling vol vs. a percentile threshold.
  - Trend:      Price relative to 200-day SMA (above = bull, below = bear).

REGIME_MODELS routing:
  HIGH_VOL_BULL → TFTModel     (VSN + LSTM + attention + quantile heads; best for volatile uptrends)
  HIGH_VOL_BEAR → OnlineModel  (River ARF with ADWIN drift detection; adapts to fast bear dynamics)
  LOW_VOL_BULL  → TCNModel     (pure TCN; low-vol uptrends are smooth and regular)
  LOW_VOL_BEAR  → TCNLSTMModel (TCN + LSTM; captures sequential dependencies in ranging markets)

All models are wrapped in ConformalWrapper at training time (see train.py),
which provides calibrated prediction intervals and conformal confidence scores
that flow into position sizing via the signal generator.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from src.models.tcn import TCNModel
from src.models.tcn_lstm import TCNLSTMModel
from src.models.tft import TFTModel
from src.models.online import OnlineModel


class Regime(Enum):
    HIGH_VOL_BULL = "high_vol_bull"
    HIGH_VOL_BEAR = "high_vol_bear"
    LOW_VOL_BULL  = "low_vol_bull"
    LOW_VOL_BEAR  = "low_vol_bear"


def detect_regime(df: pd.DataFrame, vol_percentile: float = 0.70) -> Regime:
    """Classify the current market regime from the most recent row of a feature DataFrame.

    Args:
        df:             Feature DataFrame produced by build_features().
        vol_percentile: Fraction threshold above which volatility is considered "high".

    Returns:
        A Regime enum value.
    """
    last = df.iloc[-1]

    # --- Volatility classification ---
    # Prefer TDA L1 norm if it has been computed (non-zero, non-NaN values present).
    tda_series = df["tda_l1"].replace(0.0, np.nan).dropna()
    if len(tda_series) > 10:
        is_high_vol = last["tda_l1"] > tda_series.quantile(vol_percentile)
    else:
        vol_series = df["vol_20"].dropna()
        is_high_vol = last["vol_20"] > vol_series.quantile(vol_percentile)

    # --- Trend classification ---
    # sma_ratio_200 > 1 means close > 200-day SMA → uptrend
    is_bull = last["sma_ratio_200"] > 1.0

    if is_high_vol and is_bull:
        return Regime.HIGH_VOL_BULL
    elif is_high_vol:
        return Regime.HIGH_VOL_BEAR
    elif is_bull:
        return Regime.LOW_VOL_BULL
    else:
        return Regime.LOW_VOL_BEAR


REGIME_MODELS: dict[Regime, type] = {
    Regime.HIGH_VOL_BULL: TFTModel,
    Regime.HIGH_VOL_BEAR: OnlineModel,   # A2: adaptive ARF replaces static LightGBM
    Regime.LOW_VOL_BULL:  TCNModel,
    Regime.LOW_VOL_BEAR:  TCNLSTMModel,
}
