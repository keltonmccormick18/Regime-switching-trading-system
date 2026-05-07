"""Feature engineering pipeline.

build_features() is the single entry point for the training pipeline.
It accepts raw OHLCV data and returns a DataFrame with all model inputs
and the next-day close price as the prediction target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.rolling_stats import add_rolling_stats
from src.features.tda import add_tda_features

# Ordered list of base feature columns consumed by all models.
# When use_tda=False, tda_l1/tda_l2 are filled with 0.0 instead of NaN.
FEATURE_COLS = [
    "logret",
    "vol_20",
    "rsi_14",
    "sma_ratio_50",
    "sma_ratio_200",
    "tda_l1",
    "tda_l2",
]

# B1: Cross-asset macro features (appended when use_macro=True).
MACRO_FEATURE_COLS = [
    "vix_pct",          # VIX as expanding percentile [0,1] — implied vol level
    "vix_ret",          # VIX daily log return — fear-spike signal
    "credit_spread",    # log(HYG/LQD) — credit risk proxy
    "credit_spread_ret",# daily change in credit spread
    "dxy_ret",          # UUP log return — USD strength
]

# Full feature list when macro features are enabled (7 + 5 = 12)
FEATURE_COLS_MACRO = FEATURE_COLS + MACRO_FEATURE_COLS


def build_features(
    df: pd.DataFrame,
    tda_window: int = 100,
    use_tda: bool = True,
    macro_df: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Build the full feature set from raw OHLCV data.

    Input:
        df        — DataFrame with [open, high, low, close, volume] indexed by date.
        tda_window — Rolling window size passed to the TDA computation.
        use_tda   — When False, tda_l1/tda_l2 are set to 0 (fast path).
        macro_df  — B1: Optional macro DataFrame returned by load_macro_data().
                    When provided, the five macro feature columns are merged in
                    and forward-filled; missing days default to 0.

    Output:
        DataFrame containing FEATURE_COLS (or FEATURE_COLS_MACRO when macro_df
        is given) + 'close' + 'target' (next-day close).
    """
    df = add_rolling_stats(df)

    if use_tda:
        df = add_tda_features(df, window=tda_window)
        required = FEATURE_COLS + ["target"]
    else:
        df = df.copy()
        df["tda_l1"] = 0.0
        df["tda_l2"] = 0.0
        required = [c for c in FEATURE_COLS if c not in ("tda_l1", "tda_l2")] + ["target"]

    # ── B1: merge macro features ───────────────────────────────────────────────
    if macro_df is not None and not macro_df.empty:
        macro_aligned = (
            macro_df[MACRO_FEATURE_COLS]
            .reindex(df.index)
            .ffill()
            .fillna(0.0)
        )
        df = df.join(macro_aligned, how="left")
        df[MACRO_FEATURE_COLS] = df[MACRO_FEATURE_COLS].fillna(0.0)
        # required stays the same — macro cols are fully filled so no NaN risk

    df["target"] = df["close"].shift(-1)
    return df.dropna(subset=required)
