from __future__ import annotations

import numpy as np
import pandas as pd


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns, rolling volatility, SMA ratios, and RSI to df.

    Expects a 'close' column; returns a copy with new columns appended.
    """
    df = df.copy()
    c = df["close"]
    # Guard against concurrent yfinance calls that occasionally return a
    # DataFrame instead of a Series for a single column (thread-safety issue).
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]

    df["logret"] = np.log(c / c.shift(1))
    df["vol_20"] = df["logret"].rolling(20).std()

    df["sma_50"] = c.rolling(50).mean()
    df["sma_200"] = c.rolling(200).mean()
    df["sma_ratio_50"] = c / df["sma_50"]
    df["sma_ratio_200"] = c / df["sma_200"]

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    return df
