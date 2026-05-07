"""OHLCV resampling utilities — polars (primary) and pandas (fallback).

Resampling collapses higher-frequency OHLCV bars into a lower-frequency target:
    open   → first bar's open
    high   → max across all bars in the window
    low    → min across all bars in the window
    close  → last bar's close
    volume → sum of all bars in the window

Polars interval strings understood by this module:
    "1m", "2m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"

These map 1-to-1 to polars duration literals (m = minutes, h = hours, d = days, w = weeks).
"""
from __future__ import annotations

import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Polars (primary)
# ---------------------------------------------------------------------------

# Canonical interval labels → polars duration strings
_TO_POLARS: dict[str, str] = {
    "1m":  "1m",
    "2m":  "2m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "6h":  "6h",
    "12h": "12h",
    "1d":  "1d",
    "1w":  "1w",
}


def resample_ohlcv(df: pl.DataFrame, to_interval: str) -> pl.DataFrame:
    """Resample a polars OHLCV DataFrame to a coarser interval.

    Args:
        df:          polars DataFrame with a 'timestamp' column (Datetime) and
                     columns: open, high, low, close, volume.
        to_interval: Target interval string, e.g. "5m", "1h", "1d".

    Returns:
        Resampled polars DataFrame with the same column schema, sorted by timestamp.
    """
    duration = _resolve_polars_duration(to_interval)
    return (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every=duration)
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
        .sort("timestamp")
    )


def resample_multi(
    df: pl.DataFrame,
    intervals: list[str],
) -> dict[str, pl.DataFrame]:
    """Resample a single base DataFrame to multiple target intervals at once.

    Returns a dict mapping interval string → resampled DataFrame.
    """
    return {interval: resample_ohlcv(df, interval) for interval in intervals}


# ---------------------------------------------------------------------------
# Pandas fallback
# ---------------------------------------------------------------------------

# Canonical interval labels → pandas offset aliases
_TO_PANDAS: dict[str, str] = {
    "1m":  "1min",
    "2m":  "2min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "6h":  "6h",
    "12h": "12h",
    "1d":  "1D",
    "1w":  "1W",
}


def resample_ohlcv_pd(df: pd.DataFrame, to_interval: str) -> pd.DataFrame:
    """Resample a pandas OHLCV DataFrame to a coarser interval.

    Args:
        df:          pandas DataFrame indexed by timestamp (DatetimeIndex) with
                     columns: open, high, low, close, volume.
        to_interval: Target interval string, e.g. "5m", "1h", "1d".

    Returns:
        Resampled pandas DataFrame with a DatetimeIndex.
    """
    rule = _TO_PANDAS.get(to_interval, to_interval)
    agg = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    })
    return agg.dropna(subset=["open", "close"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_polars_duration(interval: str) -> str:
    if interval in _TO_POLARS:
        return _TO_POLARS[interval]
    # Pass through if the caller already used polars notation
    return interval


def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a polars OHLCV DataFrame to a pandas DataFrame with DatetimeIndex."""
    pdf = df.to_pandas()
    pdf = pdf.set_index("timestamp")
    pdf.index = pd.to_datetime(pdf.index, utc=True)
    return pdf


def pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert a pandas OHLCV DataFrame (DatetimeIndex) to polars."""
    df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    return pl.from_pandas(df).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )
