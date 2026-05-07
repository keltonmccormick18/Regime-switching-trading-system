"""TDA volatility features via persistent homology (Vietoris-Rips + persistence landscapes).

Ported from TDA-basics: constructs distance matrices from rolling windows of log-returns,
computes H1 persistence diagrams, and returns L1/L2 landscape norms as time-series features.

gudhi and persim are optional — if not installed, tda columns are silently left as NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gudhi as gd
    from persim import PersLandscapeExact
    from scipy.spatial.distance import pdist, squareform
    _HAS_TDA = True
except ImportError:
    _HAS_TDA = False


def compute_tda_norms(
    log_returns: np.ndarray,
    window: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling persistence landscape L1/L2 norms over log-return windows.

    Args:
        log_returns: 1-D array of daily log returns.
        window:      Number of days per rolling window (matches TDA-basics default of 100).

    Returns:
        l1_norms, l2_norms — arrays of length len(log_returns); NaN for first `window` entries.
    """
    n = len(log_returns)
    l1 = np.full(n, np.nan)
    l2 = np.full(n, np.nan)

    if not _HAS_TDA:
        return l1, l2

    for i in range(window, n):
        seg = log_returns[i - window:i].reshape(-1, 1)
        D = squareform(pdist(seg, metric="euclidean"))
        rc = gd.RipsComplex(distance_matrix=D, max_edge_length=1.0)
        st = rc.create_simplex_tree(max_dimension=2)
        diag = st.persistence()
        pts_h1 = np.array([pt[1] for pt in diag if pt[0] == 1])
        if len(pts_h1) == 0:
            l1[i] = 0.0
            l2[i] = 0.0
            continue
        pl = PersLandscapeExact([pts_h1])
        l1[i] = pl.p_norm(p=1)
        l2[i] = pl.p_norm(p=2)

    return l1, l2


def add_tda_features(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Append tda_l1 and tda_l2 columns to df.

    Requires a 'logret' column (added by add_rolling_stats); computes it from 'close' if missing.
    """
    df = df.copy()
    if "logret" not in df.columns:
        df["logret"] = np.log(df["close"] / df["close"].shift(1))
    log_returns = df["logret"].fillna(0).values
    l1, l2 = compute_tda_norms(log_returns, window=window)
    df["tda_l1"] = l1
    df["tda_l2"] = l2
    return df
