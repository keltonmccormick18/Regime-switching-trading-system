"""Vectorized backtest engine.

Converts a signal series into strategy returns, then computes a full suite
of performance metrics.  Designed to be called from the training pipeline
and the /metrics API endpoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    sharpe: float
    sortino: float
    total_return: float          # fractional, e.g. 0.25 = +25 %
    max_drawdown: float          # fractional, negative, e.g. -0.15 = -15 %
    win_rate: float              # fractional
    n_trades: int
    equity: list[float]          # equity curve starting at 1.0
    daily_returns: list[float]
    calmar: float = 0.0
    avg_trade_return: float = 0.0
    notes: str = ""
    # Risk/benchmark metrics (vs asset buy-and-hold)
    alpha: float = 0.0           # annualised Jensen's alpha
    beta: float = 0.0            # market beta vs buy-and-hold
    # Per-bar regime labels aligned to the equity curve
    regime_curve: list = field(default_factory=list)
    # Buy-and-hold benchmark (same asset, same period, fully invested day 1)
    bah_equity: list[float] = field(default_factory=list)
    bah_return: float = 0.0
    # Bar indices where buy / sell orders were generated (for chart markers)
    buy_bars:  list[int] = field(default_factory=list)
    sell_bars: list[int] = field(default_factory=list)
    # Bar indices where strategy exits a long to go flat (long → cash transitions)
    exit_bars: list[int] = field(default_factory=list)
    # Full per-bar signal array (1=long, 0=flat/cash, -1=short)
    signal_series: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "total_return": round(self.total_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "n_trades": self.n_trades,
            "calmar": round(self.calmar, 4),
            "avg_trade_return": round(self.avg_trade_return, 4),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "equity": [round(v, 6) for v in self.equity],
            "daily_returns": [round(v, 6) for v in self.daily_returns],
            "regime_curve": self.regime_curve,
            "bah_equity": [round(v, 6) for v in self.bah_equity],
            "bah_return": round(self.bah_return, 6),
            "buy_bars":   self.buy_bars,
            "sell_bars":  self.sell_bars,
            "exit_bars":  self.exit_bars,
            "signal_series": self.signal_series,
            "notes": self.notes,
        }


def backtest(
    df: pd.DataFrame,
    signals: Optional[pd.Series] = None,
    price_col: str = "close",
    signal_col: str = "signal",
    transaction_cost: float = 0.001,   # 10 bps per trade (one-way)
    annualization: int = 252,
) -> BacktestResult:
    """
    Run a vectorized backtest.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `price_col`. May also contain `signal_col` (overridden
        by the `signals` argument if provided).
    signals : pd.Series, optional
        Signal series aligned to df's index. Values should be -1, 0, or 1.
        If None, `df[signal_col]` is used.
    price_col : str
        Column name for price data.
    signal_col : str
        Column name for pre-existing signal in df (used when signals=None).
    transaction_cost : float
        One-way cost fraction applied whenever the signal changes.
    annualization : int
        Trading days per year for Sharpe/Sortino calculation.

    Returns
    -------
    BacktestResult
    """
    work = df[[price_col]].copy()

    if signals is not None:
        work["signal"] = signals.values
    elif signal_col in df.columns:
        work["signal"] = df[signal_col].values
    else:
        raise ValueError(
            f"No signal provided and '{signal_col}' not found in df columns."
        )

    work["signal"] = work["signal"].fillna(0).clip(-1, 1)

    # ── daily price returns ──────────────────────────────────────────────
    work["returns"] = work[price_col].pct_change().fillna(0.0)

    # ── strategy returns: signal from *previous* bar (no look-ahead) ─────
    work["pos"] = work["signal"].shift(1).fillna(0.0)
    work["strat_returns"] = work["pos"] * work["returns"]

    # ── transaction costs ────────────────────────────────────────────────
    work["trade"] = work["pos"].diff().abs()          # 0 or 2 on a flip, 1 on open/close
    work["cost"] = work["trade"] * transaction_cost
    work["strat_returns"] -= work["cost"]

    # ── equity curve ─────────────────────────────────────────────────────
    equity = (1 + work["strat_returns"]).cumprod()
    equity = equity.fillna(1.0)

    # ── drawdown ─────────────────────────────────────────────────────────
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max.replace(0, np.nan)
    max_drawdown = float(drawdown.min())

    # ── Sharpe ───────────────────────────────────────────────────────────
    r = work["strat_returns"]
    mean_r = float(r.mean())
    std_r = float(r.std())
    sharpe = (mean_r / std_r * math.sqrt(annualization)) if std_r > 1e-12 else 0.0

    # ── Sortino ──────────────────────────────────────────────────────────
    downside = r[r < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 1e-12
    sortino = (mean_r / downside_std * math.sqrt(annualization)) if downside_std > 1e-12 else 0.0

    # ── trade-level stats ─────────────────────────────────────────────────
    trade_returns = _extract_trade_returns(work["pos"], work["returns"])
    n_trades = len(trade_returns)
    win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0.0
    avg_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0

    # ── Calmar ───────────────────────────────────────────────────────────
    total_return = float(equity.iloc[-1]) - 1.0
    calmar = (total_return / abs(max_drawdown)) if max_drawdown < -1e-12 else 0.0

    return BacktestResult(
        sharpe=sharpe,
        sortino=sortino,
        total_return=total_return,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        n_trades=n_trades,
        equity=equity.tolist(),
        daily_returns=r.tolist(),
        calmar=calmar,
        avg_trade_return=avg_trade_return,
    )


def _extract_trade_returns(
    positions: pd.Series,
    returns: pd.Series,
) -> list[float]:
    """
    Extract per-trade returns.

    A "trade" is defined as a continuous run of non-zero position.
    Return = cumulative strategy return over that holding period.
    """
    trade_returns: list[float] = []
    in_trade = False
    trade_ret = 0.0

    pos_arr = positions.values
    ret_arr = returns.values

    prev_p = pos_arr[0] if len(pos_arr) > 0 else 0

    for i in range(1, len(pos_arr)):
        p = pos_arr[i - 1]   # signal from prev bar drives this bar's return
        r = ret_arr[i] * p   # strategy return this bar

        if p != 0:
            # Cut trade on direction flip (e.g. long → short without going flat)
            if in_trade and p != prev_p:
                trade_returns.append(trade_ret)
                trade_ret = 0.0
            trade_ret += r
            in_trade = True
        else:
            if in_trade:
                trade_returns.append(trade_ret)
                trade_ret = 0.0
                in_trade = False

        prev_p = p

    if in_trade and trade_ret != 0.0:
        trade_returns.append(trade_ret)

    return trade_returns
