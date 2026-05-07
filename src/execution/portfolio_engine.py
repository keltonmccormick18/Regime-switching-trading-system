"""Shared-capital multi-asset portfolio backtest engine.

Replaces the independent per-ticker equity-curve blending with a true
single-loop simulation: all tickers share one capital pool, fills carry
real cost consequences, and position sizing is portfolio-aware.

Three allocation features
-------------------------
1. Inverse-vol weighting
   Target allocation ∝ 1/vol_20 so high-volatility assets receive less
   capital.  Weights are re-normalised at every prediction bar.

2. Regime-conditional shifts
   Each ticker's raw allocation is multiplied by 0.5 when its current
   regime is HIGH_VOL_BULL or HIGH_VOL_BEAR, then re-normalised.  Capital
   naturally flows toward low-vol / defensive assets during turbulent
   markets without any explicit rule-based override.

3. Periodic rebalancing
   At a configured calendar interval (monthly / quarterly / annual),
   current position weights are compared to target weights and orders are
   generated to close the drift.  Commission and slippage apply to every
   rebalance fill, so the friction cost is realistic.

Usage::

    engine = SharedCapitalPortfolioEngine(
        tickers        = ["SPY", "GLD", "TLT"],
        base_weights   = {"SPY": 0.4, "GLD": 0.3, "TLT": 0.3},
        fitted_models  = {"SPY": {"LOW_VOL_BULL": m1, ...}, ...},
        feat_dfs       = {"SPY": df_spy, ...},   # full df, DatetimeIndex
        train_splits   = {"SPY": 1200, ...},      # index in feat_df
        broker         = SimulatedBroker.retail(),
        rebalance_freq = "quarterly",
        vol_weight     = True,
        regime_shift   = True,
    )
    result = engine.run()
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.execution.broker import SimulatedBroker
from src.execution.order import Fill, Order


# ─────────────────────────── Constants ────────────────────────────────────

_HIGH_VOL_REGIMES: frozenset[str] = frozenset({"HIGH_VOL_BULL", "HIGH_VOL_BEAR"})

_REBALANCE_BARS: dict[str, int] = {
    "never":     0,
    "monthly":   21,
    "quarterly": 63,
    "annual":    252,
}

# Mirror of _REGIME_SIGNAL_THRESHOLDS from backtest_engine (avoids circular import)
_REGIME_SIGNAL_THRESHOLDS: dict[str, float] = {
    "HIGH_VOL_BULL": 0.008,
    "HIGH_VOL_BEAR": 0.005,
    "LOW_VOL_BULL":  0.002,
    "LOW_VOL_BEAR":  0.003,
}


# ─────────────────────────── Output types ─────────────────────────────────

@dataclass
class TickerPortfolioStats:
    ticker:       str
    weight:       float
    equity:       list[float]     # per-bar equity attribution for this ticker
    fills:        list[Fill]
    regime_curve: list[str]
    sharpe:       float = 0.0
    sortino:      float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate:     float = 0.0
    n_trades:     int   = 0
    calmar:       float = 0.0
    alpha:        float = 0.0
    beta:         float = 0.0


@dataclass
class PortfolioEngineResult:
    portfolio_equity: list[float]
    ticker_stats:     list[TickerPortfolioStats]
    rebalance_bars:   list[int]
    initial_capital:  float


# ─────────────────────────── Engine ───────────────────────────────────────

class SharedCapitalPortfolioEngine:
    """Single-loop shared-capital portfolio backtest.

    Parameters
    ----------
    tickers
        Ordered list of ticker symbols to simulate.
    base_weights
        Dict ticker → target fraction (must sum to ≤ 1).  Used as fallback
        when vol_weight=False and as the seed allocation for all tickers.
    fitted_models
        Dict ticker → {regime_str → fitted model}.  All four regime strings
        should be present; missing ones fall back to the first available.
    feat_dfs
        Dict ticker → full feature DataFrame *with DatetimeIndex preserved*.
        Must include FEATURE_COLS + 'close'.  'open' is used for fill prices
        when present, otherwise 'close' is used.
    train_splits
        Dict ticker → integer position in feat_df where out-of-sample begins
        (i.e. the first bar the backtest may trade on).
    broker
        SimulatedBroker instance.  Shared across all tickers and all fills.
    initial_capital
        Starting cash balance.
    lookback
        Feature window length in bars.  Must match the trained models.
    horizon
        Prediction horizon in bars.  Must match the trained models.
    vol_weight
        True → inverse-vol allocation weighted by 1/vol_20.
        False → use base_weights directly.
    regime_shift
        True → multiply allocation by 0.5 for any ticker whose current
        regime is HIGH_VOL_BULL or HIGH_VOL_BEAR, then re-normalise.
    rebalance_freq
        "never" | "monthly" | "quarterly" | "annual".
    annualisation
        Trading days per year for Sharpe / Sortino calculations.
    """

    def __init__(
        self,
        tickers:         list[str],
        base_weights:    dict[str, float],
        fitted_models:   dict[str, dict[str, Any]],
        feat_dfs:        dict[str, pd.DataFrame],
        train_splits:    dict[str, int],
        broker:          SimulatedBroker,
        initial_capital: float = 100_000.0,
        lookback:        int   = 64,
        horizon:         int   = 16,
        vol_weight:      bool  = True,
        regime_shift:    bool  = True,
        rebalance_freq:  str   = "never",
        long_only:       bool  = True,
        annualisation:   int   = 252,
    ) -> None:
        self.tickers         = tickers
        self.base_weights    = base_weights
        self.fitted_models   = fitted_models
        self.feat_dfs        = feat_dfs
        self.train_splits    = train_splits
        self.broker          = broker
        self.initial_capital = initial_capital
        self.lookback        = lookback
        self.horizon         = horizon
        self.vol_weight      = vol_weight
        self.regime_shift    = regime_shift
        self.rebalance_freq  = rebalance_freq
        self.long_only       = long_only
        self.annualisation   = annualisation

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> PortfolioEngineResult:
        from src.features.pipeline import FEATURE_COLS, MACRO_FEATURE_COLS
        from src.execution.backtest_engine import _regime_series
        from src.strategy.signals import SignalGenerator

        # ── 1. Build per-ticker test windows ──────────────────────────────
        # Include `lookback` bars before the OOS split so the first prediction
        # can immediately look back into real data rather than zeros.
        test_dfs: dict[str, pd.DataFrame] = {}
        for t in self.tickers:
            feat_df = self.feat_dfs[t]
            split   = self.train_splits[t]
            start   = max(0, split - self.lookback)
            test_dfs[t] = feat_df.iloc[start:]          # DatetimeIndex intact

        # ── 2. Align to common date range ─────────────────────────────────
        try:
            common = sorted(
                set.intersection(*[set(df.index) for df in test_dfs.values()])
            )
        except TypeError:
            # Non-datetime index — fall back to positional alignment
            min_len = min(len(df) for df in test_dfs.values())
            common  = list(range(min_len))

        if len(common) < self.lookback + 10:
            raise ValueError(
                f"Only {len(common)} common bars after aligning "
                f"{len(self.tickers)} tickers — need {self.lookback + 10}."
            )

        # ── 3. Build numpy arrays for O(1) bar access ──────────────────────
        all_possible  = FEATURE_COLS + MACRO_FEATURE_COLS
        feat_arrays:   dict[str, np.ndarray] = {}
        close_arrays:  dict[str, np.ndarray] = {}
        open_arrays:   dict[str, np.ndarray] = {}
        vol_arrays:    dict[str, np.ndarray] = {}
        sma_arrays:    dict[str, np.ndarray] = {}
        regime_arrays: dict[str, list[str]]  = {}

        for t in self.tickers:
            df = test_dfs[t].reindex(common, method="ffill")
            feat_cols = [c for c in all_possible if c in df.columns]
            feat_arrays[t]  = df[feat_cols].values.astype(np.float32)
            close_arrays[t] = df["close"].values.astype(np.float64)
            open_arrays[t]  = (
                df["open"].values.astype(np.float64)
                if "open" in df.columns else close_arrays[t]
            )
            vol_arrays[t] = (
                df["vol_20"].values.astype(np.float64)
                if "vol_20" in df.columns
                else np.full(len(common), 0.01)
            )
            sma_arrays[t] = (
                df["sma_ratio_200"].values.astype(np.float64)
                if "sma_ratio_200" in df.columns
                else np.ones(len(common))
            )
            regime_arrays[t] = _regime_series(df)

        n_bars = len(common)

        # ── 4. Shared portfolio state ──────────────────────────────────────
        cash      = self.initial_capital
        positions = {t: 0.0 for t in self.tickers}   # signed quantity
        signals   = {t: 0   for t in self.tickers}

        # Per-ticker virtual cash: starts at initial allocation, then
        # updated on every fill so that ticker_equity[t] = ticker_cash[t]
        # + positions[t]*price[t] gives a meaningful P&L attribution.
        # Invariant: sum(ticker_cash.values()) == cash (always).
        ticker_cash: dict[str, float] = {
            t: self.base_weights.get(t, 1.0 / len(self.tickers)) * self.initial_capital
            for t in self.tickers
        }
        ticker_fills:    dict[str, list[Fill]] = {t: [] for t in self.tickers}
        ticker_regimes:  dict[str, list[str]]  = {t: [] for t in self.tickers}
        ticker_equity_h: dict[str, list[float]] = {t: [] for t in self.tickers}

        portfolio_equity: list[float] = []
        pending: list[tuple[int, str, Order]] = []   # (fill_bar, ticker, order)
        rebalance_bars: list[int] = []

        rebalance_interval = _REBALANCE_BARS.get(self.rebalance_freq, 0)
        last_rebalance     = 0
        sig_gen            = SignalGenerator()

        # ── 5. Bar-by-bar simulation loop ──────────────────────────────────
        for i in range(n_bars):

            # ── a. Process fills due this bar (execution lag = 1) ──────────
            still_pending: list[tuple[int, str, Order]] = []
            for fill_bar, t, order in pending:
                if i >= fill_bar:
                    open_p = float(open_arrays[t][i])
                    if open_p <= 0:
                        continue
                    ann_vol = float(vol_arrays[t][i]) * math.sqrt(self.annualisation)
                    report  = self.broker.submit(
                        order,
                        market_price = open_p,
                        volatility   = ann_vol,
                        adv          = 0.0,
                        cash         = cash,
                    )
                    if report.is_filled or report.status == "partial":
                        fill = report.fill
                        direction = -1 if fill.side == "buy" else 1
                        cash           += direction * fill.fill_price * fill.quantity
                        cash           -= fill.commission
                        ticker_cash[t] += direction * fill.fill_price * fill.quantity
                        ticker_cash[t] -= fill.commission
                        positions[t]   += (
                            fill.quantity if fill.side == "buy" else -fill.quantity
                        )
                        ticker_fills[t].append(fill)
                else:
                    still_pending.append((fill_bar, t, order))
            pending = still_pending

            # ── b. Mark to market ──────────────────────────────────────────
            for t in self.tickers:
                cp = float(close_arrays[t][i])
                ticker_equity_h[t].append(ticker_cash[t] + positions[t] * cp)
                ticker_regimes[t].append(
                    regime_arrays[t][i] if i < len(regime_arrays[t]) else ""
                )
            equity = cash + sum(
                positions[t] * float(close_arrays[t][i]) for t in self.tickers
            )
            portfolio_equity.append(equity)

            # ── c. Need full lookback window before predicting ─────────────
            if i < self.lookback:
                continue

            # ── d. Generate signals for all tickers ────────────────────────
            new_signals: dict[str, int] = {}
            for t in self.tickers:
                regime = (
                    regime_arrays[t][i] if i < len(regime_arrays[t]) else ""
                )
                sig_gen.min_return_threshold = _REGIME_SIGNAL_THRESHOLDS.get(
                    regime, 0.005
                )

                window = feat_arrays[t][i - self.lookback:i].T   # (n_feat, lb)
                X      = window[np.newaxis, :, :]                  # (1, n_feat, lb)

                models = self.fitted_models.get(t, {})
                model  = (
                    models.get(regime) or next(iter(models.values()), None)
                )
                if model is None:
                    new_signals[t] = 0
                    continue

                pred_returns = model.predict(X)[0]                 # (horizon,)
                cp           = float(close_arrays[t][i])
                pred_prices  = cp * (1.0 + pred_returns)

                sig = sig_gen.from_price_prediction(
                    symbol        = t,
                    prediction    = pred_prices,
                    current_price = cp,
                    regime        = regime,
                )
                raw_sig = 0 if sig.is_flat else (1 if sig.is_long else -1)
                new_signals[t] = max(0, raw_sig) if self.long_only else raw_sig

            # ── e. Compute target allocations ──────────────────────────────
            regime_multipliers: dict[str, float] = {}
            for t in self.tickers:
                r = regime_arrays[t][i] if i < len(regime_arrays[t]) else ""
                regime_multipliers[t] = (
                    0.5
                    if (self.regime_shift and r in _HIGH_VOL_REGIMES)
                    else 1.0
                )

            target_weights = _compute_target_weights(
                signals     = new_signals,
                vol_values  = {t: float(vol_arrays[t][i]) for t in self.tickers},
                sma_values  = {t: float(sma_arrays[t][i]) for t in self.tickers},
                base_weights= self.base_weights,
                vol_weight  = self.vol_weight,
                multipliers = regime_multipliers,
            )

            # ── f. Decide whether to trade ─────────────────────────────────
            changed_tickers = {t for t in self.tickers if new_signals[t] != signals[t]}
            should_rebalance = (
                rebalance_interval > 0
                and (i - last_rebalance) >= rebalance_interval
            )

            if changed_tickers or should_rebalance:
                if should_rebalance:
                    rebalance_bars.append(i)
                    last_rebalance = i
                    # On a calendar rebalance, touch every ticker.
                    # On a signal-only event, only touch the tickers whose
                    # signal changed (plus any ticker that needs to be closed).
                    tickers_to_touch = set(self.tickers)
                else:
                    # Signal-driven: close tickers that went flat, open/resize
                    # tickers whose signal changed.  Leave untouched tickers alone
                    # so their positions don't get inadvertently resized.
                    tickers_to_touch = changed_tickers | {
                        t for t in self.tickers if new_signals[t] == 0 and positions[t] != 0
                    }

                for t in tickers_to_touch:
                    cp = float(close_arrays[t][i])
                    if cp <= 0 or equity <= 0:
                        continue

                    sig_dir     = new_signals[t]
                    tw          = target_weights.get(t, 0.0)
                    current_pos = positions[t]

                    if sig_dir == 0:
                        target_qty = 0.0
                    else:
                        target_qty = math.copysign(tw * equity / cp, sig_dir)

                    diff_qty = target_qty - current_pos
                    if abs(diff_qty) < 0.01:
                        continue

                    side = "buy" if diff_qty > 0 else "sell"
                    order = Order(
                        symbol   = t,
                        side     = side,
                        quantity = abs(diff_qty),
                        strategy = "rebalance" if should_rebalance else "signal",
                        regime   = (
                            regime_arrays[t][i]
                            if i < len(regime_arrays[t]) else ""
                        ),
                    )
                    pending.append((i + 1, t, order))

            signals = new_signals

        # ── 6. Build per-ticker statistics ─────────────────────────────────
        # Use SPY close prices as the benchmark for alpha/beta if available,
        # otherwise fall back to the first ticker.
        bench_ticker = "SPY" if "SPY" in self.tickers else self.tickers[0]
        bench_close  = close_arrays[bench_ticker]
        bench_rets   = np.diff(bench_close) / np.where(bench_close[:-1] > 0, bench_close[:-1], 1.0)

        ticker_stats: list[TickerPortfolioStats] = []
        for t in self.tickers:
            initial_alloc = (
                self.base_weights.get(t, 1.0 / len(self.tickers))
                * self.initial_capital
            )
            stats = _compute_ticker_stats(
                ticker            = t,
                weight            = self.base_weights.get(t, 0.0),
                equity            = ticker_equity_h[t],
                fills             = ticker_fills[t],
                regime_curve      = ticker_regimes[t],
                annualisation     = self.annualisation,
                initial_alloc     = initial_alloc,
                benchmark_returns = bench_rets,
            )
            ticker_stats.append(stats)

        return PortfolioEngineResult(
            portfolio_equity = portfolio_equity,
            ticker_stats     = ticker_stats,
            rebalance_bars   = rebalance_bars,
            initial_capital  = self.initial_capital,
        )


# ─────────────────────────── Helpers ──────────────────────────────────────

def _compute_target_weights(
    signals:      dict[str, int],
    vol_values:   dict[str, float],
    sma_values:   dict[str, float],
    base_weights: dict[str, float],
    vol_weight:   bool,
    multipliers:  dict[str, float],
) -> dict[str, float]:
    """Return target allocation fractions, capped to each ticker's base weight.

    Key design: inactive (flat-signal) tickers hold their fraction as cash
    rather than redistributing it to active tickers.  This prevents a single
    active ticker from absorbing 100% of the portfolio when others are quiet.

    Within the active tickers, vol-weighting (or base-weight) determines the
    relative split, but the total deployed never exceeds the combined base
    weight of the active set.

    Example — 5 equal-weight tickers, only XLE active:
        total_investable = 0.20  →  XLE gets 20%, 80% stays as cash.
    """
    tickers = list(signals.keys())
    n       = len(tickers) or 1

    active = [t for t in tickers if signals[t] != 0 and sma_values.get(t, 1.0) >= 1.0]
    if not active:
        return {t: 0.0 for t in tickers}

    # Capital ceiling = combined base weight of active tickers only.
    # The rest stays as cash; it is NOT redistributed to active tickers.
    total_investable = sum(base_weights.get(t, 1.0 / n) for t in active)

    if vol_weight:
        raw = {
            t: (1.0 / max(vol_values.get(t, 0.01), 1e-4)) * multipliers.get(t, 1.0)
            for t in active
        }
    else:
        raw = {
            t: base_weights.get(t, 1.0 / n) * multipliers.get(t, 1.0)
            for t in active
        }

    total_raw = sum(raw.values())
    if total_raw <= 0:
        return {t: 0.0 for t in tickers}

    # Scale active tickers so their combined weight = total_investable.
    return {
        t: (raw[t] / total_raw) * total_investable if t in raw else 0.0
        for t in tickers
    }


def _compute_ticker_stats(
    ticker:            str,
    weight:            float,
    equity:            list[float],
    fills:             list[Fill],
    regime_curve:      list[str],
    annualisation:     int,
    initial_alloc:     float,
    benchmark_returns: np.ndarray | None = None,
) -> TickerPortfolioStats:
    """Derive per-ticker Sharpe, drawdown, win-rate, alpha, beta from equity attribution."""
    eq = np.array(equity, dtype=float)

    if len(eq) < 2 or initial_alloc <= 0:
        return TickerPortfolioStats(
            ticker=ticker, weight=weight, equity=equity,
            fills=fills, regime_curve=regime_curve,
        )

    eq_rets  = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)
    mean_r   = float(eq_rets.mean())
    std_r    = float(eq_rets.std())
    sharpe   = mean_r / std_r * math.sqrt(annualisation) if std_r > 1e-12 else 0.0

    down     = eq_rets[eq_rets < 0]
    down_std = float(down.std()) if len(down) > 1 else 1e-12
    sortino  = mean_r / down_std * math.sqrt(annualisation) if down_std > 1e-12 else 0.0

    peak   = np.maximum.accumulate(eq)
    dd     = (eq - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(dd.min())

    total_ret = float(eq[-1] - initial_alloc) / initial_alloc
    calmar    = total_ret / abs(max_dd) if max_dd < -1e-12 else 0.0

    buy_fills  = [f for f in fills if f.side == "buy"]
    sell_fills = [f for f in fills if f.side == "sell"]
    n_trades   = min(len(buy_fills), len(sell_fills))

    trade_rets: list[float] = []
    for b, s in zip(buy_fills, sell_fills):
        if b.fill_price > 0:
            trade_rets.append((s.fill_price - b.fill_price) / b.fill_price)

    win_rate = float(np.mean(np.array(trade_rets) > 0)) if trade_rets else 0.0

    # Alpha / Beta via OLS against benchmark
    alpha = beta = 0.0
    if benchmark_returns is not None and len(benchmark_returns) > 10:
        n = min(len(eq_rets), len(benchmark_returns))
        t_r = eq_rets[-n:]
        b_r = benchmark_returns[-n:]
        var_b = float(np.var(b_r))
        if var_b > 1e-12:
            beta  = float(np.cov(t_r, b_r)[0, 1] / var_b)
            alpha = float((mean_r - beta * float(b_r.mean())) * annualisation)

    return TickerPortfolioStats(
        ticker       = ticker,
        weight       = weight,
        equity       = eq.tolist(),
        fills        = fills,
        regime_curve = regime_curve,
        sharpe       = round(sharpe, 4),
        sortino      = round(sortino, 4),
        total_return = round(total_ret, 6),
        max_drawdown = round(max_dd, 6),
        win_rate     = round(win_rate, 4),
        n_trades     = n_trades,
        calmar       = round(calmar, 4),
        alpha        = round(alpha, 4),
        beta         = round(beta, 4),
    )
