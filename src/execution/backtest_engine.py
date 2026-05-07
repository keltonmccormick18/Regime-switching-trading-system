"""Event-driven backtest engine.

Unlike the vectorized backtest in backtest.py (which applies signals as a
column operation), this engine replays history bar-by-bar through the same
StrategyEngine + SimulatedBroker stack used in live paper trading.  The
result is a fully realistic simulation:

  • Orders placed at bar T fill at bar T+1 open (execution lag)
  • Fill price = open ± slippage ± spread
  • Commission deducted per fill
  • Latency is simulated but compressed (1 bar = latency_mean_ms wall-clock)
  • Stop-losses fire on the bar after the trigger bar

Two modes
---------
  run_signals(df, signals)
      Fastest path: give it a pre-computed signal series and a price DataFrame.
      No model is called; just apply signals through the broker.

  run_model(ticker, df, model, lookback, horizon)
      Walk-forward: build features, predict at each bar, feed predictions
      through the strategy engine.  The model is NOT re-trained during the
      loop (use the pre-fitted model passed in).

Output
------
  BacktestResult  (from src/execution/backtest.py) — same schema used by the
  vectorized backtest, plus an extras dict with fill-level details.

Usage::

    from src.execution.backtest_engine import BacktestEngine
    from src.execution.broker import SimulatedBroker, BrokerConfig

    broker = SimulatedBroker(BrokerConfig(slippage_model="sqrt_impact"))
    engine = BacktestEngine(broker=broker, initial_capital=100_000)

    result = engine.run_signals(df, signals=pd.Series([1, 0, -1, ...]))
    print(result.sharpe, result.total_return)
    print(engine.broker.cost_summary())
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.execution.backtest import BacktestResult
from src.execution.broker import BrokerConfig, SimulatedBroker
from src.execution.order import ExecutionReport, Fill, Order
from src.execution.paper_trader import PositionTracker

# Per-regime signal thresholds — low-vol regimes have smaller natural return
# magnitudes, so they need a lower bar to generate signals.
_REGIME_SIGNAL_THRESHOLDS: dict[str, float] = {
    "HIGH_VOL_BULL": 0.008,
    "HIGH_VOL_BEAR": 0.005,
    "LOW_VOL_BULL":  0.002,
    "LOW_VOL_BEAR":  0.003,
}

# All four regime labels (must match REGIME_MODELS keys)
_ALL_REGIMES = frozenset(["HIGH_VOL_BULL", "HIGH_VOL_BEAR", "LOW_VOL_BULL", "LOW_VOL_BEAR"])


# ─────────────────────────── BacktestEngine ───────────────────────────────

class BacktestEngine:
    """Event-driven backtest with realistic execution simulation.

    Parameters
    ----------
    broker : SimulatedBroker
        Handles all fills.  Defaults to SimulatedBroker() with standard settings.
    initial_capital : float
        Starting cash balance.
    execution_lag : int
        Bars between signal generation and fill.  1 = fill at next bar open.
    annualisation : int
        Trading days per year for Sharpe/vol calculations.
    """

    def __init__(
        self,
        broker:           Optional[SimulatedBroker] = None,
        initial_capital:  float = 100_000.0,
        execution_lag:    int   = 1,
        annualisation:    int   = 252,
    ) -> None:
        self.broker          = broker or SimulatedBroker()
        self.initial_capital = initial_capital
        self.execution_lag   = execution_lag
        self.annualisation   = annualisation

        # Runtime state (reset on each run)
        self._cash:     float = initial_capital
        self._tracker:  PositionTracker = PositionTracker()
        self._equity:   list[float] = []
        self._fills:    list[Fill]  = []
        self._pending:  list[tuple[int, Order]] = []   # (fill_bar_idx, order)

    # ── Public API ─────────────────────────────────────────────────────────

    def run_signals(
        self,
        df:             pd.DataFrame,
        signals:        pd.Series,
        price_col:      str   = "close",
        open_col:       str   = "open",
        volume_col:     str   = "volume",
        vol_window:     int   = 20,
        adv_window:     int   = 20,
        position_size:  float = 0.02,      # fraction of equity risked per trade
        stop_loss_pct:  float = 0.05,      # 0.0 = disabled; otherwise max loss per position
    ) -> BacktestResult:
        """Run signals through the event-driven engine.

        Parameters
        ----------
        df          : OHLCV DataFrame (must have price_col and open_col).
        signals     : Series aligned to df, values in {-1, 0, 1}.
        position_size : Fraction of equity per trade (fixed-fractional sizing).

        Returns
        -------
        BacktestResult with fill-level realism baked in.
        """
        self._reset()
        df = df.copy().reset_index(drop=True)
        sigs = signals.fillna(0).clip(-1, 1).values

        # Realised vol (annualised) per bar — for slippage models
        rets = df[price_col].pct_change().fillna(0).values
        ann_vol = _rolling_vol(rets, vol_window, self.annualisation)

        # ADV in shares — for sqrt-impact model
        if volume_col in df.columns:
            adv_shares = _rolling_mean(df[volume_col].values, adv_window)
        else:
            adv_shares = np.zeros(len(df))

        open_prices  = df[open_col].values  if open_col  in df.columns else df[price_col].values
        close_prices = df[price_col].values

        prev_signal  = 0.0
        pending_queue: list[tuple[int, Order, float]] = []  # (fill_bar, order, adv)

        for i in range(len(df)):
            close = float(close_prices[i])
            if close <= 0:
                self._equity.append(self._mark_equity(close_prices, i))
                continue

            # ── 1. Process fills due this bar ──────────────────────────────
            still_pending = []
            for fill_bar, order, adv_at_signal in pending_queue:
                if i >= fill_bar:
                    fill_price = float(open_prices[i])
                    report = self.broker.submit(
                        order,
                        market_price = fill_price,
                        volatility   = float(ann_vol[i]),
                        adv          = adv_at_signal,
                        cash         = self._cash,
                    )
                    if report.is_filled or report.status == "partial":
                        self._apply_fill(report.fill)
                else:
                    still_pending.append((fill_bar, order, adv_at_signal))
            pending_queue = still_pending

            # ── 2. Stop-loss check ────────────────────────────────────────────
            for sym, pos in list(self._tracker._positions.items()):
                if pos.quantity == 0 or stop_loss_pct <= 0:
                    continue
                if pos.avg_entry > 0:
                    loss_pct = (pos.avg_entry - close) / pos.avg_entry if pos.quantity > 0 else \
                               (close - pos.avg_entry) / pos.avg_entry
                    if loss_pct >= stop_loss_pct:
                        stop_order = Order(
                            symbol    = sym,
                            side      = "sell" if pos.quantity > 0 else "buy",
                            quantity  = abs(pos.quantity),
                            order_type= "market",
                            strategy  = "stop_loss",
                        )
                        pending_queue.append((i + self.execution_lag, stop_order, float(adv_shares[i])))

            # ── 3. Generate new signal order ───────────────────────────────
            sig = float(sigs[i])
            if sig != prev_signal:
                equity_now = self._mark_equity(close_prices, i)
                # Close existing opposite position
                for sym, pos in list(self._tracker._positions.items()):
                    if pos.quantity == 0:
                        continue
                    opposite = (pos.quantity > 0 and sig < 0) or (pos.quantity < 0 and sig > 0)
                    flat     = sig == 0
                    if opposite or flat:
                        close_order = Order(
                            symbol   = sym,
                            side     = "sell" if pos.quantity > 0 else "buy",
                            quantity = abs(pos.quantity),
                            strategy = "signal_close",
                        )
                        pending_queue.append((i + self.execution_lag, close_order, float(adv_shares[i])))

                # Open new position
                if sig != 0:
                    qty = _size_position(sig, equity_now, close, position_size)
                    if qty > 0:
                        side = "buy" if sig > 0 else "sell"
                        new_order = Order(
                            symbol   = "ASSET",    # single-asset backtest
                            side     = side,
                            quantity = qty,
                            strategy = "signal_open",
                        )
                        pending_queue.append((i + self.execution_lag, new_order, float(adv_shares[i])))
                        if side == "buy":
                            self._buy_bars.append(i)
                        else:
                            self._sell_bars.append(i)

            prev_signal = sig

            # ── 4. Record equity ────────────────────────────────────────────
            self._equity.append(self._mark_equity(close_prices, i))

        result = self._build_result(rets)
        # Buy-and-hold benchmark: fully invested from day 1, same price series
        if len(close_prices) > 0 and float(close_prices[0]) > 0:
            bah = close_prices.astype(float) / float(close_prices[0]) * self.initial_capital
            result.bah_equity = bah.tolist()
            result.bah_return = float(close_prices[-1]) / float(close_prices[0]) - 1.0
        return result

    def run_model(
        self,
        ticker:        str,
        df:            pd.DataFrame,
        model,                           # fitted BaseModel (fallback when regime_models not given)
        lookback:      int   = 64,
        horizon:       int   = 16,
        price_col:     str   = "close",
        feature_cols:  Optional[list[str]] = None,
        position_size: float = 0.02,
        vol_window:    int   = 20,
        adv_window:    int   = 20,
        regime_models: Optional[dict] = None,   # regime label str → fitted BaseModel
        # Signal structure
        signal_hold:        int   = 1,      # re-predict every N bars (1 = every bar)
        long_only:          bool  = False,  # convert short signals to flat
        signal_threshold:         float = 0.005, # min |predicted return| to generate signal
        min_confidence:           float = 0.30,  # min SNR confidence to generate signal
        stop_loss_pct:            float = 0.05,  # max position loss before forced exit; 0 = disabled
        regime_signal_thresholds: Optional[dict] = None,  # per-regime override; None = use _REGIME_SIGNAL_THRESHOLDS
        regime_long_only:         Optional[dict] = None,  # per-regime long_only; None = use global long_only
        soft_blend:               bool           = True,  # A3: blend all 4 models by soft regime probs
        native_regime:            Optional[str]  = None,  # when set, model only trades in this regime
        regime_default_long:      Optional[set]  = None,  # regimes where strategy is long by default
        regime_always_long:       Optional[set]  = None,  # regimes where strategy is unconditionally long (no model)
        # Walk-forward retraining
        retrain_every:  int            = 0,     # 0 = disabled; N = retrain every N bars
        full_X:         Optional[np.ndarray] = None,  # full sequence array (train+test)
        full_y:         Optional[np.ndarray] = None,
        full_regimes:   Optional[list]  = None, # per-window regime label (len = len(full_X))
        train_split:    int            = 0,     # index in full_X where OOS begins
        retrain_epochs: int            = 10,    # epochs per walk-forward retrain
    ) -> BacktestResult:
        """Walk-forward backtest driven by a fitted model.

        The model predicts at each bar; predictions are fed through the
        strategy engine's signal generator, then executed via the broker.

        Parameters
        ----------
        ticker    : Ticker symbol (used for position tracking).
        df        : Feature + price DataFrame produced by build_features().
        model     : Fitted BaseModel (TCNModel, TCNLSTMModel, TransformerModel).
        lookback  : Input sequence length (must match model training).
        horizon   : Prediction horizon (must match model training).
        """
        from src.features.pipeline import FEATURE_COLS, MACRO_FEATURE_COLS
        from src.strategy.signals import SignalGenerator

        # Pre-import walk-forward retraining helpers (no-op if disabled)
        if retrain_every > 0 and full_X is not None:
            from src.models.regime import REGIME_MODELS as _REGIME_MODELS   # noqa: F401
            from src.models.train import train_model as _train_model         # noqa: F401
        else:
            _REGIME_MODELS = None   # type: ignore[assignment]
            _train_model   = None   # type: ignore[assignment]

        self._reset()
        # Auto-detect which feature columns are present in the DataFrame —
        # matches the same logic as prepare_sequences() in training_pipeline.py,
        # so inference always uses the same number of features the model was
        # trained on (7 base-only or 12 base+macro).
        if feature_cols is not None:
            cols = feature_cols
        else:
            _all_possible = FEATURE_COLS + MACRO_FEATURE_COLS
            cols = [c for c in _all_possible if c in df.columns]
        df = df.dropna().copy().reset_index(drop=True)

        prices   = df[price_col].values
        features = df[cols].values          # (T, n_features)
        rets     = np.diff(prices, prepend=prices[0]) / np.where(prices > 0, prices, 1.0)
        ann_vol  = _rolling_vol(rets, vol_window, self.annualisation)

        adv_shares = np.zeros(len(df))

        sig_gen = SignalGenerator(
            min_return_threshold=signal_threshold,
            min_confidence=min_confidence,
        )
        signals  = np.zeros(len(df))
        _hold    = max(1, signal_hold)   # bars between signal updates

        # A3: soft ensemble blending — active when all 4 regime models are available
        _soft_blend = (
            soft_blend
            and regime_models is not None
            and _ALL_REGIMES.issubset(regime_models)
            and "vol_20" in df.columns
            and "sma_ratio_200" in df.columns
        )
        _vol_thresh = float(df["vol_20"].quantile(0.70)) if _soft_blend else None

        # Precompute per-bar regime labels (expanding-window, no look-ahead).
        # Used both for dynamic model switching and for the regime_curve overlay.
        _regime_cols = {"tda_l1", "vol_20", "sma_ratio_200"}
        bar_regimes: list[str] = (
            _regime_series(df) if _regime_cols.issubset(df.columns) else []
        )

        # Pre-compute all sequence windows so online models can update using
        # lagged actual labels (horizon bars after each prediction bar).
        all_windows = np.stack(
            [features[j - lookback:j].T for j in range(lookback, len(df))],
            axis=0,
        )  # (T_oos, n_features, lookback)

        # Walk-forward: predict at each bar after we have enough lookback.
        # When signal_hold > 1 the signal is only recomputed every _hold bars;
        # between updates the previous signal is carried forward unchanged.
        last_signal = 0.0
        for i in range(lookback, len(df)):
            # ── Walk-forward retraining ────────────────────────────────────────
            oos_bars = i - lookback
            if (
                retrain_every > 0
                and full_X is not None
                and full_y is not None
                and _REGIME_MODELS is not None
                and oos_bars % retrain_every == 0
            ):
                oos_idx = train_split + oos_bars   # index in full_X up to now
                if oos_idx > lookback:             # need at least one full window
                    X_rt = full_X[:oos_idx]
                    y_rt = full_y[:oos_idx]
                    n_feat   = X_rt.shape[1]
                    _horizon = y_rt.shape[1]
                    new_models: dict = {}
                    for _re, _mc in _REGIME_MODELS.items():
                        # Filter to windows whose regime matches this model's regime
                        if full_regimes and len(full_regimes) >= oos_idx:
                            _rname = _re.name
                            _idx = [k for k, r in enumerate(full_regimes[:oos_idx])
                                    if r == _rname]
                            if len(_idx) >= 10:
                                _X_fit = X_rt[_idx]
                                _y_fit = y_rt[_idx]
                            else:
                                _X_fit, _y_fit = X_rt, y_rt  # fallback: too few samples
                                warnings.warn(
                                    f"WF retrain bar {oos_bars}: {_rname} has only "
                                    f"{len(_idx)} windows — using full set ({oos_idx})",
                                    stacklevel=2,
                                )
                        else:
                            _X_fit, _y_fit = X_rt, y_rt
                        _m = _train_model(        # type: ignore[misc]
                            _mc, _X_fit, _y_fit,
                            n_features=n_feat,
                            lookback=lookback,
                            horizon=_horizon,
                            epochs=retrain_epochs,
                        )
                        new_models[_re.name] = _m
                    if regime_models is not None:
                        regime_models.update(new_models)
                    else:
                        regime_models = new_models
                    model = next(iter(new_models.values()))

            # ── Resolve current regime (needed for gating + routing + threshold) ─
            current_regime = bar_regimes[i] if i < len(bar_regimes) else ""

            # ── Native-regime gate ─────────────────────────────────────────────
            # When a native_regime is specified (benchmark mode), the model is a
            # regime-specialist trained only on those market dynamics.  Trading
            # outside the native regime produces inverted or garbage signals
            # because the model has never seen those conditions during training.
            # Gate is checked every bar (not just on signal-hold bars) so a
            # position entered in the native regime is immediately closed when
            # the regime transitions, rather than being held through alien bars.
            if native_regime and current_regime and current_regime != native_regime:
                last_signal = 0.0
                signals[i] = 0.0
                continue

            # ── Regime always long — pure buy-and-hold, no model consulted ────
            # When a regime is in regime_always_long (e.g. LOW_VOL_BULL in the
            # "buy and hold low-vol uptrends" mode), skip the model entirely and
            # hold long unconditionally.  Different from regime_default_long,
            # which still calls direction_prob() before deciding to exit.
            if regime_always_long and current_regime in regime_always_long:
                last_signal = 1.0
                signals[i] = 1.0
                continue

            # ── Online model update using lagged actual labels ─────────────────
            # For models that expose update() (River ARF etc.), feed the actual
            # return that materialised `horizon` bars ago — no look-ahead.
            # Throttled to every `horizon` bars: labels only become "new" once
            # per horizon cycle (overlapping windows share most observations),
            # and firing every bar multiplies ARF training cost by ~horizon×.
            if oos_bars >= horizon and oos_bars % horizon == 0 and hasattr(model, "update"):
                try:
                    _lag_oos = oos_bars - horizon          # oos index of the past window
                    _lag_abs = lookback + _lag_oos         # absolute index in df
                    _lag_X   = all_windows[_lag_oos][np.newaxis, :, :]   # (1, f, lb)
                    _anchor  = float(prices[_lag_abs - 1])
                    _fwd_end = min(_lag_abs + horizon, len(prices))
                    if _anchor > 0 and _fwd_end > _lag_abs:
                        _lag_y = (prices[_lag_abs:_fwd_end] / _anchor - 1.0
                                  ).astype(np.float32)
                        # Pad to full horizon if near end of data
                        if len(_lag_y) < horizon:
                            _lag_y = np.pad(_lag_y, (0, horizon - len(_lag_y)),
                                            mode="edge")
                        model.update(_lag_X, _lag_y[np.newaxis, :])
                except Exception:
                    pass

            # ── Signal hold: only recompute every _hold bars ───────────────────
            if oos_bars % _hold == 0:
                # Dynamic regime → model routing
                if regime_models and current_regime:
                    active_model = regime_models.get(current_regime, model)
                else:
                    active_model = model

                # Regime-adaptive signal threshold: low-vol regimes produce
                # smaller predicted returns and need a lower floor to avoid
                # sitting in cash the entire time.
                _thresholds = regime_signal_thresholds if regime_signal_thresholds is not None else _REGIME_SIGNAL_THRESHOLDS
                sig_gen.min_return_threshold = _thresholds.get(current_regime, signal_threshold)

                window = features[i - lookback:i].T        # (n_features, lookback)
                X = window[np.newaxis, :, :]                # (1, n_features, lookback)

                # ── A3: soft ensemble blend vs hard regime routing ─────────────
                if _soft_blend and _vol_thresh is not None:
                    _vol_val = float(df["vol_20"].iloc[i])
                    _sma_val = float(df["sma_ratio_200"].iloc[i])
                    _sprobs  = _regime_soft_probs(_vol_val, _sma_val, _vol_thresh)
                    pred_returns = np.zeros(horizon, dtype=np.float32)
                    for _rn, _wt in _sprobs.items():
                        _rm = regime_models.get(_rn, model)   # type: ignore[index]
                        pred_returns += _wt * _rm.predict(X)[0]
                else:
                    pred_returns = active_model.predict(X)[0]   # (horizon,) — forward returns

                current_price = float(prices[i])
                pred_prices   = current_price * (1.0 + pred_returns)

                # ── A1: direction-head agreement-gated confidence override ────
                # Use direction_prob() (raw sigmoid, >0.5 = bullish) to check
                # whether the direction head and the regression head agree.
                #
                #  • Both heads agree + head is certain  → certainty becomes
                #    the confidence override (positive selection).
                #  • Heads conflict + head is certain    → override = 0 so
                #    the signal is suppressed (avoids wrong-direction trades).
                #  • Head is uncertain (|prob−0.5|<0.20) → no override; the
                #    existing SNR-based confidence from SignalGenerator decides.
                #
                # This fixes two regressions:
                #   LOW_VOL_BEAR / TCNLSTMModel: direction head was confident
                #     DOWN but override was treated as "confident UP" → longs
                #     in bear market → 12.5 % win rate.
                #   LOW_VOL_BULL / TCNModel: direction head uncertain in
                #     low-vol → override ≈ 0.04 → all signals killed.
                _conf_override: Optional[float] = None
                if hasattr(active_model, "direction_prob"):
                    try:
                        dir_prob      = float(active_model.direction_prob(X)[0])
                        dir_certainty = abs(dir_prob - 0.5) * 2     # [0, 1]
                        reg_up = float(pred_returns[-1]) > 0         # regression head: up?
                        dir_up = dir_prob > 0.5                      # direction head:  up?

                        if dir_certainty >= 0.30:
                            # Head is genuinely confident — use agreement check.
                            if dir_up == reg_up:
                                # Both heads agree → certainty becomes the confidence
                                _conf_override = dir_certainty
                            else:
                                # Heads conflict → suppress wrong-direction trade
                                _conf_override = 0.0
                        elif dir_certainty >= 0.10 and dir_up != reg_up:
                            # Head is mildly opinionated but DISAGREES with regression.
                            # Small disagreement penalty: cut confidence in half so
                            # only the strongest SNR signals still pass through.
                            # (Low-vol bull: tiny upward moves get mildly conflicting
                            # direction signals → half-penalty avoids suppressing
                            # all bull signals while reducing noise.)
                            _conf_override = 0.15   # below default 0.30 threshold
                        # else (very uncertain, or mildly certain + agreeing):
                        #   no override — SNR-based confidence from SignalGenerator
                    except Exception:
                        pass

                sig = sig_gen.from_price_prediction(
                    symbol              = ticker,
                    prediction          = pred_prices,
                    current_price       = current_price,
                    confidence_override = _conf_override,
                )
                new_signal = float(sig.direction)

                # ── Regime default long ────────────────────────────────────────
                # In designated regimes (e.g. LOW_VOL_BULL) the regime itself IS
                # the signal: the stock is in a steady, low-volatility uptrend and
                # active model switching just adds friction.  Default to long on
                # every bar; only step aside when BOTH heads are confidently
                # bearish — direction head >65% sure prices fall AND regression
                # mean is negative.  Any ambiguity keeps the long position open.
                if regime_default_long and current_regime in regime_default_long:
                    _confidently_bearish = False
                    if hasattr(active_model, "direction_prob"):
                        try:
                            _rdl_dp = float(active_model.direction_prob(X)[0])
                            _rdl_dc = abs(_rdl_dp - 0.5) * 2          # certainty in [0,1]
                            _confidently_bearish = (
                                _rdl_dc >= 0.30                        # head is confident
                                and _rdl_dp < 0.35                     # head says DOWN
                                and float(pred_returns.mean()) < 0     # regression also DOWN
                            )
                        except Exception:
                            pass
                    if not _confidently_bearish:
                        new_signal = 1.0

                # Per-regime long_only: bear regimes can short, bull regimes stay long-only
                _is_long_only = (
                    regime_long_only.get(current_regime, long_only)
                    if regime_long_only is not None
                    else long_only
                )
                if _is_long_only and new_signal < 0:
                    new_signal = 0.0
                last_signal = new_signal

            signals[i] = last_signal

        result = self.run_signals(
            df             = df,
            signals        = pd.Series(signals),
            price_col      = price_col,
            vol_window     = vol_window,
            adv_window     = adv_window,
            position_size  = position_size,
            stop_loss_pct  = stop_loss_pct,
        )

        # ── Regime curve — reuse what we already computed above ────────────
        result.regime_curve = bar_regimes

        # ── Signal series + exit bars ──────────────────────────────────────
        # signal_series: full per-bar signal so the frontend can shade cash periods.
        # exit_bars: bars where the strategy transitions from long → flat/cash
        # (distinct from sell_bars which only captures short entries).
        result.signal_series = signals.tolist()
        result.exit_bars = [
            i for i in range(1, len(signals))
            if signals[i - 1] > 0 and signals[i] == 0
        ]

        # ── Fair buy-and-hold benchmark — start from bar `lookback` ────────
        # The strategy can't trade until it has a full lookback window, so it
        # sits flat in cash for bars 0..lookback-1.  Starting B&H from bar 0
        # gives it an unfair head start of up to `lookback` bars.  Instead,
        # normalise B&H so both curves begin investing at the same bar.
        if lookback < len(prices) and float(prices[lookback]) > 0:
            bah = np.full(len(prices), self.initial_capital, dtype=float)
            bah[lookback:] = (
                prices[lookback:].astype(float)
                / float(prices[lookback])
                * self.initial_capital
            )
            result.bah_equity  = bah.tolist()
            result.bah_return  = float(prices[-1]) / float(prices[lookback]) - 1.0

        # ── Alpha / beta vs asset buy-and-hold ─────────────────────────────
        # Use only the OOS portion (after lookback) so alpha/beta are also
        # computed on the same window both strategies were active.
        asset_rets   = np.diff(prices[lookback:]) / np.where(
                           prices[lookback:-1] > 0, prices[lookback:-1], 1.0)
        strat_rets   = np.array(result.daily_returns[lookback + 1:])
        result.alpha, result.beta = _alpha_beta(strat_rets, asset_rets, self.annualisation)

        return result

    # ── Accessors ──────────────────────────────────────────────────────────

    def fills(self) -> list[Fill]:
        return list(self._fills)

    def equity_curve(self) -> list[float]:
        return list(self._equity)

    def cost_summary(self) -> dict:
        return self.broker.cost_summary()

    # ── Internal ──────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._cash      = self.initial_capital
        self._tracker   = PositionTracker()
        self._equity    = []
        self._fills     = []
        self._buy_bars  = []
        self._sell_bars = []
        self.broker.reset()

    def _apply_fill(self, fill: Fill) -> None:
        """Update cash and position tracker from a confirmed fill."""
        self._fills.append(fill)
        # Cash update: buys cost money, sells bring money in
        direction = -1 if fill.side == "buy" else 1
        self._cash += direction * fill.fill_price * fill.quantity
        self._cash -= fill.commission   # commission always reduces cash
        self._tracker.apply_trade(fill.symbol, fill.side, fill.quantity, fill.fill_price)

    def _mark_equity(self, close_prices: np.ndarray, bar_idx: int) -> float:
        """Cash + market value of all open positions at current close."""
        close = float(close_prices[bar_idx])
        pos_value = sum(
            p.quantity * close
            for p in self._tracker._positions.values()
            if p.quantity != 0
        )
        return self._cash + pos_value

    def _build_result(self, daily_rets: np.ndarray) -> BacktestResult:
        """Convert the equity curve into a BacktestResult."""
        equity = np.array(self._equity, dtype=float)
        if len(equity) < 2:
            return BacktestResult(
                sharpe=0.0, sortino=0.0, total_return=0.0, max_drawdown=0.0,
                win_rate=0.0, n_trades=len(self._fills) // 2,
                equity=equity.tolist(), daily_returns=daily_rets.tolist(),
            )

        eq_rets = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1.0)
        mean_r  = float(eq_rets.mean())
        std_r   = float(eq_rets.std())
        sharpe  = mean_r / std_r * math.sqrt(self.annualisation) if std_r > 1e-12 else 0.0

        down     = eq_rets[eq_rets < 0]
        down_std = float(down.std()) if len(down) > 1 else 1e-12
        sortino  = mean_r / down_std * math.sqrt(self.annualisation) if down_std > 1e-12 else 0.0

        peak        = np.maximum.accumulate(equity)
        dd          = (equity - peak) / np.where(peak > 0, peak, 1.0)
        max_dd      = float(dd.min())
        total_ret   = float(equity[-1] / self.initial_capital) - 1.0
        calmar      = total_ret / abs(max_dd) if max_dd < -1e-12 else 0.0

        # Per-trade stats from fills
        buy_fills  = [f for f in self._fills if f.side == "buy"]
        sell_fills = [f for f in self._fills if f.side == "sell"]
        n_trades   = min(len(buy_fills), len(sell_fills))

        trade_rets: list[float] = []
        for b, s in zip(buy_fills, sell_fills):
            tr = (s.fill_price - b.fill_price) / b.fill_price if b.fill_price > 0 else 0.0
            trade_rets.append(tr)

        win_rate       = float(np.mean(np.array(trade_rets) > 0)) if trade_rets else 0.0
        avg_trade_ret  = float(np.mean(trade_rets)) if trade_rets else 0.0

        return BacktestResult(
            sharpe        = sharpe,
            sortino       = sortino,
            total_return  = total_ret,
            max_drawdown  = max_dd,
            win_rate      = win_rate,
            n_trades      = n_trades,
            equity        = equity.tolist(),
            daily_returns = eq_rets.tolist(),
            calmar        = calmar,
            avg_trade_return = avg_trade_ret,
            buy_bars      = list(self._buy_bars),
            sell_bars     = list(self._sell_bars),
            notes         = f"event-driven | fills={len(self._fills)} | "
                            f"friction=${self.broker.total_friction():.2f}",
        )


# ─────────────────────────── Helpers ──────────────────────────────────────

def _regime_series(df: pd.DataFrame, vol_percentile: float = 0.70) -> list:
    """Return a regime label for every row of df using expanding-window thresholds.

    Uses the same two-signal logic as detect_regime() but vectorised so the
    whole curve can be computed in one pass without looping over detect_regime().
    """
    tda = df["tda_l1"].replace(0.0, float("nan"))
    if tda.dropna().shape[0] > 10:
        thresh    = tda.expanding(min_periods=10).quantile(vol_percentile)
        is_high   = (tda > thresh).fillna(False)
    else:
        vol       = df["vol_20"]
        thresh    = vol.expanding(min_periods=10).quantile(vol_percentile)
        is_high   = (vol > thresh).fillna(False)

    is_bull = df["sma_ratio_200"] > 1.0

    out = []
    for hv, bull in zip(is_high, is_bull):
        if hv and bull:
            out.append("HIGH_VOL_BULL")
        elif hv:
            out.append("HIGH_VOL_BEAR")
        elif bull:
            out.append("LOW_VOL_BULL")
        else:
            out.append("LOW_VOL_BEAR")
    return out


def _regime_soft_probs(vol: float, sma_ratio: float, vol_thresh: float) -> dict[str, float]:
    """A3: Soft regime probability weights from continuous feature values.

    Instead of hard-routing to a single regime model, these weights blend all
    four models' predictions in proportion to how likely each regime is:

      vol_score  = sigmoid(30 * (vol - vol_thresh))   # prob of high-vol
      bull_score = sigmoid(20 * (sma_ratio - 1.0))    # prob of bull

    The sharpness constants (30, 20) set how quickly the blend transitions
    near the threshold.  The four weights always sum to 1.
    """
    _clamp = lambda x: max(-50.0, min(50.0, x))
    vol_score  = 1.0 / (1.0 + math.exp(_clamp(-30.0 * (vol - vol_thresh))))
    bull_score = 1.0 / (1.0 + math.exp(_clamp(-20.0 * (sma_ratio - 1.0))))
    return {
        "HIGH_VOL_BULL": vol_score * bull_score,
        "HIGH_VOL_BEAR": vol_score * (1.0 - bull_score),
        "LOW_VOL_BULL":  (1.0 - vol_score) * bull_score,
        "LOW_VOL_BEAR":  (1.0 - vol_score) * (1.0 - bull_score),
    }


def _alpha_beta(strategy_rets: np.ndarray, asset_rets: np.ndarray,
                annualisation: int = 252) -> tuple[float, float]:
    """Jensen's alpha and beta vs the asset's own buy-and-hold."""
    min_len = min(len(strategy_rets), len(asset_rets))
    if min_len < 10:
        return 0.0, 0.0
    s = strategy_rets[-min_len:]
    b = asset_rets[-min_len:]
    var_b = float(np.var(b))
    if var_b < 1e-10:
        return 0.0, 0.0
    beta  = float(np.cov(s, b)[0, 1]) / var_b
    alpha = float(np.mean(s) - beta * np.mean(b)) * annualisation
    return round(alpha, 6), round(beta, 6)


def _rolling_vol(returns: np.ndarray, window: int, annualisation: int) -> np.ndarray:
    out = np.full(len(returns), 0.20)   # default 20%
    for i in range(window, len(returns)):
        s = returns[i - window:i].std()
        out[i] = s * math.sqrt(annualisation) if s > 0 else out[i - 1]
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = arr[start:i + 1].mean()
    return out


def _size_position(signal: float, equity: float, price: float, pct: float) -> float:
    """Fixed-fractional position sizing: risk `pct` of equity per trade."""
    if price <= 0:
        return 0.0
    return max(0.0, (abs(signal) * pct * equity) / price)
