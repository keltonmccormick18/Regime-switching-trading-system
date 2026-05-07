"""Paper trading engine — live simulation with real market data.

Runs the same StrategyEngine + SimulatedBroker stack as the backtest, but
connected to a live DataFeed.  Fills are simulated with configurable costs,
slippage, and latency.  Position state is stored in the StrategyEngine's
Portfolio and optionally persisted to Redis.

Architecture
------------
  PaperEngine
    ├─ DataFeed          ← live prices (Binance WS / Alpaca / yfinance poll)
    ├─ StrategyEngine    ← signals, position sizing, risk management
    ├─ SimulatedBroker   ← fills with friction
    ├─ PostgresDB        ← trade log (optional)
    └─ SignalCache        ← Redis broadcast (optional)

Lifecycle
---------
  engine = PaperEngine(...)
  engine.start()          ← non-blocking: launches background thread
  engine.status()         ← inspect live state at any time
  engine.submit_order(o)  ← manually inject an order
  engine.stop()           ← graceful shutdown

The engine runs a prediction cycle every `prediction_interval` bars.
Between predictions it just does stop-loss / mark-to-market on every bar.

Usage::

    from src.execution.paper_engine import PaperEngine, PaperConfig

    pe = PaperEngine(
        ticker    = "AAPL",
        config    = PaperConfig(prediction_interval=5),
    )
    pe.start()
    time.sleep(3600)
    pe.stop()
    print(pe.status())
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from src.execution.broker import BrokerConfig, SimulatedBroker
from src.execution.order import ExecutionReport, Order
from src.strategy.engine import StrategyEngine, TradeOrder
from src.strategy.position_sizer import SizingConfig
from src.strategy.risk_manager import RiskConfig

log = logging.getLogger(__name__)


# ─────────────────────────── Config ───────────────────────────────────────

@dataclass
class PaperConfig:
    # ── Data ────────────────────────────────────────────────────────────
    interval:             str   = "1d"       # bar interval ("1m", "5m", "1d")
    lookback:             int   = 64         # model input window length
    horizon:              int   = 16         # model prediction horizon
    history_start:        str   = "2022-01-01"

    # ── Strategy cycle ───────────────────────────────────────────────────
    prediction_interval:  int   = 1          # predict every N bars
    use_tda:              bool  = False

    # ── Execution ────────────────────────────────────────────────────────
    broker_preset:        str   = "retail"   # "zero_cost" | "retail" | "institutional"
    initial_capital:      float = 100_000.0
    latency_ms:           float = 100.0      # mean simulated latency

    # ── Risk ─────────────────────────────────────────────────────────────
    max_drawdown_limit:   float = 0.15
    stop_loss_pct:        float = 0.05

    # ── Sizing (dynamic reallocation) ─────────────────────────────────────
    max_position_pct:     float = 0.40       # max % of capital in a single position
    max_leverage:         float = 1.5        # hard leverage cap
    rebalance_threshold:  float = 0.05       # min weight drift to trigger rebalance

    # ── Persistence ──────────────────────────────────────────────────────
    persist_trades:       bool  = True       # write fills to PostgresDB
    broadcast_signals:    bool  = True       # publish signals to Redis


# ─────────────────────────── PaperEngine ──────────────────────────────────

class PaperEngine:
    """Live paper trading simulation.

    Parameters
    ----------
    ticker : str
        Primary ticker symbol to trade.
    model : BaseModel, optional
        Pre-fitted model for predictions.  If None, trains fresh at start.
    config : PaperConfig, optional
        Paper trading configuration.
    on_order : callable, optional
        Callback invoked with ExecutionReport after every fill.
    """

    def __init__(
        self,
        ticker:     str,
        model=None,
        config:     Optional[PaperConfig] = None,
        on_order:   Optional[Callable[[ExecutionReport], None]] = None,
    ) -> None:
        self.ticker   = ticker.upper()
        self.model    = model
        self.cfg      = config or PaperConfig()
        self.on_order = on_order

        # Build broker
        preset = self.cfg.broker_preset.lower()
        if preset == "zero_cost":
            self.broker = SimulatedBroker.zero_cost()
        elif preset == "institutional":
            self.broker = SimulatedBroker.institutional()
        else:
            self.broker = SimulatedBroker.retail()
        # Override latency from config
        self.broker.cfg.latency_mean_ms = self.cfg.latency_ms

        # Build strategy engine
        self.engine = StrategyEngine(
            initial_capital = self.cfg.initial_capital,
            sizing_config   = SizingConfig(
                max_position_pct = self.cfg.max_position_pct,
                max_leverage     = self.cfg.max_leverage,
            ),
            risk_config     = RiskConfig(
                max_drawdown_limit = self.cfg.max_drawdown_limit,
                stop_loss_pct      = self.cfg.stop_loss_pct,
            ),
        )

        # Confidence cache for dynamic rebalancing between prediction cycles
        self._last_confidence: float = 0.5

        # Runtime state
        self._running      = False
        self._thread:       Optional[threading.Thread] = None
        self._bar_count    = 0
        self._last_bar_ts  = None
        self._order_log:   list[ExecutionReport] = []
        self._error_log:   list[str] = []
        self._equity_history: list[dict] = []   # [{timestamp, equity, drawdown}]
        self._lock         = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the paper engine in a background thread."""
        if self._running:
            log.warning("PaperEngine already running")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True, name="PaperEngine")
        self._thread.start()
        log.info("PaperEngine started for %s [%s]", self.ticker, self.cfg.interval)

    def stop(self) -> None:
        """Signal the engine to stop and wait for clean shutdown."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        log.info("PaperEngine stopped")

    def submit_order(self, order: Order, current_price: float) -> ExecutionReport:
        """Manually submit an order outside the automatic cycle."""
        report = self._execute_order(order, current_price)
        return report

    def status(self) -> dict:
        """Snapshot of engine state — safe to call from any thread."""
        with self._lock:
            return {
                "ticker":         self.ticker,
                "running":        self._running,
                "bar_count":      self._bar_count,
                "last_bar_ts":    self._last_bar_ts,
                "n_orders":       len(self._order_log),
                "n_errors":       len(self._error_log),
                "portfolio":      self.engine.portfolio.summary(),
                "risk_status":    self.engine.risk.status(),
                "cost_summary":   self.broker.cost_summary(),
                "recent_errors":  self._error_log[-5:],
                "equity_history": list(self._equity_history),
            }

    def snapshot_equity(self) -> None:
        """Record a lightweight equity snapshot without running a full tick.
        Called by the API on every status/equity poll so the dashboard chart
        grows in real-time even between bar intervals."""
        with self._lock:
            p  = self.engine.portfolio
            ts = datetime.now(timezone.utc).isoformat()
            # Skip duplicate timestamps (same second)
            if self._equity_history and self._equity_history[-1]["timestamp"][:19] == ts[:19]:
                return
            self._equity_history.append({
                "timestamp": ts,
                "equity":    round(p.equity, 2),
                "drawdown":  round(p.drawdown, 6),
            })
            if len(self._equity_history) > 2000:
                self._equity_history = self._equity_history[-2000:]

    def open_positions(self) -> list[dict]:
        """Current open positions as plain dicts for the dashboard."""
        with self._lock:
            out = []
            for sym, pos in self.engine.portfolio.positions.items():
                if pos.quantity == 0:
                    continue
                last = self.engine.portfolio._last_prices.get(sym, pos.avg_entry)
                out.append({
                    "symbol":             sym,
                    "side":               pos.side,
                    "quantity":           pos.quantity,
                    "avg_entry":          round(pos.avg_entry, 4),
                    "current_price":      round(last, 4),
                    "unrealized_pnl":     round(pos.unrealized_pnl(last), 2),
                    "unrealized_pnl_pct": round(pos.unrealized_pnl_pct(last), 6),
                })
            return out

    def order_log(self) -> list[ExecutionReport]:
        with self._lock:
            return list(self._order_log)

    # ── Main loop ──────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main event loop: fetch bars → strategy step → execute orders."""
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                msg = f"{datetime.now(timezone.utc).isoformat()} {exc}"
                log.error("PaperEngine tick error: %s", exc)
                with self._lock:
                    self._error_log.append(msg)
                # Back-off on repeated errors
                time.sleep(min(60, 2 ** min(len(self._error_log), 6)))

            # Sleep until next bar
            interval_seconds = _interval_to_seconds(self.cfg.interval)
            time.sleep(interval_seconds)

    def _tick(self) -> None:
        """Single tick: fetch latest bar, run strategy, execute."""
        from src.features.pipeline import build_features, FEATURE_COLS
        from src.ingestion.historical import load_data, load_historical

        today = datetime.now().strftime("%Y-%m-%d")

        # ── Fetch history (interval-aware) ─────────────────────────────────
        if self.cfg.interval == "1d":
            raw_df = load_data(self.ticker, self.cfg.history_start, today)
            if raw_df is None or len(raw_df) < self.cfg.lookback + 20:
                return
        else:
            pl_df = load_historical(
                self.ticker, self.cfg.history_start, today,
                interval=self.cfg.interval,
            )
            if pl_df.is_empty() or len(pl_df) < self.cfg.lookback + 20:
                return
            raw_df = (
                pl_df.to_pandas()
                .rename(columns={"timestamp": "date"})
                .set_index("date")
            )

        feat_df = build_features(raw_df, use_tda=self.cfg.use_tda).dropna()
        if len(feat_df) < self.cfg.lookback:
            return

        current_price = float(feat_df["close"].iloc[-1])
        current_prices = {self.ticker: current_price}

        # ── Mark-to-market + stop-loss check ──────────────────────────────
        close_orders = self.engine.on_bar(current_prices)
        for trade_order in close_orders:
            order = _trade_order_to_order(trade_order)
            self._execute_order(order, current_price, is_stop=True)

        # ── Prediction cycle ───────────────────────────────────────────────
        with self._lock:
            self._bar_count += 1
            bar_idx = self._bar_count
            self._last_bar_ts = datetime.now(timezone.utc).isoformat()
            # Record equity snapshot for the dashboard chart
            p = self.engine.portfolio
            self._equity_history.append({
                "timestamp": self._last_bar_ts,
                "equity":    round(p.equity, 2),
                "drawdown":  round(p.drawdown, 6),
            })
            # Keep at most 2000 points in memory
            if len(self._equity_history) > 2000:
                self._equity_history = self._equity_history[-2000:]

        if bar_idx % self.cfg.prediction_interval != 0:
            return

        # ── Ensure model is available ──────────────────────────────────────
        if self.model is None:
            self.model = self._train_model(feat_df)

        if self.model is None:
            return

        # ── Build prediction window ────────────────────────────────────────
        feature_data = feat_df[FEATURE_COLS].values
        window = feature_data[-self.cfg.lookback:]
        X_pred = window.T[np.newaxis, :, :]   # (1, n_features, lookback)

        # ── Online model: incremental update with latest data ──────────────
        from src.models.online import OnlineModel
        from src.models.conformal import ConformalWrapper
        base = self.model.base_model if isinstance(self.model, ConformalWrapper) else self.model
        if isinstance(base, OnlineModel) and len(feature_data) > self.cfg.lookback + self.cfg.horizon:
            # Build last window's X/y for one incremental learn step
            x_update = feature_data[-(self.cfg.lookback + self.cfg.horizon):-self.cfg.horizon]
            y_update = feat_df["close"].values[-(self.cfg.horizon):]
            X_upd = x_update.T[np.newaxis, :, :]
            y_upd = y_update[np.newaxis, :]
            try:
                base.update(X_upd, y_upd)
            except Exception:
                pass

        # ── Predict ────────────────────────────────────────────────────────
        prediction = self.model.predict(X_pred)[0]

        # ── Conformal confidence (replaces SNR-based confidence if available)
        conf_override = None
        if isinstance(self.model, ConformalWrapper):
            try:
                conf_override = float(self.model.conformal_confidence(X_pred)[0])
            except Exception:
                pass

        if conf_override is not None:
            with self._lock:
                self._last_confidence = conf_override

        # ── Dynamic rebalance check ────────────────────────────────────────
        self._maybe_rebalance(current_price)

        # ── Strategy engine signal + sizing + risk ────────────────────────
        from src.models.regime import detect_regime, REGIME_MODELS
        regime = detect_regime(feat_df)

        trade_order = self.engine.on_prediction(
            ticker               = self.ticker,
            prediction           = prediction,
            current_price        = current_price,
            regime               = regime.name,
            model_used           = type(self.model).__name__,
            confidence_override  = conf_override,
        )

        if trade_order is not None:
            order = _trade_order_to_order(trade_order)
            report = self._execute_order(order, current_price)

            # ── Persist + broadcast ────────────────────────────────────────
            if self.cfg.persist_trades and report.is_filled:
                self._persist_fill(report, regime.name)
            if self.cfg.broadcast_signals:
                self._broadcast_signal(trade_order, current_price)

    def _maybe_rebalance(self, current_price: float) -> None:
        """Trim or add to position if actual weight drifts from confidence-weighted target."""
        from src.execution.order import Order

        pos = self.engine.portfolio.positions.get(self.ticker)
        if pos is None or pos.quantity == 0:
            return

        equity = self.engine.portfolio.equity
        if equity <= 0 or current_price <= 0:
            return

        ann_vol = self.engine.portfolio.rolling_vol(20)
        if ann_vol < 1e-6:
            ann_vol = 0.20

        with self._lock:
            confidence = self._last_confidence

        # Target weight: vol-scaled by confidence, capped at max_position_pct
        vol_target = 0.15
        target_weight = min(
            vol_target / ann_vol * confidence,
            self.cfg.max_position_pct,
        )
        current_weight = abs(pos.quantity) * current_price / equity

        drift = abs(target_weight - current_weight)
        if drift <= self.cfg.rebalance_threshold:
            return

        target_qty = (target_weight * equity) / current_price
        delta = abs(pos.quantity) - target_qty

        if delta > 0:
            # Trim: reduce toward target
            side = "sell" if pos.quantity > 0 else "buy"
            trim_qty = delta
        else:
            # Add: grow toward target (only if headroom exists)
            if current_weight >= self.cfg.max_position_pct:
                return
            side = "buy" if pos.quantity > 0 else "sell"
            trim_qty = abs(delta)

        if trim_qty < 1.0:
            return

        order = Order(
            symbol     = self.ticker,
            side       = side,
            quantity   = trim_qty,
            order_type = "market",
            strategy   = "paper",
            regime     = "rebalance",
            model_used = "rebalance",
            notes      = f"rebalance drift={drift:.3f} tw={target_weight:.3f} cw={current_weight:.3f}",
        )
        self._execute_order(order, current_price)

    def _execute_order(
        self,
        order:         Order,
        current_price: float,
        is_stop:       bool = False,
    ) -> ExecutionReport:
        """Submit order to broker, apply fill to portfolio, invoke callback."""
        # Simulate latency: in paper trading, just sleep
        if self.broker.cfg.simulate_latency:
            latency_s = self.broker._sample_latency() / 1000.0
            time.sleep(min(latency_s, 1.0))   # cap at 1s in paper mode

        report = self.broker.submit(
            order,
            market_price = current_price,
            volatility   = self.engine.portfolio.rolling_vol(20),
            cash         = self.engine.portfolio.cash,
        )

        if report.is_filled or report.status == "partial":
            # Apply to the portfolio's position book
            self.engine.portfolio.apply_trade(
                order.symbol, order.side,
                report.fill.quantity, report.fill.fill_price,
            )

        with self._lock:
            self._order_log.append(report)

        if self.on_order is not None:
            try:
                self.on_order(report)
            except Exception:
                pass

        return report

    def _train_model(self, feat_df):
        """Train a fresh model if none was provided."""
        try:
            from src.models.regime import detect_regime, REGIME_MODELS
            from pipelines.training_pipeline import prepare_sequences

            regime    = detect_regime(feat_df)
            model_cls = REGIME_MODELS[regime]
            X, y      = prepare_sequences(feat_df, self.cfg.lookback, self.cfg.horizon)
            if len(X) < 50:
                return None
            model = model_cls(horizon=self.cfg.horizon)
            model.fit(X, y)
            log.info("PaperEngine trained fresh %s for %s", model_cls.__name__, self.ticker)
            return model
        except Exception as exc:
            log.error("PaperEngine model training failed: %s", exc)
            return None

    def _persist_fill(self, report: ExecutionReport, regime: str) -> None:
        try:
            from src.api.dependencies import get_db_optional
            from src.storage.db import Trade
            db = get_db_optional()
            if db and report.fill:
                db.insert_trade(Trade(
                    symbol     = report.fill.symbol,
                    side       = report.fill.side,
                    price      = report.fill.fill_price,
                    quantity   = report.fill.quantity,
                    strategy   = "paper",
                    regime     = regime,
                    model_used = self.model.__class__.__name__ if self.model else "",
                    notes      = f"fill={report.fill.fill_id} lat={report.fill.latency_ms:.0f}ms",
                ))
        except Exception as exc:
            log.warning("PaperEngine persist_fill failed: %s", exc)

    def _broadcast_signal(self, trade_order: TradeOrder, price: float) -> None:
        try:
            from src.api.dependencies import get_cache_optional
            from src.storage.cache import Signal
            cache = get_cache_optional()
            if cache:
                sig = Signal(
                    symbol     = trade_order.symbol,
                    signal     = 1 if trade_order.side == "buy" else -1,
                    confidence = trade_order.signal.confidence if trade_order.signal else 0.5,
                    regime     = trade_order.regime,
                    model_used = trade_order.model_used,
                    timestamp  = datetime.now(timezone.utc).isoformat(),
                )
                cache.broadcast(sig)
        except Exception as exc:
            log.warning("PaperEngine broadcast_signal failed: %s", exc)


# ─────────────────────────── Helpers ──────────────────────────────────────

def _trade_order_to_order(to: TradeOrder) -> Order:
    return Order(
        symbol     = to.symbol,
        side       = to.side,
        quantity   = to.quantity,
        order_type = to.order_type,
        strategy   = "paper",
        regime     = to.regime,
        model_used = to.model_used,
        notes      = to.reason,
    )


def _interval_to_seconds(interval: str) -> float:
    """Convert interval string to approximate sleep duration in seconds."""
    table = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "4h": 14400,
        "1d": 86400, "1D": 86400,
    }
    return float(table.get(interval, 60))
