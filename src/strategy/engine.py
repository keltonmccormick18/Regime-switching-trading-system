"""Strategy Engine — orchestrates the full prediction → order pipeline.

Data flow on each bar
─────────────────────
                ┌──────────────────────────────────────┐
  DataFeed ──►  │  StrategyEngine.step()                │
                │                                       │
  prediction    │  1. SignalGenerator.from_price_pred() │
  array    ──►  │     → GeneratedSignal                 │
                │                                       │
                │  2. RiskManager.can_trade()           │
                │     → halt check                      │
                │                                       │
                │  3. PositionSizer.size()              │
                │     → raw quantity                    │
                │                                       │
                │  4. RiskManager.validate_order()      │
                │     → vol scalar + concentration clip │
                │                                       │
                │  5. Portfolio.apply_trade()           │
                │     → internal book update            │
                │                                       │
                └──────────────► TradeOrder | None ─────┘

Stop-loss check (on each new price bar, independent of new predictions):
  StrategyEngine.on_bar(prices) → list[TradeOrder] for forced closes

Usage::

    engine = StrategyEngine(initial_capital=100_000)

    # Called after each model prediction:
    order = engine.on_prediction(
        ticker        = "AAPL",
        prediction    = model.predict(X)[-1],   # (horizon,) array
        current_price = 175.30,
        regime        = "LOW_VOL_BULL",
        model_used    = "TCNModel",
    )
    if order:
        db.insert_trade(...)

    # Called on every new price bar:
    close_orders = engine.on_bar({"AAPL": 178.0, "TSLA": 239.5})
    for o in close_orders:
        db.insert_trade(...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

import numpy as np

from src.strategy.portfolio     import Portfolio
from src.strategy.position_sizer import PositionSizer, SizingConfig, SizingMethod
from src.strategy.risk_manager   import RiskConfig, RiskManager
from src.strategy.signals        import GeneratedSignal, SignalGenerator


# ─────────────────────────── TradeOrder ───────────────────────────────────

@dataclass
class TradeOrder:
    """A fully-specified order ready for execution."""
    symbol:         str
    side:           Literal["buy", "sell"]
    quantity:       float
    price:          float                    # indicative / limit price
    order_type:     Literal["market", "limit"] = "market"
    reason:         str = "signal"           # "signal" | "stop_loss" | "rebalance"
    signal:         Optional[GeneratedSignal] = None
    regime:         str = ""
    model_used:     str = ""
    timestamp:      str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "symbol":     self.symbol,
            "side":       self.side,
            "quantity":   round(self.quantity, 4),
            "price":      self.price,
            "order_type": self.order_type,
            "reason":     self.reason,
            "regime":     self.regime,
            "model_used": self.model_used,
            "timestamp":  self.timestamp,
        }


# ─────────────────────────── StrategyEngine ───────────────────────────────

class StrategyEngine:
    """End-to-end pipeline from model prediction to executable trade order.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio equity.
    sizing_config : SizingConfig, optional
        Position sizing method and parameters.
    risk_config : RiskConfig, optional
        Risk constraint parameters.
    signal_generator : SignalGenerator, optional
        Signal generation parameters (thresholds, regime blacklist, …).
    realized_vol_window : int
        Rolling window (bars) for computing realised vol.  Default 20.
    """

    def __init__(
        self,
        initial_capital:    float = 100_000.0,
        sizing_config:      Optional[SizingConfig] = None,
        risk_config:        Optional[RiskConfig] = None,
        signal_generator:   Optional[SignalGenerator] = None,
        realized_vol_window: int = 20,
    ) -> None:
        self.portfolio    = Portfolio(initial_capital)
        self.sizer        = PositionSizer(sizing_config)
        self.risk         = RiskManager(risk_config)
        self.sig_gen      = signal_generator or SignalGenerator()
        self.vol_window   = realized_vol_window

        self._order_log:  list[TradeOrder] = []
        self._signal_log: list[GeneratedSignal] = []

    # ── Main pipeline ──────────────────────────────────────────────────────

    def on_prediction(
        self,
        ticker:               str,
        prediction:           np.ndarray,      # (horizon,) predicted future prices
        current_price:        float,
        regime:               str = "",
        model_used:           str = "",
        realized_vol:         Optional[float] = None,  # annualised; computed if None
        confidence_override:  Optional[float] = None,  # from ConformalWrapper
    ) -> Optional[TradeOrder]:
        """Generate a trade order from a model price prediction.

        Returns a TradeOrder if the signal passes all risk checks, else None.
        """
        # ── 1. Generate signal ────────────────────────────────────────────
        sig = self.sig_gen.from_price_prediction(
            symbol               = ticker,
            prediction           = prediction,
            current_price        = current_price,
            regime               = regime,
            model_used           = model_used,
            confidence_override  = confidence_override,
        )
        self._signal_log.append(sig)

        if sig.is_flat:
            return None

        # ── 2. Circuit breaker ────────────────────────────────────────────
        if not self.risk.can_trade(self.portfolio):
            return None

        # ── 3. Determine side and whether to flip/close first ─────────────
        existing = self.portfolio.position(ticker)
        existing_qty = existing.quantity if existing else 0.0

        target_side: Literal["buy", "sell"] = "buy" if sig.is_long else "sell"

        # If we're already in the opposite direction, close first
        if existing_qty != 0:
            existing_dir = 1 if existing_qty > 0 else -1
            if existing_dir != sig.direction:
                close_order = self._build_close_order(
                    ticker, existing_qty, current_price, sig, reason="flip"
                )
                self._execute_order(close_order)
                # After close, existing qty is 0
                existing_qty = 0.0

        # ── 4. Size the new position ──────────────────────────────────────
        rv = realized_vol if realized_vol is not None else self.portfolio.rolling_vol(self.vol_window)
        raw_qty = self.sizer.size(
            signal        = sig,
            capital       = self.portfolio.equity,
            current_price = current_price,
            realized_vol  = rv,
            current_qty   = existing_qty,
        )

        # ── 5. Risk validation (vol scalar + concentration) ───────────────
        validated_qty = self.risk.validate_order(
            symbol    = ticker,
            side      = target_side,
            quantity  = raw_qty,
            price     = current_price,
            portfolio = self.portfolio,
            is_close  = False,
        )

        if validated_qty <= 0:
            return None

        # ── 6. Build order and execute against internal portfolio ─────────
        order = TradeOrder(
            symbol     = ticker,
            side       = target_side,
            quantity   = validated_qty,
            price      = current_price,
            reason     = "signal",
            signal     = sig,
            regime     = regime,
            model_used = model_used,
        )
        self._execute_order(order)
        return order

    def on_bar(
        self,
        prices: dict[str, float],
        realized_vol: Optional[float] = None,
    ) -> list[TradeOrder]:
        """Process a new price bar.

        1. Mark portfolio to market.
        2. Check stop-losses.
        3. Return close orders for any triggered stops.

        Call this on every new bar regardless of whether a new prediction
        is available.
        """
        self.portfolio.mark_to_market(prices)

        stop_symbols = self.risk.check_stops(self.portfolio, prices)
        close_orders: list[TradeOrder] = []

        for sym in stop_symbols:
            pos = self.portfolio.position(sym)
            if pos is None or pos.quantity == 0:
                continue
            price = prices.get(sym, pos.last_price)
            order = self._build_close_order(
                sym, pos.quantity, price, signal=None, reason="stop_loss"
            )
            self._execute_order(order)
            close_orders.append(order)

        return close_orders

    def step(
        self,
        ticker:        str,
        prediction:    np.ndarray,
        current_price: float,
        prices:        dict[str, float],   # all current prices for on_bar
        regime:        str = "",
        model_used:    str = "",
        realized_vol:  Optional[float] = None,
    ) -> list[TradeOrder]:
        """Combined step: on_bar() + on_prediction() in one call.

        Returns a list of all orders generated (0–2 orders: stops + signal).
        """
        orders = self.on_bar(prices, realized_vol=realized_vol)

        signal_order = self.on_prediction(
            ticker        = ticker,
            prediction    = prediction,
            current_price = current_price,
            regime        = regime,
            model_used    = model_used,
            realized_vol  = realized_vol,
        )
        if signal_order is not None:
            orders.append(signal_order)

        return orders

    # ── Day boundaries ─────────────────────────────────────────────────────

    def start_day(self) -> None:
        """Call at the beginning of each trading day."""
        self.risk.start_day(self.portfolio)

    # ── Accessors ──────────────────────────────────────────────────────────

    def order_log(self) -> list[TradeOrder]:
        return list(self._order_log)

    def signal_log(self) -> list[GeneratedSignal]:
        return list(self._signal_log)

    def summary(self) -> dict:
        return {
            "portfolio":   self.portfolio.summary(),
            "risk_status": self.risk.status(),
            "n_orders":    len(self._order_log),
            "n_signals":   len(self._signal_log),
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_close_order(
        self,
        symbol:   str,
        quantity: float,           # signed (negative for short)
        price:    float,
        signal:   Optional[GeneratedSignal],
        reason:   str,
    ) -> TradeOrder:
        side: Literal["buy", "sell"] = "sell" if quantity > 0 else "buy"
        return TradeOrder(
            symbol     = symbol,
            side       = side,
            quantity   = abs(quantity),
            price      = price,
            reason     = reason,
            signal     = signal,
            regime     = signal.regime     if signal else "",
            model_used = signal.model_used if signal else "",
        )

    def _execute_order(self, order: TradeOrder) -> None:
        """Apply the order to the internal portfolio and update stop tracking."""
        self.portfolio.apply_trade(
            order.symbol, order.side, order.quantity, order.price
        )
        self._order_log.append(order)

        if order.reason in ("stop_loss", "flip") and order.quantity > 0:
            self.risk.deregister_position(order.symbol)
        elif order.reason == "signal" and order.quantity > 0:
            direction = 1 if order.side == "buy" else -1
            self.risk.register_position(order.symbol, order.price, direction)
