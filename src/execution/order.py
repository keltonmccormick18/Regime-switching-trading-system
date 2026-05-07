"""Order, Fill, and ExecutionReport dataclasses.

These are the canonical trade representations that flow through the
execution engine.  Both the backtest and paper-trading engines produce
the same Fill objects, so downstream analytics are mode-agnostic.

Lifecycle
---------
  Order (pending)
    └─► SimulatedBroker.submit()
          ├─► latency delay
          ├─► slippage model
          ├─► commission calc
          └─► Fill (filled | partial | rejected)
                └─► ExecutionReport
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional


# ─────────────────────────── Order ────────────────────────────────────────

OrderStatus = Literal["pending", "filled", "partial", "cancelled", "rejected"]
OrderType   = Literal["market", "limit", "stop", "stop_limit"]


@dataclass
class Order:
    symbol:      str
    side:        Literal["buy", "sell"]
    quantity:    float                        # requested quantity (positive)
    order_type:  OrderType = "market"

    # Price fields (only relevant for limit / stop orders)
    limit_price: Optional[float] = None
    stop_price:  Optional[float] = None

    # Metadata
    strategy:    str = "engine"
    regime:      str = ""
    model_used:  str = ""
    notes:       str = ""

    # System fields (set on creation)
    order_id:    str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    status:      OrderStatus = "pending"
    created_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_buy(self) -> bool:
        return self.side == "buy"

    def to_dict(self) -> dict:
        return {
            "order_id":    self.order_id,
            "symbol":      self.symbol,
            "side":        self.side,
            "quantity":    round(self.quantity, 4),
            "order_type":  self.order_type,
            "limit_price": self.limit_price,
            "stop_price":  self.stop_price,
            "strategy":    self.strategy,
            "regime":      self.regime,
            "model_used":  self.model_used,
            "status":      self.status,
            "created_at":  self.created_at.isoformat(),
        }


# ─────────────────────────── Fill ─────────────────────────────────────────

@dataclass
class Fill:
    """Execution record for a completed (or partial) order fill."""

    order_id:     str
    symbol:       str
    side:         Literal["buy", "sell"]
    quantity:     float         # actual filled quantity (may differ for partial fills)
    fill_price:   float         # actual execution price including slippage
    gross_price:  float         # mid-market price before slippage / spread

    # Cost breakdown
    commission:   float         # broker commission in dollars
    slippage_cost: float        # price impact in dollars (fill_price - gross_price) × qty
    spread_cost:  float         # bid-ask spread cost in dollars

    # Timing
    latency_ms:   float         # simulated order-to-fill latency
    filled_at:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Derived
    fill_id:      str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    @property
    def total_cost(self) -> float:
        """Total friction cost: commission + slippage + spread."""
        return self.commission + self.slippage_cost + self.spread_cost

    @property
    def net_price(self) -> float:
        """Effective cost basis per share including all frictions."""
        if self.quantity <= 0:
            return self.fill_price
        direction = 1 if self.side == "buy" else -1
        return self.fill_price + direction * self.commission / self.quantity

    @property
    def notional(self) -> float:
        return self.quantity * self.fill_price

    def to_dict(self) -> dict:
        return {
            "fill_id":       self.fill_id,
            "order_id":      self.order_id,
            "symbol":        self.symbol,
            "side":          self.side,
            "quantity":      round(self.quantity, 4),
            "fill_price":    round(self.fill_price, 4),
            "gross_price":   round(self.gross_price, 4),
            "commission":    round(self.commission, 4),
            "slippage_cost": round(self.slippage_cost, 4),
            "spread_cost":   round(self.spread_cost, 4),
            "total_cost":    round(self.total_cost, 4),
            "latency_ms":    round(self.latency_ms, 1),
            "notional":      round(self.notional, 2),
            "filled_at":     self.filled_at.isoformat(),
        }


# ─────────────────────────── ExecutionReport ──────────────────────────────

@dataclass
class ExecutionReport:
    """Combined order + fill result returned by the broker."""

    order:   Order
    fill:    Optional[Fill] = None
    status:  OrderStatus = "pending"
    message: str = ""

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"

    @property
    def is_rejected(self) -> bool:
        return self.status == "rejected"

    def to_dict(self) -> dict:
        return {
            "order":   self.order.to_dict(),
            "fill":    self.fill.to_dict() if self.fill else None,
            "status":  self.status,
            "message": self.message,
        }
