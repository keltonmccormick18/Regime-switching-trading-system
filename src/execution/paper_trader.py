"""Paper trading position tracker.

Maintains an in-memory book of open positions and computes P&L in real time.
Thread-safe via a per-symbol lock; safe to call from multiple FastAPI workers.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0       # signed: + = long, - = short
    avg_entry: float = 0.0
    realized_pnl: float = 0.0

    @property
    def side(self) -> Literal["long", "short", "flat"]:
        if self.quantity > 0:
            return "long"
        if self.quantity < 0:
            return "short"
        return "flat"

    def unrealized_pnl(self, current_price: float) -> float:
        return self.quantity * (current_price - self.avg_entry)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.avg_entry == 0:
            return 0.0
        return (current_price - self.avg_entry) / self.avg_entry * 100.0


class PositionTracker:
    """
    Track paper positions across multiple symbols.

    apply_trade(symbol, side, qty, price) is the only write path.
    It handles:
      - New position (flat → long/short)
      - Adding to existing position (same direction)
      - Partial close
      - Full close
      - Position flip (close + reopen in opposite direction)
    """

    def __init__(self) -> None:
        self._positions: Dict[str, Position] = {}
        self._lock = threading.Lock()

    # ─────────────────────── Public API ───────────────────────

    def apply_trade(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        price: float,
    ) -> Position:
        """Apply a trade and return the updated Position."""
        signed_qty = qty if side == "buy" else -qty

        with self._lock:
            pos = self._positions.setdefault(symbol, Position(symbol=symbol))
            pos = self._update(pos, signed_qty, price)
            self._positions[symbol] = pos
            return pos

    def get_position(self, symbol: str) -> Optional[Position]:
        with self._lock:
            return self._positions.get(symbol)

    def all_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    def open_positions(self) -> list[Position]:
        with self._lock:
            return [p for p in self._positions.values() if p.quantity != 0]

    def close_all(self, prices: Dict[str, float]) -> None:
        """Mark-to-market close all positions at given prices."""
        with self._lock:
            for sym, pos in self._positions.items():
                if pos.quantity != 0 and sym in prices:
                    close_price = prices[sym]
                    pos.realized_pnl += pos.unrealized_pnl(close_price)
                    pos.quantity = 0.0
                    pos.avg_entry = 0.0

    # ─────────────────────── Internal ───────────────────────

    @staticmethod
    def _update(pos: Position, signed_qty: float, price: float) -> Position:
        """
        Core position update logic.

        Cases:
          A) Flat → open in direction of signed_qty
          B) Same direction → scale in, weighted avg entry
          C) Opposite direction, smaller than position → partial close
          D) Opposite direction, equal → full close
          E) Opposite direction, larger → flip
        """
        if pos.quantity == 0.0:
            # Case A: fresh open
            pos.quantity = signed_qty
            pos.avg_entry = price
            return pos

        same_direction = (pos.quantity > 0 and signed_qty > 0) or \
                         (pos.quantity < 0 and signed_qty < 0)

        if same_direction:
            # Case B: add to position, update weighted avg entry
            total_cost = pos.avg_entry * abs(pos.quantity) + price * abs(signed_qty)
            new_qty = pos.quantity + signed_qty
            pos.avg_entry = total_cost / abs(new_qty)
            pos.quantity = new_qty
            return pos

        # Opposite direction — closing some or all
        close_qty = min(abs(pos.quantity), abs(signed_qty))
        # P&L on the closed portion
        if pos.quantity > 0:
            pnl = close_qty * (price - pos.avg_entry)
        else:
            pnl = close_qty * (pos.avg_entry - price)
        pos.realized_pnl += pnl

        remaining_pos = abs(pos.quantity) - close_qty
        remaining_trade = abs(signed_qty) - close_qty

        if remaining_pos > 0:
            # Case C: partial close — keep same avg_entry, reduce qty
            pos.quantity = remaining_pos * (1 if pos.quantity > 0 else -1)
        elif remaining_trade > 0:
            # Case E: flip — new position in opposite direction
            pos.quantity = remaining_trade * (1 if signed_qty > 0 else -1)
            pos.avg_entry = price
        else:
            # Case D: full close
            pos.quantity = 0.0
            pos.avg_entry = 0.0

        return pos
