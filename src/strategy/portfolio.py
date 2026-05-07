"""Portfolio state tracker for the strategy engine.

Maintains a real-time book of cash, positions, and equity with full history.
Designed to be fast (no DB calls) and thread-safe for single-threaded strategy
loops, with mark-to-market updates on every new price bar.

Distinct from `src/execution/paper_trader.PositionTracker` (which is the
execution-layer paper trading book).  This class is the strategy layer's view
of the portfolio — it feeds into risk checks, position sizing, and signal
generation.

Usage::

    port = Portfolio(initial_capital=100_000)
    port.apply_trade("AAPL", "buy",  10, 175.30)
    port.apply_trade("TSLA", "buy",   5, 240.00)
    port.mark_to_market({"AAPL": 178.00, "TSLA": 235.00})

    print(port.equity)           # 100_000 + unrealised P&L
    print(port.drawdown)         # peak-to-trough as a negative fraction
    print(port.rolling_vol(20))  # 20-day annualised volatility of equity returns
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


# ─────────────────────────── Sub-objects ──────────────────────────────────

@dataclass
class PortfolioPosition:
    symbol: str
    quantity: float = 0.0        # signed: positive = long, negative = short
    avg_entry: float = 0.0
    last_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        if self.last_price == 0:
            return 0.0
        return self.quantity * (self.last_price - self.avg_entry)

    @property
    def market_value(self) -> float:
        """Signed market value (negative for short positions)."""
        return self.quantity * self.last_price

    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.avg_entry

    @property
    def pnl_pct(self) -> float:
        if self.avg_entry == 0:
            return 0.0
        return (self.last_price - self.avg_entry) / self.avg_entry


@dataclass
class EquitySnapshot:
    timestamp: str
    equity: float
    cash: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown: float          # fraction, ≤ 0
    n_positions: int


# ─────────────────────────── Portfolio ────────────────────────────────────

class Portfolio:
    """Real-time portfolio state for the strategy engine.

    Parameters
    ----------
    initial_capital : float
        Starting cash balance (default 100 000).
    snapshot_every : int
        Record an EquitySnapshot every N mark_to_market calls (default 1).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        snapshot_every: int = 1,
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, PortfolioPosition] = {}
        self._peak_equity = initial_capital
        self._realized_pnl = 0.0
        self._equity_history: list[float] = [initial_capital]
        self._snapshots: list[EquitySnapshot] = []
        self._snapshot_every = snapshot_every
        self._mark_count = 0

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def equity(self) -> float:
        """Total equity: cash + market value of all open positions.

        Uses market_value (quantity × last_price) rather than unrealised P&L
        so the number stays correct when last_price differs from avg_entry.
        """
        return self._cash + sum(p.market_value for p in self._positions.values())

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def drawdown(self) -> float:
        """Current peak-to-trough drawdown as a negative fraction (e.g. -0.12)."""
        if self._peak_equity <= 0:
            return 0.0
        dd = (self.equity - self._peak_equity) / self._peak_equity
        return min(dd, 0.0)

    @property
    def positions(self) -> dict[str, PortfolioPosition]:
        return {s: p for s, p in self._positions.items() if p.quantity != 0}

    @property
    def open_symbols(self) -> list[str]:
        return list(self.positions.keys())

    @property
    def equity_curve(self) -> list[float]:
        return list(self._equity_history)

    @property
    def snapshots(self) -> list[EquitySnapshot]:
        return list(self._snapshots)

    def position(self, symbol: str) -> Optional[PortfolioPosition]:
        return self._positions.get(symbol)

    def gross_exposure(self) -> float:
        """Sum of |market_value| of all positions."""
        return sum(abs(p.market_value) for p in self._positions.values())

    def net_exposure(self) -> float:
        """Sum of signed market_value (long - short)."""
        return sum(p.market_value for p in self._positions.values())

    # ── Write ──────────────────────────────────────────────────────────────

    def apply_trade(
        self,
        symbol: str,
        side: str,        # "buy" | "sell"
        quantity: float,
        price: float,
    ) -> None:
        """Apply a trade to the portfolio.  Handles open, add, reduce, flip."""
        if quantity <= 0 or price <= 0:
            return

        signed_qty = quantity if side == "buy" else -quantity

        # Cash settlement: short proceeds come in, long costs go out
        self._cash -= signed_qty * price

        pos = self._positions.setdefault(symbol, PortfolioPosition(symbol=symbol))
        pos.last_price = price
        self._update_position(pos, signed_qty, price)

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update last prices for all positions and record equity snapshot."""
        for symbol, price in prices.items():
            if symbol in self._positions and price > 0:
                self._positions[symbol].last_price = price

        eq = self.equity
        self._equity_history.append(eq)
        if eq > self._peak_equity:
            self._peak_equity = eq

        self._mark_count += 1
        if self._mark_count % self._snapshot_every == 0:
            self._snapshots.append(EquitySnapshot(
                timestamp     = datetime.now(tz=timezone.utc).isoformat(),
                equity        = eq,
                cash          = self._cash,
                unrealized_pnl= self.unrealized_pnl,
                realized_pnl  = self._realized_pnl,
                drawdown      = self.drawdown,
                n_positions   = len(self.positions),
            ))

    def reset(self) -> None:
        """Reset to initial state (for backtesting reset)."""
        self._cash = self._initial_capital
        self._positions.clear()
        self._peak_equity = self._initial_capital
        self._realized_pnl = 0.0
        self._equity_history = [self._initial_capital]
        self._snapshots.clear()
        self._mark_count = 0

    # ── Analytics ─────────────────────────────────────────────────────────

    def rolling_vol(self, window: int = 20, annualisation: int = 252) -> float:
        """Annualised rolling volatility of equity returns over last `window` bars."""
        hist = self._equity_history
        if len(hist) < window + 1:
            return 0.0
        arr = np.array(hist[-(window + 1):], dtype=float)
        returns = np.diff(arr) / arr[:-1]
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * math.sqrt(annualisation))

    def sharpe(self, window: int = 252, annualisation: int = 252) -> float:
        """Rolling Sharpe ratio over the last `window` equity bars."""
        hist = self._equity_history
        n = min(window + 1, len(hist))
        if n < 3:
            return 0.0
        arr = np.array(hist[-n:], dtype=float)
        r = np.diff(arr) / arr[:-1]
        std = float(r.std())
        if std < 1e-12:
            return 0.0
        return float(r.mean() / std * math.sqrt(annualisation))

    def max_drawdown(self) -> float:
        """Maximum drawdown over the full equity history."""
        hist = np.array(self._equity_history, dtype=float)
        if len(hist) < 2:
            return 0.0
        peak = np.maximum.accumulate(hist)
        dd = (hist - peak) / np.where(peak > 0, peak, 1.0)
        return float(dd.min())

    def total_return(self) -> float:
        """Total return from inception as a fraction."""
        if self._initial_capital <= 0:
            return 0.0
        return (self.equity - self._initial_capital) / self._initial_capital

    def summary(self) -> dict:
        return {
            "equity":         round(self.equity, 2),
            "cash":           round(self._cash, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl":   round(self._realized_pnl, 2),
            "total_return":   round(self.total_return(), 4),
            "drawdown":       round(self.drawdown, 4),
            "max_drawdown":   round(self.max_drawdown(), 4),
            "sharpe":         round(self.sharpe(), 4),
            "rolling_vol_20": round(self.rolling_vol(20), 4),
            "n_positions":    len(self.positions),
            "gross_exposure": round(self.gross_exposure(), 2),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _update_position(
        self,
        pos: PortfolioPosition,
        signed_qty: float,
        price: float,
    ) -> None:
        """Core position bookkeeping (same logic as paper_trader)."""
        if pos.quantity == 0.0:
            pos.quantity  = signed_qty
            pos.avg_entry = price
            return

        same_dir = (pos.quantity > 0 and signed_qty > 0) or \
                   (pos.quantity < 0 and signed_qty < 0)

        if same_dir:
            # Scale in — weighted avg entry
            total_cost   = pos.avg_entry * abs(pos.quantity) + price * abs(signed_qty)
            pos.quantity += signed_qty
            pos.avg_entry = total_cost / abs(pos.quantity) if pos.quantity != 0 else price
            return

        # Opposite direction — closing
        close_qty     = min(abs(pos.quantity), abs(signed_qty))
        close_pnl     = close_qty * (price - pos.avg_entry) * (1 if pos.quantity > 0 else -1)
        pos.realized_pnl  += close_pnl
        self._realized_pnl += close_pnl

        remaining_pos   = abs(pos.quantity) - close_qty
        remaining_trade = abs(signed_qty)   - close_qty

        if remaining_pos > 0:
            pos.quantity = remaining_pos * (1 if pos.quantity > 0 else -1)
        elif remaining_trade > 0:
            pos.quantity  = remaining_trade * (1 if signed_qty > 0 else -1)
            pos.avg_entry = price
        else:
            pos.quantity  = 0.0
            pos.avg_entry = 0.0
