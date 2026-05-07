"""Risk management: constraints that sit between the signal and the order.

Three layers of protection:

  1. CIRCUIT BREAKER — Portfolio-level max drawdown.
     If portfolio drawdown exceeds `max_drawdown_limit`, trading is halted
     until manually reset.  All new long/short orders are blocked; only
     flat/close orders pass through.

  2. STOP-LOSS — Per-position loss limit.
     If any open position's unrealised P&L drops below `stop_loss_pct` of
     its cost basis, a close order is queued.  Supports both fixed stop-loss
     and trailing stop (tracked internally).

  3. VOLATILITY TARGETING — Adaptive position scalar.
     If realised portfolio vol exceeds the target, a scalar < 1 is applied
     to all new position sizes.  Naturally reduces exposure during turbulent
     regimes.

  4. CONCENTRATION LIMIT — Per-symbol max exposure.
     Any order that would push a single position above `max_position_pct` of
     current equity is clipped or rejected.

Usage::

    cfg = RiskConfig(max_drawdown_limit=0.15, stop_loss_pct=0.05)
    rm  = RiskManager(cfg)

    # Before placing any order:
    if not rm.can_trade(portfolio):
        return None

    # Size scalar based on vol:
    scaled_qty = raw_qty * rm.vol_scalar(portfolio.rolling_vol(20))

    # After each price bar, check for stop-losses:
    stops = rm.check_stops(portfolio)   # → list of symbols to close
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────── Config ───────────────────────────────────────

@dataclass
class RiskConfig:
    # ── Circuit breaker ─────────────────────────────────────────────────
    max_drawdown_limit: float = 0.15    # halt trading if DD > 15 %

    # ── Per-position stop-loss ───────────────────────────────────────────
    stop_loss_pct: float = 0.05         # close if position loss > 5 %
    use_trailing_stop: bool = True      # trail the stop up with the position
    trailing_stop_pct: float = 0.07     # trail 7 % below rolling peak

    # ── Volatility targeting ─────────────────────────────────────────────
    vol_target: float = 0.15           # 15 % annualised target
    vol_lookback: int = 20             # days for rolling vol estimate

    # ── Concentration ────────────────────────────────────────────────────
    max_position_pct: float = 0.20     # no single position > 20 % of equity

    # ── Daily loss limit (optional) ──────────────────────────────────────
    max_daily_loss_pct: float = 0.03   # halt if today's P&L < -3 % of equity


# ─────────────────────────── Internal state ───────────────────────────────

@dataclass
class _StopState:
    """Tracks trailing-stop high-water mark for one position."""
    symbol: str
    entry_price: float
    direction: int                  # 1 = long, -1 = short
    peak_price: float = 0.0         # trailing HWM (long) or trough (short)

    def update_peak(self, price: float) -> None:
        if self.direction == 1:
            self.peak_price = max(self.peak_price, price)
        else:
            self.peak_price = min(self.peak_price, price) if self.peak_price != 0 else price

    def is_stopped(self, current_price: float, stop_pct: float, trailing: bool) -> bool:
        if trailing and self.peak_price > 0:
            # Trailing: stop is X% below the peak (long) or above trough (short)
            if self.direction == 1:
                stop_level = self.peak_price * (1.0 - stop_pct)
                return current_price <= stop_level
            else:
                stop_level = self.peak_price * (1.0 + stop_pct)
                return current_price >= stop_level
        else:
            # Fixed: stop is X% below/above entry
            if self.direction == 1:
                stop_level = self.entry_price * (1.0 - stop_pct)
                return current_price <= stop_level
            else:
                stop_level = self.entry_price * (1.0 + stop_pct)
                return current_price >= stop_level


# ─────────────────────────── RiskManager ──────────────────────────────────

class RiskManager:
    """Applies risk constraints to proposed orders and portfolio state.

    Parameters
    ----------
    config : RiskConfig
        Risk parameters.  Defaults give reasonable live-trading guardrails.
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.cfg = config or RiskConfig()
        self._halted: bool = False
        self._halt_reason: str = ""
        self._stops: dict[str, _StopState] = {}
        self._daily_start_equity: float = 0.0
        self._today_pnl: float = 0.0

    # ── Circuit breaker ────────────────────────────────────────────────────

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def can_trade(self, portfolio) -> bool:
        """Return True if new directional orders are permitted.

        Checks portfolio-level drawdown and daily loss limit.
        """
        # Max drawdown circuit breaker
        dd = abs(portfolio.drawdown)
        if dd >= self.cfg.max_drawdown_limit:
            self._halt("max_drawdown", f"drawdown {dd:.1%} ≥ limit {self.cfg.max_drawdown_limit:.1%}")
            return False

        # Daily loss limit
        if self._daily_start_equity > 0:
            daily_loss_pct = (portfolio.equity - self._daily_start_equity) / self._daily_start_equity
            if daily_loss_pct <= -self.cfg.max_daily_loss_pct:
                self._halt("daily_loss", f"daily loss {daily_loss_pct:.1%}")
                return False

        if self._halted:
            return False

        return True

    def reset_halt(self) -> None:
        """Manually clear a trading halt (e.g. after manual review)."""
        self._halted = False
        self._halt_reason = ""

    def start_day(self, portfolio) -> None:
        """Record start-of-day equity for daily loss tracking."""
        self._daily_start_equity = portfolio.equity

    # ── Stop-loss tracking ─────────────────────────────────────────────────

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        direction: int,       # 1 = long, -1 = short
    ) -> None:
        """Register a new position for stop-loss tracking."""
        self._stops[symbol] = _StopState(
            symbol      = symbol,
            entry_price = entry_price,
            direction   = direction,
            peak_price  = entry_price,
        )

    def deregister_position(self, symbol: str) -> None:
        """Remove stop-loss tracking when a position is closed."""
        self._stops.pop(symbol, None)

    def update_stops(self, prices: dict[str, float]) -> None:
        """Update trailing stop HWMs for all tracked positions."""
        for sym, state in self._stops.items():
            if sym in prices:
                state.update_peak(prices[sym])

    def check_stops(self, portfolio, prices: dict[str, float]) -> list[str]:
        """Return symbols that have breached their stop-loss level.

        Caller should generate close orders for each returned symbol.
        """
        self.update_stops(prices)
        triggered: list[str] = []

        for sym, pos in portfolio.positions.items():
            if sym not in prices:
                continue
            cp = prices[sym]

            # ── Fixed stop: check cost basis loss ────────────────────────
            if pos.avg_entry > 0:
                loss_pct = (pos.avg_entry - cp) / pos.avg_entry if pos.quantity > 0 else \
                           (cp - pos.avg_entry) / pos.avg_entry
                if loss_pct >= self.cfg.stop_loss_pct:
                    triggered.append(sym)
                    continue

            # ── Trailing stop ────────────────────────────────────────────
            if self.cfg.use_trailing_stop and sym in self._stops:
                state = self._stops[sym]
                if state.is_stopped(cp, self.cfg.trailing_stop_pct, trailing=True):
                    triggered.append(sym)

        return list(set(triggered))  # deduplicate

    # ── Volatility targeting ───────────────────────────────────────────────

    def vol_scalar(self, realized_vol: float) -> float:
        """Scale factor to apply to raw position size.

        Returns a number in (0, 1] that shrinks positions when realised vol
        exceeds the target.  Returns 1.0 (no scaling) when vol is below target.

        Parameters
        ----------
        realized_vol : float
            Annualised realised volatility (e.g. portfolio.rolling_vol(20)).
        """
        if realized_vol <= 0:
            return 1.0
        raw = self.cfg.vol_target / realized_vol
        # Cap at 1.0 (never lever up via this scalar)
        return min(1.0, raw)

    # ── Concentration check ────────────────────────────────────────────────

    def check_concentration(
        self,
        symbol: str,
        proposed_qty: float,
        price: float,
        portfolio,
    ) -> float:
        """Clip proposed_qty so the resulting position ≤ max_position_pct.

        Returns the (possibly reduced) quantity.
        """
        equity = portfolio.equity
        if equity <= 0:
            return 0.0

        existing = portfolio.position(symbol)
        existing_value = abs(existing.market_value) if existing else 0.0
        proposed_value = proposed_qty * price
        total_value    = existing_value + proposed_value
        max_value      = self.cfg.max_position_pct * equity

        if total_value <= max_value:
            return proposed_qty

        headroom = max(0.0, max_value - existing_value)
        return headroom / price

    # ── Combined order validation ──────────────────────────────────────────

    def validate_order(
        self,
        symbol: str,
        side: str,          # "buy" | "sell"
        quantity: float,
        price: float,
        portfolio,
        is_close: bool = False,
    ) -> float:
        """Full order validation pipeline.

        Runs circuit breaker → concentration check → vol scalar.

        Args:
            symbol:    Ticker.
            side:      "buy" | "sell".
            quantity:  Proposed quantity (positive).
            price:     Current price.
            portfolio: Portfolio instance.
            is_close:  If True, skip circuit breaker (allow closing even when halted).

        Returns:
            Validated (possibly reduced) quantity, or 0.0 if the order is blocked.
        """
        if not is_close and not self.can_trade(portfolio):
            return 0.0

        qty = self.check_concentration(symbol, quantity, price, portfolio)
        if qty <= 0:
            return 0.0

        if not is_close:
            scalar = self.vol_scalar(portfolio.rolling_vol(self.cfg.vol_lookback))
            qty *= scalar

        return qty

    # ── Internal ───────────────────────────────────────────────────────────

    def _halt(self, reason: str, detail: str) -> None:
        if not self._halted:
            self._halted = True
            self._halt_reason = f"{reason}: {detail}"

    def status(self) -> dict:
        return {
            "halted":             self._halted,
            "halt_reason":        self._halt_reason,
            "tracked_stops":      list(self._stops.keys()),
            "vol_target":         self.cfg.vol_target,
            "max_drawdown_limit": self.cfg.max_drawdown_limit,
            "stop_loss_pct":      self.cfg.stop_loss_pct,
            "trailing_stop":      self.cfg.use_trailing_stop,
            "trailing_stop_pct":  self.cfg.trailing_stop_pct,
        }
