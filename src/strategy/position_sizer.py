"""Position sizing: translate a trade signal into a share/unit quantity.

Four sizing methods, each appropriate for different scenarios:

  FIXED_FRACTIONAL   — Risk a fixed fraction of capital per trade.
                       Simple, robust, widely used baseline.

  VOL_TARGET         — Scale position so the position's expected dollar vol
                       equals a target fraction of capital.  Naturally reduces
                       size during high-volatility regimes.
                       `quantity = (vol_target * capital) / (asset_vol * price)`

  KELLY              — Full / fractional Kelly based on historical win-rate and
                       average win/loss.  Theoretically optimal but requires
                       reliable edge estimates; use fraction ≤ 0.5 in practice.

  CONFIDENCE_WEIGHTED — VOL_TARGET × signal.confidence.  The recommended default
                        for this system: integrates model uncertainty directly
                        into size, so high-confidence predictions get full size
                        and low-confidence ones are automatically dialled back.

All methods enforce:
  • minimum quantity  (default 1 share / unit)
  • maximum position value as a fraction of capital
  • a hard leverage cap
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np


# ─────────────────────────── Config ────────────────────────────────────────

class SizingMethod(Enum):
    FIXED_FRACTIONAL     = auto()
    VOL_TARGET           = auto()
    KELLY                = auto()
    CONFIDENCE_WEIGHTED  = auto()   # vol-target × confidence (recommended)


@dataclass
class SizingConfig:
    method: SizingMethod = SizingMethod.CONFIDENCE_WEIGHTED

    # FIXED_FRACTIONAL
    risk_fraction: float = 0.02        # 2 % of capital risked per trade

    # VOL_TARGET & CONFIDENCE_WEIGHTED
    vol_target: float = 0.15           # 15 % annualised vol target
    annualisation: int = 252           # trading days per year
    fallback_vol: float = 0.20         # used when realized vol is unavailable

    # KELLY
    kelly_fraction: float = 0.50       # half-Kelly for safety

    # Universal guards
    max_position_pct: float = 0.20     # ≤ 20 % of capital in any single position
    max_leverage: float = 1.0          # no leverage by default
    min_quantity: float = 1.0          # minimum order size (shares / units)
    round_lots: bool = False           # if True, round to nearest integer


# ─────────────────────────── Sizer ────────────────────────────────────────

class PositionSizer:
    """Compute order quantity from signal + portfolio state.

    Parameters
    ----------
    config : SizingConfig
        Sizing parameters (method, targets, guards).

    Usage::

        sizer = PositionSizer()
        qty = sizer.size(
            signal        = generated_signal,
            capital       = portfolio.equity,
            current_price = 175.30,
            realized_vol  = 0.18,   # daily vol × √252 for annualised
            current_qty   = 0,
        )
    """

    def __init__(self, config: Optional[SizingConfig] = None) -> None:
        self.cfg = config or SizingConfig()

    # ── Main dispatch ─────────────────────────────────────────────────────

    def size(
        self,
        signal,                         # GeneratedSignal
        capital: float,
        current_price: float,
        realized_vol: float = 0.0,      # annualised, e.g. 0.18 = 18 %
        current_qty: float = 0.0,       # existing position in same symbol (+/-)
        win_rate: float = 0.55,         # for Kelly only
        avg_win: float = 0.015,         # for Kelly only
        avg_loss: float = 0.010,        # for Kelly only
    ) -> float:
        """Return the *absolute* quantity to transact (always ≥ 0).

        The caller is responsible for attaching the correct side (buy/sell).
        Returns 0.0 if the position should not be opened.
        """
        if capital <= 0 or current_price <= 0:
            return 0.0

        vol = realized_vol if realized_vol > 1e-6 else self.cfg.fallback_vol

        method = self.cfg.method
        if method == SizingMethod.FIXED_FRACTIONAL:
            raw_qty = self._fixed_fractional(capital, current_price)
        elif method == SizingMethod.VOL_TARGET:
            raw_qty = self._vol_target(capital, current_price, vol)
        elif method == SizingMethod.KELLY:
            raw_qty = self._kelly(capital, current_price, win_rate, avg_win, avg_loss)
        else:  # CONFIDENCE_WEIGHTED (default)
            raw_qty = self._confidence_weighted(
                capital, current_price, vol, signal.confidence
            )

        return self._apply_guards(raw_qty, capital, current_price, current_qty)

    # ── Sizing methods ────────────────────────────────────────────────────

    def _fixed_fractional(self, capital: float, price: float) -> float:
        """Risk a fixed fraction of capital per trade."""
        return (self.cfg.risk_fraction * capital) / price

    def _vol_target(self, capital: float, price: float, ann_vol: float) -> float:
        """Scale so that position dollar-vol ≈ vol_target × capital.

        dollar_vol_per_share = price × daily_vol
        daily_vol            = ann_vol / sqrt(annualisation)
        target_dollar_vol    = vol_target × capital / sqrt(annualisation)

        quantity = target_dollar_vol / dollar_vol_per_share
        """
        daily_vol = ann_vol / math.sqrt(self.cfg.annualisation)
        if daily_vol < 1e-8:
            return 0.0
        target_dv = (self.cfg.vol_target / math.sqrt(self.cfg.annualisation)) * capital
        return target_dv / (price * daily_vol)

    def _kelly(
        self,
        capital: float,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Fractional Kelly criterion.

        f* = win_rate - (1 - win_rate) / b
        where b = avg_win / avg_loss (win-loss ratio).
        """
        if avg_loss <= 1e-8 or avg_win <= 1e-8:
            return 0.0
        b = avg_win / avg_loss
        f_star = win_rate - (1.0 - win_rate) / b
        f_star = max(0.0, f_star) * self.cfg.kelly_fraction
        return (f_star * capital) / price

    def _confidence_weighted(
        self,
        capital: float,
        price: float,
        ann_vol: float,
        confidence: float,
    ) -> float:
        """Vol-targeted size scaled by signal confidence (recommended default)."""
        base = self._vol_target(capital, price, ann_vol)
        return base * max(0.0, min(1.0, confidence))

    # ── Guards ────────────────────────────────────────────────────────────

    def _apply_guards(
        self,
        qty: float,
        capital: float,
        price: float,
        current_qty: float,
    ) -> float:
        """Apply max position, leverage, and minimum guards."""
        if qty <= 0:
            return 0.0

        # Max position value
        max_qty_pct = (self.cfg.max_position_pct * capital) / price
        qty = min(qty, max_qty_pct)

        # Max leverage (total position value ≤ leverage × capital)
        max_qty_lev = (self.cfg.max_leverage * capital) / price
        qty = min(qty, max_qty_lev)

        # Don't over-size if we already have a position
        if current_qty != 0:
            already_held_pct = abs(current_qty) * price / capital
            headroom = self.cfg.max_position_pct - already_held_pct
            if headroom <= 0:
                return 0.0
            qty = min(qty, headroom * capital / price)

        # Minimum
        if qty < self.cfg.min_quantity:
            return 0.0

        if self.cfg.round_lots:
            qty = math.floor(qty)

        return qty

    # ── Reduce-only sizing ────────────────────────────────────────────────

    def size_to_close(self, current_qty: float) -> float:
        """Return quantity needed to fully close an existing position."""
        return abs(current_qty)

    def size_to_reduce(
        self,
        current_qty: float,
        target_pct: float,
        capital: float,
        price: float,
    ) -> float:
        """Size a partial reduce so the remaining position = target_pct of capital."""
        target_qty = (target_pct * capital) / price
        delta = abs(current_qty) - target_qty
        return max(0.0, delta)
