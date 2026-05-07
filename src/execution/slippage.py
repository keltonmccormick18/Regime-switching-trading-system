"""Slippage models for the simulated execution engine.

Three models of increasing sophistication:

  FixedSlippage          — constant X bps on every order regardless of size.
                           Fast, simple, good baseline for liquid large-caps.

  VolatilitySlippage     — slippage scales with realised volatility.
                           High-vol regimes are harder to fill cleanly.
                           `impact = vol_factor × daily_vol × price`

  SquareRootImpact       — Almgren–Chriss square-root market impact model.
                           Widely used in academic and industry research.
                           `impact = η × σ × sqrt(Q / ADV) × price`
                           where σ = daily vol, Q = trade size, ADV = avg daily volume.
                           Falls back to FixedSlippage when volume is unavailable.

All models return (fill_price, slippage_cost_dollars):
  fill_price      = mid_price ± slippage   (+ for buys, - for sells)
  slippage_cost   = slippage_dollars ≥ 0   (always a cost, sign is built into fill_price)

Usage::

    model = SquareRootImpact(eta=0.1)
    fill_price, cost = model.apply(
        side       = "buy",
        price      = 175.30,
        quantity   = 100,
        volatility = 0.18,    # annualised
        adv        = 50_000_000 / 175.30,  # ADV in shares
    )
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────── Abstract base ─────────────────────────────────

class SlippageModel(ABC):
    """Base class for all slippage models."""

    @abstractmethod
    def apply(
        self,
        side:       str,            # "buy" | "sell"
        price:      float,          # mid-market price
        quantity:   float,          # shares / units
        volatility: float = 0.20,   # annualised realised vol (fraction)
        adv:        float = 0.0,    # average daily volume in shares (0 = unknown)
    ) -> tuple[float, float]:
        """Return (fill_price, slippage_cost_dollars)."""

    def direction(self, side: str) -> int:
        """+1 for buy (slippage pushes price up), -1 for sell."""
        return 1 if side == "buy" else -1


# ─────────────────────────── Fixed ─────────────────────────────────────────

@dataclass
class FixedSlippage(SlippageModel):
    """Constant slippage in basis points regardless of order size.

    Parameters
    ----------
    bps : float
        One-way slippage in basis points (default 5 = 0.05%).
    add_noise : bool
        If True, add small random noise (±25% of bps) so fills look realistic.
    """
    bps:       float = 5.0
    add_noise: bool  = True

    def apply(self, side, price, quantity, volatility=0.20, adv=0.0):
        slip_frac = self.bps / 10_000
        if self.add_noise:
            slip_frac *= (1.0 + random.uniform(-0.25, 0.25))
        slip_frac = max(0.0, slip_frac)

        fill_price = price * (1.0 + self.direction(side) * slip_frac)
        cost       = abs(fill_price - price) * quantity
        return fill_price, cost


# ─────────────────────────── Volatility-based ──────────────────────────────

@dataclass
class VolatilitySlippage(SlippageModel):
    """Slippage proportional to the current realised volatility.

    Markets are harder to fill cleanly when they are moving fast.

    Parameters
    ----------
    vol_factor : float
        Multiplier on daily vol.  Default 0.1 → at 20% ann vol, daily vol
        ≈ 1.26%, so slippage ≈ 0.126%.
    annualisation : int
        Trading days per year (default 252).
    add_noise : bool
        Random multiplicative noise in [0.5, 1.5] on the slip fraction.
    """
    vol_factor:    float = 0.10
    annualisation: int   = 252
    add_noise:     bool  = True

    def apply(self, side, price, quantity, volatility=0.20, adv=0.0):
        daily_vol  = volatility / math.sqrt(self.annualisation)
        slip_frac  = self.vol_factor * daily_vol
        if self.add_noise:
            slip_frac *= random.uniform(0.5, 1.5)
        slip_frac = max(0.0, slip_frac)

        fill_price = price * (1.0 + self.direction(side) * slip_frac)
        cost       = abs(fill_price - price) * quantity
        return fill_price, cost


# ─────────────────────────── Square-root impact ────────────────────────────

@dataclass
class SquareRootImpact(SlippageModel):
    """Almgren–Chriss square-root market impact model.

    `impact_frac = η × σ_daily × sqrt(Q / ADV)`

    where:
      η   = market impact coefficient (typical range 0.05–0.3)
      σ   = daily volatility
      Q   = trade size in shares
      ADV = average daily volume in shares

    Falls back to FixedSlippage(bps=fallback_bps) when ADV is unknown.

    Parameters
    ----------
    eta : float
        Market impact coefficient.  Lower = more liquid.  Default 0.1.
    annualisation : int
        Trading days per year.
    fallback_bps : float
        Basis points used when ADV is unavailable.
    add_noise : bool
        Multiplicative noise in [0.7, 1.3] to simulate fill uncertainty.
    """
    eta:           float = 0.10
    annualisation: int   = 252
    fallback_bps:  float = 8.0
    add_noise:     bool  = True
    _fallback:     FixedSlippage = None   # type: ignore

    def __post_init__(self):
        self._fallback = FixedSlippage(bps=self.fallback_bps, add_noise=self.add_noise)

    def apply(self, side, price, quantity, volatility=0.20, adv=0.0):
        if adv <= 0:
            return self._fallback.apply(side, price, quantity, volatility, adv)

        daily_vol  = volatility / math.sqrt(self.annualisation)
        slip_frac  = self.eta * daily_vol * math.sqrt(quantity / adv)
        if self.add_noise:
            slip_frac *= random.uniform(0.7, 1.3)
        slip_frac = max(0.0, slip_frac)

        fill_price = price * (1.0 + self.direction(side) * slip_frac)
        cost       = abs(fill_price - price) * quantity
        return fill_price, cost


# ─────────────────────────── Factory ───────────────────────────────────────

def make_slippage_model(name: str, **kwargs) -> SlippageModel:
    """Construct a slippage model by name.

    Args:
        name:   "fixed" | "volatility" | "sqrt_impact"
        kwargs: Passed to the model constructor.

    Returns:
        SlippageModel instance.
    """
    name = name.lower().strip()
    if name in ("fixed", "constant"):
        return FixedSlippage(**kwargs)
    if name in ("volatility", "vol"):
        return VolatilitySlippage(**kwargs)
    if name in ("sqrt_impact", "square_root", "almgren_chriss"):
        return SquareRootImpact(**kwargs)
    raise ValueError(
        f"Unknown slippage model '{name}'. "
        "Choose from: fixed, volatility, sqrt_impact"
    )
