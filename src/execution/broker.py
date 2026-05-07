"""Simulated broker: apply transaction costs, slippage, and latency to orders.

The SimulatedBroker is the single point that converts an Order into a Fill.
Both the BacktestEngine and PaperEngine route every order through it, so
realism settings are centralised and consistent across modes.

Cost model
----------
  fill_price  = mid_price ± slippage ± half_spread
  commission  = max(commission_min,
                    commission_per_share × qty + commission_pct × notional)
  total_cost  = commission + slippage_cost + spread_cost

Latency model
-------------
  latency_ms ~ LogNormal(μ, σ) clipped to [min_ms, max_ms]

  In backtest mode, latency means the fill uses the NEXT bar's open price
  rather than the signal bar's close.  In paper mode, it means an actual
  asyncio.sleep() before the fill is confirmed.

Usage::

    cfg    = BrokerConfig(slippage_model="sqrt_impact", commission_per_share=0.005)
    broker = SimulatedBroker(cfg)

    order = Order(symbol="AAPL", side="buy", quantity=100)
    report = broker.submit(order, market_price=175.30, volatility=0.18, adv=8e6)

    print(report.fill.fill_price)    # 175.30 + slippage + spread
    print(report.fill.total_cost)    # commission + slippage_cost + spread_cost
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.execution.order import ExecutionReport, Fill, Order
from src.execution.slippage import SlippageModel, make_slippage_model


# ─────────────────────────── Config ────────────────────────────────────────

@dataclass
class BrokerConfig:
    # ── Commission structure ─────────────────────────────────────────────
    commission_per_share: float = 0.005      # $0.005 per share (Interactive Brokers-like)
    commission_pct:       float = 0.0        # fraction of notional (alternative)
    commission_min:       float = 1.0        # minimum per order

    # ── Bid-ask spread ───────────────────────────────────────────────────
    spread_bps:           float = 5.0        # half-spread in bps (full spread = 10 bps)

    # ── Slippage ─────────────────────────────────────────────────────────
    slippage_model:       str   = "sqrt_impact"  # "fixed" | "volatility" | "sqrt_impact"
    slippage_bps:         float = 5.0        # for FixedSlippage
    slippage_vol_factor:  float = 0.10       # for VolatilitySlippage
    slippage_eta:         float = 0.10       # for SquareRootImpact

    # ── Latency ──────────────────────────────────────────────────────────
    latency_mean_ms:      float = 50.0       # mean latency (ms)
    latency_std_ms:       float = 20.0       # std of latency  (log-normal σ)
    latency_min_ms:       float = 5.0        # hard floor
    latency_max_ms:       float = 500.0      # hard ceiling (outlier spike)
    simulate_latency:     bool  = True

    # ── Fill probability (partial fills / rejects) ───────────────────────
    fill_probability:     float = 1.0        # 1.0 = always fill
    partial_fill_prob:    float = 0.0        # probability of a partial fill (0 = never)
    partial_fill_pct:     float = 0.80       # fraction filled when partial

    # ── Other ────────────────────────────────────────────────────────────
    allow_shorting:       bool  = True
    reject_if_no_cash:    bool  = False      # if True, reject orders exceeding cash


# ─────────────────────────── SimulatedBroker ──────────────────────────────

class SimulatedBroker:
    """Convert Orders into Fills with configurable realism.

    Parameters
    ----------
    config : BrokerConfig
        All cost, slippage, and latency parameters.
    """

    def __init__(self, config: Optional[BrokerConfig] = None) -> None:
        self.cfg = config or BrokerConfig()
        self._slippage_model: SlippageModel = self._build_slippage_model()
        self._fill_history: list[Fill] = []

    # ── Main entry point ──────────────────────────────────────────────────

    def submit(
        self,
        order:      Order,
        market_price: float,          # mid-market price at time of submission
        volatility:   float = 0.20,   # annualised realised vol
        adv:          float = 0.0,    # average daily volume (shares)
        cash:         float = float("inf"),  # available cash for reject_if_no_cash
    ) -> ExecutionReport:
        """Submit an order and return an ExecutionReport with simulated fill.

        Args:
            order:        The order to execute.
            market_price: Current mid-market price.
            volatility:   Annualised realised vol (used by slippage models).
            adv:          Average daily volume in shares (for sqrt-impact model).
            cash:         Available portfolio cash (for cash-check rejection).

        Returns:
            ExecutionReport with filled / partial / rejected status.
        """
        if market_price <= 0:
            return ExecutionReport(order=order, status="rejected",
                                   message="invalid market price")

        # ── Latency ───────────────────────────────────────────────────────
        latency_ms = self._sample_latency()

        # ── Fill probability ──────────────────────────────────────────────
        if random.random() > self.cfg.fill_probability:
            order.status = "rejected"
            return ExecutionReport(order=order, status="rejected",
                                   message="fill rejected (stochastic)")

        # ── Partial fill ──────────────────────────────────────────────────
        filled_qty = order.quantity
        if self.cfg.partial_fill_prob > 0 and random.random() < self.cfg.partial_fill_prob:
            filled_qty = order.quantity * self.cfg.partial_fill_pct

        # ── Limit / stop price check ──────────────────────────────────────
        if order.order_type == "limit":
            if order.side == "buy"  and market_price > (order.limit_price or float("inf")):
                order.status = "pending"
                return ExecutionReport(order=order, status="pending",
                                       message="limit not reached")
            if order.side == "sell" and market_price < (order.limit_price or 0.0):
                order.status = "pending"
                return ExecutionReport(order=order, status="pending",
                                       message="limit not reached")

        # ── Spread cost ───────────────────────────────────────────────────
        half_spread_frac = (self.cfg.spread_bps / 2.0) / 10_000
        direction = 1 if order.side == "buy" else -1
        spread_adj   = market_price * half_spread_frac * direction
        post_spread  = market_price + spread_adj
        spread_cost  = abs(spread_adj) * filled_qty

        # ── Slippage ──────────────────────────────────────────────────────
        fill_price, slippage_cost = self._slippage_model.apply(
            side       = order.side,
            price      = post_spread,
            quantity   = filled_qty,
            volatility = volatility,
            adv        = adv,
        )

        # ── Commission ────────────────────────────────────────────────────
        commission = self._calc_commission(filled_qty, fill_price)

        # ── Cash check ───────────────────────────────────────────────────
        if self.cfg.reject_if_no_cash and order.side == "buy":
            notional = fill_price * filled_qty + commission
            if notional > cash:
                order.status = "rejected"
                return ExecutionReport(order=order, status="rejected",
                                       message=f"insufficient cash: need {notional:.2f}, have {cash:.2f}")

        # ── Build fill ────────────────────────────────────────────────────
        status = "filled" if abs(filled_qty - order.quantity) < 1e-9 else "partial"
        order.status = status

        fill = Fill(
            order_id      = order.order_id,
            symbol        = order.symbol,
            side          = order.side,
            quantity      = filled_qty,
            fill_price    = fill_price,
            gross_price   = market_price,
            commission    = commission,
            slippage_cost = slippage_cost,
            spread_cost   = spread_cost,
            latency_ms    = latency_ms,
            filled_at     = datetime.now(timezone.utc),
        )
        self._fill_history.append(fill)

        return ExecutionReport(order=order, fill=fill, status=status)

    # ── Utility ───────────────────────────────────────────────────────────

    def fill_history(self) -> list[Fill]:
        return list(self._fill_history)

    def total_commissions(self) -> float:
        return sum(f.commission for f in self._fill_history)

    def total_slippage(self) -> float:
        return sum(f.slippage_cost for f in self._fill_history)

    def total_friction(self) -> float:
        return sum(f.total_cost for f in self._fill_history)

    def cost_summary(self) -> dict:
        fills = self._fill_history
        if not fills:
            return {"n_fills": 0, "total_commission": 0.0,
                    "total_slippage": 0.0, "total_spread": 0.0, "total_friction": 0.0}
        return {
            "n_fills":          len(fills),
            "total_commission": round(sum(f.commission    for f in fills), 2),
            "total_slippage":   round(sum(f.slippage_cost for f in fills), 2),
            "total_spread":     round(sum(f.spread_cost   for f in fills), 2),
            "total_friction":   round(sum(f.total_cost    for f in fills), 2),
            "avg_latency_ms":   round(sum(f.latency_ms    for f in fills) / len(fills), 1),
        }

    def reset(self) -> None:
        self._fill_history.clear()

    # ── Presets ───────────────────────────────────────────────────────────

    @classmethod
    def zero_cost(cls) -> "SimulatedBroker":
        """No-friction broker for academic comparison."""
        return cls(BrokerConfig(
            commission_per_share = 0.0,
            commission_min       = 0.0,
            spread_bps           = 0.0,
            slippage_model       = "fixed",
            slippage_bps         = 0.0,
            simulate_latency     = False,
        ))

    @classmethod
    def retail(cls) -> "SimulatedBroker":
        """Retail broker preset (zero-commission, moderate spread/slippage)."""
        return cls(BrokerConfig(
            commission_per_share = 0.0,
            commission_min       = 0.0,
            spread_bps           = 8.0,
            slippage_model       = "volatility",
            slippage_vol_factor  = 0.08,
            latency_mean_ms      = 100.0,
        ))

    @classmethod
    def institutional(cls) -> "SimulatedBroker":
        """Institutional preset: low commission, tight spread, sqrt-impact slippage."""
        return cls(BrokerConfig(
            commission_per_share = 0.002,
            commission_min       = 1.0,
            spread_bps           = 2.0,
            slippage_model       = "sqrt_impact",
            slippage_eta         = 0.07,
            latency_mean_ms      = 20.0,
        ))

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_slippage_model(self) -> SlippageModel:
        cfg = self.cfg
        name = cfg.slippage_model.lower()
        if name == "fixed":
            return make_slippage_model("fixed", bps=cfg.slippage_bps)
        if name == "volatility":
            return make_slippage_model("volatility", vol_factor=cfg.slippage_vol_factor)
        return make_slippage_model("sqrt_impact", eta=cfg.slippage_eta,
                                   fallback_bps=cfg.slippage_bps)

    def _calc_commission(self, qty: float, price: float) -> float:
        per_share = self.cfg.commission_per_share * qty
        pct_based = self.cfg.commission_pct * qty * price
        raw = per_share + pct_based
        return max(self.cfg.commission_min, raw)

    def _sample_latency(self) -> float:
        """Sample latency from a log-normal distribution."""
        if not self.cfg.simulate_latency:
            return 0.0
        mu  = self.cfg.latency_mean_ms
        sig = self.cfg.latency_std_ms
        # Log-normal parameters
        ln_var = math.log(1 + (sig / mu) ** 2) if mu > 0 else 0
        ln_mu  = math.log(mu) - ln_var / 2 if mu > 0 else 0
        raw = math.exp(random.gauss(ln_mu, math.sqrt(ln_var)))
        return max(self.cfg.latency_min_ms, min(self.cfg.latency_max_ms, raw))
