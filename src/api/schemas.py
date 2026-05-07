"""Pydantic request/response schemas for the quant trading API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────── Predict ───────────────────────────

class PredictRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    start: str = Field("2010-01-01", description="Historical data start date (YYYY-MM-DD)")
    horizon: int = Field(16, ge=1, le=252)
    lookback: int = Field(64, ge=10, le=512)
    use_tda: bool = False
    use_macro: bool = Field(False, description="B1: include cross-asset macro features (VIX, credit spread, USD)")
    model_path: Optional[str] = Field(None, description="Path to saved .pt artifact; if None, trains fresh")

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper().strip()


class PredictResponse(BaseModel):
    ticker: str
    regime: str
    model_used: str
    prediction: list[float]
    horizon: int
    timestamp: datetime


# ─────────────────────────── Train ───────────────────────────

class TrainRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    start: str = Field("2010-01-01")
    use_tda: bool = False
    use_macro: bool = Field(False, description="B1: include cross-asset macro features")
    lookback: int = Field(64, ge=10, le=512)
    horizon: int = Field(16, ge=1, le=252)
    epochs: int = Field(30, ge=1, le=500)
    save_artifact: bool = True
    artifact_name: Optional[str] = None

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper().strip()


class TrainResponse(BaseModel):
    job_id: str
    ticker: str
    status: Literal["queued", "running", "done", "error"]
    regime: Optional[str] = None
    model_used: Optional[str] = None
    artifact_path: Optional[str] = None
    message: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None


# ─────────────────────────── Trade ───────────────────────────

class TradeRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    side: Literal["buy", "sell"]
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    strategy: str = Field("manual")
    regime: Optional[str] = None
    model_used: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, v: str) -> str:
        return v.upper().strip()


class TradeResponse(BaseModel):
    trade_id: int
    symbol: str
    side: str
    price: float
    quantity: float
    strategy: str
    regime: Optional[str]
    model_used: Optional[str]
    created_at: datetime
    status: str = "filled"


# ─────────────────────────── Metrics ───────────────────────────

class MetricRecord(BaseModel):
    id: Optional[int] = None
    run_id: str
    ticker: str
    sharpe: Optional[float] = None
    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    n_trades: Optional[int] = None
    regime: Optional[str] = None
    model_used: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None


class MetricsResponse(BaseModel):
    ticker: Optional[str] = None
    run_id: Optional[str] = None
    records: list[MetricRecord]
    count: int


# ─────────────────────────── Positions ───────────────────────────

class PositionRecord(BaseModel):
    symbol: str
    quantity: float          # signed: positive = long, negative = short
    avg_entry: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: Literal["long", "short", "flat"]


class PositionsResponse(BaseModel):
    positions: list[PositionRecord]
    total_unrealized_pnl: float
    as_of: datetime


# ─────────────────────────── Signals ───────────────────────────

class SignalResponse(BaseModel):
    symbol: str
    signal: float           # -1, 0, 1
    confidence: float
    regime: Optional[str]
    model_used: Optional[str]
    timestamp: datetime


# ─────────────────────────── Regime ───────────────────────────

class RegimeResponse(BaseModel):
    ticker: str
    regime: str
    tda_l1: Optional[float] = None
    tda_l2: Optional[float] = None
    vol_20: Optional[float] = None
    sma_ratio_200: Optional[float] = None
    timestamp: datetime


# ─────────────────────────── Artifacts ───────────────────────────

class ArtifactRecord(BaseModel):
    name: str
    path: str
    class_name: str
    config: dict
    metadata: dict
    created_at: Optional[str] = None


class ArtifactsResponse(BaseModel):
    artifacts: list[ArtifactRecord]
    count: int


# ─────────────────────────── Health ───────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    postgres: bool
    redis: bool
    timestamp: datetime
    version: str = "1.0.0"


# ─────────────────────────── Execution / Backtest ───────────────

class BrokerPreset(str):
    pass


class BacktestRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    start: str = Field("2010-01-01")
    end: Optional[str] = None
    use_tda: bool = False
    lookback: int = Field(64, ge=10, le=512)
    horizon: int = Field(16, ge=1, le=252)
    position_size: float = Field(0.02, gt=0, le=1.0)
    use_macro: bool = Field(False, description="B1: include cross-asset macro features (VIX, credit, USD)")
    broker_preset: Literal["zero_cost", "retail", "institutional"] = "retail"
    initial_capital: float = Field(100_000.0, gt=0)
    model_path: Optional[str] = None    # pre-fitted artifact; trains fresh if None
    n_simulations: int = Field(1, ge=1, le=30,
        description="Number of Monte Carlo runs with different random seeds. "
                    "1 = single run (random seed). >1 = aggregate over N models.")
    retrain_every: int = Field(0, ge=0, le=252,
        description="Walk-forward retraining interval in OOS bars. 0 = disabled. "
                    "E.g. 63 ≈ quarterly, 21 ≈ monthly.")
    retrain_epochs: int = Field(10, ge=1, le=100,
        description="Training epochs per walk-forward retrain step.")
    signal_hold: int = Field(1, ge=1, le=252,
        description="Re-evaluate the signal every N bars. Set to horizon to trade "
                    "at the model's natural frequency (e.g. 16 for a 16-day model).")
    long_only: bool = Field(False,
        description="If True, short signals are converted to flat (hold cash). "
                    "Recommended for broad equity indices.")
    stop_loss_pct: float = Field(0.0, ge=0.0, le=0.50,
        description="Hard stop-loss as a fraction of position entry price. "
                    "0.0 = disabled (rely on signal_hold for exits). "
                    "Scale this with position_size: at 90% deployment a 5% stop "
                    "costs 4.5% of total equity per fire.")
    signal_threshold: float = Field(0.005, ge=0.0, le=0.20,
        description="Minimum predicted return magnitude to generate a signal. "
                    "0.0 = trade on any non-zero prediction (most trades). "
                    "0.01 = require 1%+ predicted return.")
    min_confidence: float = Field(0.30, ge=0.0, le=1.0,
        description="Minimum SNR confidence score to generate a signal. "
                    "0.0 = ignore confidence filter entirely.")
    regime_signal_thresholds: Optional[dict[str, float]] = Field(
        None,
        description="Per-regime min |return| thresholds. Keys: HIGH_VOL_BULL, HIGH_VOL_BEAR, "
                    "LOW_VOL_BULL, LOW_VOL_BEAR. When provided, overrides the engine defaults."
    )
    regime_long_only: Optional[dict[str, bool]] = Field(
        None,
        description="Per-regime long_only flag. True = long/flat only; False = allow shorting. "
                    "Overrides the global long_only per regime."
    )
    regime_default_long: Optional[list[str]] = Field(
        None,
        description="Regimes where the strategy defaults to long (buy-and-hold mode). "
                    "In these regimes the model only steps aside when both heads are "
                    "confidently bearish. Typical use: ['LOW_VOL_BULL']."
    )
    low_vol_bull_buy_hold: bool = Field(
        True,
        description="When True, LOW_VOL_BULL periods use pure buy-and-hold (no model trained "
                    "or consulted). When False, the TCN model is trained and used actively."
    )

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper().strip()


class SimulationStats(BaseModel):
    """Aggregate statistics across N Monte Carlo backtest runs."""
    n_simulations: int
    # Per-metric mean and standard deviation across runs
    sharpe_mean: float
    sharpe_std: float
    total_return_mean: float
    total_return_std: float
    max_drawdown_mean: float
    max_drawdown_std: float
    win_rate_mean: float
    win_rate_std: float
    alpha_mean: float
    alpha_std: float
    beta_mean: float
    beta_std: float
    # Equity curve distribution (all interpolated to the same length)
    equity_mean: list[float]
    equity_p10: list[float]
    equity_p90: list[float]


class CostSummary(BaseModel):
    n_fills: int
    total_commission: float
    total_slippage: float
    total_spread: float
    total_friction: float
    avg_latency_ms: float


class BacktestResponse(BaseModel):
    ticker: str
    broker_preset: str
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    calmar: float
    avg_trade_return: float
    alpha: float = 0.0
    beta: float = 0.0
    cost_summary: CostSummary
    equity: list[float]
    regime_curve: list[str] = []
    simulation_stats: Optional[SimulationStats] = None
    bah_equity: list[float] = []
    bah_return: float = 0.0
    buy_bars:      list[int]   = []
    sell_bars:     list[int]   = []
    exit_bars:     list[int]   = []
    signal_series: list[float] = []
    notes: str
    timestamp: datetime


# ─────────────────────────── Paper Trading ──────────────────────

class PaperStartRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    interval: Literal["1m", "5m", "15m", "1h", "1d"] = "1d"
    broker_preset: Literal["zero_cost", "retail", "institutional"] = "retail"
    initial_capital: float = Field(100_000.0, gt=0)
    prediction_interval: int = Field(1, ge=1, le=100)
    max_drawdown_limit: float = Field(0.15, gt=0, le=1.0)
    stop_loss_pct: float = Field(0.05, gt=0, le=1.0)
    latency_ms: float = Field(100.0, ge=0)
    max_position_pct: float = Field(0.40, gt=0, le=1.0)
    max_leverage: float = Field(1.5, gt=0, le=5.0)
    rebalance_threshold: float = Field(0.05, gt=0, le=0.5)
    model_path: Optional[str] = None

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper().strip()


class PaperStatusResponse(BaseModel):
    ticker: str
    running: bool
    bar_count: int
    last_bar_ts: Optional[str]
    n_orders: int
    portfolio: dict
    risk_status: dict
    cost_summary: dict
    timestamp: datetime


# ─────────────────────────── Portfolio Backtest ──────────────────────────

class PortfolioBacktestRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=20)
    weights: Optional[list[float]] = Field(None, description="Portfolio weights; equal-weight if None")
    start: str = Field("2010-01-01")
    end: Optional[str] = None
    use_tda: bool = False
    use_macro: bool = Field(False, description="B1: include cross-asset macro features")
    lookback: int = Field(64, ge=10, le=512)
    horizon: int = Field(16, ge=1, le=252)
    broker_preset: Literal["zero_cost", "retail", "institutional"] = "retail"
    initial_capital: float = Field(100_000.0, gt=0)
    vol_weight: bool = Field(True, description="Inverse-vol allocation: weight ∝ 1/vol_20")
    regime_shift: bool = Field(True, description="Halve allocation for HIGH_VOL-regime tickers")
    rebalance_freq: Literal["never", "monthly", "quarterly", "annual"] = "never"
    long_only: bool = Field(True, description="Long/flat only — no short selling")

    @field_validator("tickers")
    @classmethod
    def upper_tickers(cls, v: list[str]) -> list[str]:
        return [t.upper().strip() for t in v]

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None:
            if any(w <= 0 for w in v):
                raise ValueError("All weights must be positive")
        return v


class TickerBacktestResult(BaseModel):
    ticker: str
    weight: float
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    calmar: float
    alpha: float = 0.0
    beta: float = 0.0
    equity: list[float]
    regime_curve: list[str] = []
    error: Optional[str] = None


class PortfolioBacktestResponse(BaseModel):
    tickers: list[str]
    weights: list[float]
    ticker_results: list[TickerBacktestResult]
    portfolio_equity: list[float]
    portfolio_sharpe: float
    portfolio_return: float
    portfolio_max_drawdown: float
    correlation_matrix: list[list[float]]
    initial_capital: float
    timestamp: datetime


# ─────────────────────────── Model Benchmark ────────────────────────────────

class BenchmarkRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    model_name: Literal["TFTModel", "OnlineModel", "TCNModel", "TCNLSTMModel"] = Field(
        ..., description="Which model to benchmark. Each model has a designated native regime."
    )
    start: str = Field("2010-01-01")
    end: Optional[str] = None
    use_tda: bool = False
    use_macro: bool = Field(False, description="B1: include cross-asset macro features (VIX, credit, USD)")
    lookback: int = Field(64, ge=10, le=512)
    horizon: int = Field(16, ge=1, le=252)
    position_size: float = Field(0.02, gt=0, le=1.0)
    broker_preset: Literal["zero_cost", "retail", "institutional"] = "retail"
    initial_capital: float = Field(100_000.0, gt=0)
    signal_hold: int = Field(1, ge=1, le=252)
    long_only: bool = False
    signal_threshold: float = Field(0.005, ge=0.0, le=0.20,
        description="Fallback threshold when no regime-specific override is present.")
    min_confidence: float = Field(0.30, ge=0.0, le=1.0)
    stop_loss_pct: float = Field(0.0, ge=0.0, le=0.50)
    regime_signal_thresholds: Optional[dict[str, float]] = Field(
        None,
        description="Per-regime min |return| thresholds. Keys: HIGH_VOL_BULL, HIGH_VOL_BEAR, LOW_VOL_BULL, LOW_VOL_BEAR. Overrides the module defaults when provided."
    )
    n_simulations: int = Field(1, ge=1, le=20,
        description="Number of Monte Carlo seeds. 1 = single run. >1 = mean ± std across N independent weight initialisations.")
    regime_default_long: Optional[list[str]] = Field(
        None,
        description="Regimes where the strategy defaults to long (buy-and-hold mode). "
                    "Set automatically for TCNModel (LOW_VOL_BULL native) when not provided."
    )

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper().strip()


class RegimeMetrics(BaseModel):
    regime: str
    is_native: bool
    n_bars: int
    sharpe: float
    sharpe_std: float = 0.0
    total_return: float
    total_return_std: float = 0.0
    max_drawdown: float
    win_rate: float


class BenchmarkResponse(BaseModel):
    ticker: str
    model_name: str
    native_regime: str
    overall_sharpe: float
    overall_sortino: float
    overall_return: float
    overall_max_drawdown: float
    overall_win_rate: float
    overall_calmar: float
    n_trades: int
    alpha: float
    beta: float
    cost_summary: CostSummary
    equity: list[float]
    regime_curve: list[str]
    regime_breakdown: list[RegimeMetrics]
    simulation_stats: Optional[SimulationStats] = None
    bah_equity: list[float]
    bah_return: float
    notes: str
    timestamp: datetime
