// ─── Existing monitoring types ────────────────────────────────────────────

export interface MetricsSummary {
  run_id:       string
  ticker:       string
  sharpe:       number
  total_return: number
  max_drawdown: number
  win_rate:     number
  n_trades:     number
  regime:       string
  model_used:   string
}

export interface PnLPoint {
  timestamp: string
  equity:    number
  drawdown:  number
}

export interface Trade {
  id:         number
  created_at: string
  symbol:     string
  side:       'buy' | 'sell'
  price:      number
  quantity:   number
  strategy:   string
  regime:     string
  model_used: string
  notes:      string
}

export interface Signal {
  symbol:     string
  signal:     number
  timestamp:  string
  regime:     string
  model_used: string
  prediction: number[]
  confidence: number
}

export interface Position {
  symbol:            string
  side:              'long' | 'short' | 'flat'
  quantity:          number
  avg_entry:         number
  current_price:     number
  unrealized_pnl:    number
  unrealized_pnl_pct:number
}

// ─── Health ───────────────────────────────────────────────────────────────

export interface HealthResponse {
  status:    'ok' | 'degraded'
  postgres:  boolean
  redis:     boolean
  timestamp: string
  version:   string
}

// ─── Predict ──────────────────────────────────────────────────────────────

export interface PredictRequest {
  ticker:     string
  start:      string
  horizon:    number
  lookback:   number
  use_tda:    boolean
  use_macro:  boolean
  model_path: string | null
}

export interface PredictResponse {
  ticker:     string
  regime:     string
  model_used: string
  prediction: number[]
  horizon:    number
  timestamp:  string
}

// ─── Regime ───────────────────────────────────────────────────────────────

export interface RegimeResponse {
  ticker:        string
  regime:        string
  tda_l1:        number | null
  tda_l2:        number | null
  vol_20:        number | null
  sma_ratio_200: number | null
  timestamp:     string
}

// ─── Backtest ─────────────────────────────────────────────────────────────

export interface BacktestRequest {
  ticker:          string
  start:           string
  end:             string | null
  use_tda:         boolean
  use_macro:       boolean
  lookback:        number
  horizon:         number
  position_size:   number
  broker_preset:   'zero_cost' | 'retail' | 'institutional'
  initial_capital: number
  model_path:      string | null
  n_simulations:   number
  retrain_every:   number
  retrain_epochs:  number
  signal_hold:              number
  long_only:                boolean
  signal_threshold:         number
  min_confidence:           number
  stop_loss_pct:            number
  regime_signal_thresholds?: Record<string, number>
  regime_long_only?:         Record<string, boolean>
  regime_default_long?:      string[]
  low_vol_bull_buy_hold:     boolean
}

export interface SimulationStats {
  n_simulations:     number
  sharpe_mean:       number
  sharpe_std:        number
  total_return_mean: number
  total_return_std:  number
  max_drawdown_mean: number
  max_drawdown_std:  number
  win_rate_mean:     number
  win_rate_std:      number
  alpha_mean:        number
  alpha_std:         number
  beta_mean:         number
  beta_std:          number
  equity_mean:       number[]
  equity_p10:        number[]
  equity_p90:        number[]
}

export interface CostSummary {
  n_fills:          number
  total_commission: number
  total_slippage:   number
  total_spread:     number
  total_friction:   number
  avg_latency_ms:   number
}

export interface BacktestResponse {
  ticker:           string
  broker_preset:    string
  sharpe:           number
  sortino:          number
  total_return:     number
  max_drawdown:     number
  win_rate:         number
  n_trades:         number
  calmar:           number
  avg_trade_return: number
  alpha:            number
  beta:             number
  cost_summary:     CostSummary
  equity:           number[]
  regime_curve:     string[]
  simulation_stats: SimulationStats | null
  bah_equity:       number[]
  bah_return:       number
  buy_bars:         number[]
  sell_bars:        number[]
  exit_bars:        number[]
  signal_series:    number[]
  notes:            string
  timestamp:        string
}

// ─── Train ────────────────────────────────────────────────────────────────

export interface TrainRequest {
  ticker:        string
  start:         string
  use_tda:       boolean
  lookback:      number
  horizon:       number
  epochs:        number
  save_artifact: boolean
  artifact_name: string | null
}

export interface TrainResponse {
  job_id:        string
  ticker:        string
  status:        'queued' | 'running' | 'done' | 'error'
  regime:        string | null
  model_used:    string | null
  artifact_path: string | null
  message:       string | null
  started_at:    string
  finished_at:   string | null
}

// ─── Strategy ─────────────────────────────────────────────────────────────

export interface PortfolioSummary {
  equity:          number
  cash:            number
  unrealized_pnl:  number
  realized_pnl:    number
  total_return:    number
  drawdown:        number
  max_drawdown:    number
  sharpe:          number
  rolling_vol_20:  number
  n_positions:     number
  gross_exposure:  number
}

export interface RiskStatus {
  halted:              boolean
  halt_reason:         string
  tracked_stops:       string[]
  vol_target:          number
  max_drawdown_limit:  number
  stop_loss_pct:       number
  trailing_stop:       boolean
  trailing_stop_pct:   number
}

export interface StrategyStatusResponse {
  portfolio:   PortfolioSummary
  risk_status: RiskStatus
  n_orders:    number
  n_signals:   number
}

// ─── Portfolio Backtest ────────────────────────────────────────────────────

export interface PortfolioBacktestRequest {
  tickers:         string[]
  weights:         number[] | null
  start:           string
  end:             string | null
  use_tda:         boolean
  use_macro:       boolean
  lookback:        number
  horizon:         number
  broker_preset:   'zero_cost' | 'retail' | 'institutional'
  initial_capital: number
  vol_weight:      boolean
  regime_shift:    boolean
  rebalance_freq:  'never' | 'monthly' | 'quarterly' | 'annual'
  long_only:       boolean
}

export interface TickerBacktestResult {
  ticker:       string
  weight:       number
  sharpe:       number
  sortino:      number
  total_return: number
  max_drawdown: number
  win_rate:     number
  n_trades:     number
  calmar:       number
  alpha:        number
  beta:         number
  equity:       number[]
  regime_curve: string[]
  error:        string | null
}

export interface PortfolioBacktestResponse {
  tickers:                string[]
  weights:                number[]
  ticker_results:         TickerBacktestResult[]
  portfolio_equity:       number[]
  portfolio_sharpe:       number
  portfolio_return:       number
  portfolio_max_drawdown: number
  correlation_matrix:     number[][]
  initial_capital:        number
  timestamp:              string
}

// ─── Model Benchmark ──────────────────────────────────────────────────────

export type ModelName = 'TFTModel' | 'OnlineModel' | 'TCNModel' | 'TCNLSTMModel'

export interface BenchmarkRequest {
  ticker:           string
  model_name:       ModelName
  start:            string
  end:              string | null
  use_tda:          boolean
  use_macro:        boolean
  lookback:         number
  horizon:          number
  position_size:    number
  broker_preset:    'zero_cost' | 'retail' | 'institutional'
  initial_capital:  number
  signal_hold:      number
  long_only:        boolean
  signal_threshold:          number
  min_confidence:            number
  stop_loss_pct:             number
  regime_signal_thresholds?: Record<string, number>
  regime_long_only?:         Record<string, boolean>
  regime_default_long?:      string[]
  n_simulations:             number
}

export interface RegimeMetrics {
  regime:           string
  is_native:        boolean
  n_bars:           number
  sharpe:           number
  sharpe_std:       number
  total_return:     number
  total_return_std: number
  max_drawdown:     number
  win_rate:         number
}

export interface BenchmarkResponse {
  ticker:               string
  model_name:           string
  native_regime:        string
  overall_sharpe:       number
  overall_sortino:      number
  overall_return:       number
  overall_max_drawdown: number
  overall_win_rate:     number
  overall_calmar:       number
  n_trades:             number
  alpha:                number
  beta:                 number
  cost_summary:         CostSummary
  equity:               number[]
  regime_curve:         string[]
  regime_breakdown:     RegimeMetrics[]
  simulation_stats:     SimulationStats | null
  bah_equity:           number[]
  bah_return:           number
  notes:                string
  timestamp:            string
}
