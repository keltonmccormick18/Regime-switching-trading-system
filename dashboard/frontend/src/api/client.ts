import type {
  BacktestRequest, BacktestResponse,
  BenchmarkRequest, BenchmarkResponse,
  HealthResponse,
  MetricsSummary, PnLPoint, Position,
  PaperStartRequest, PaperStatusResponse, PaperListItem,
  PortfolioBacktestRequest, PortfolioBacktestResponse,
  PredictRequest, PredictResponse,
  RegimeResponse,
  Signal,
  StrategyStatusResponse,
  Trade,
  TrainRequest, TrainResponse,
} from '../types'

const BASE = '/api'

async function req<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    let detail = body
    try { detail = JSON.parse(body).detail ?? body } catch { /* raw */ }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

const get  = <T>(path: string)          => req<T>(path)
const post = <T>(path: string, body: unknown) =>
  req<T>(path, { method: 'POST', body: JSON.stringify(body) })

export const api = {
  // ── Dashboard (legacy app.py routes kept for PnL/summary widgets) ───────
  summary:   ()               => get<MetricsSummary>('/summary').catch(() => null),
  pnl:       ()               => get<PnLPoint[]>('/pnl').catch(() => []),
  trades:    (symbol?: string) => get<{trades: Trade[]; count: number}>(
                                   `/trades${symbol ? `?symbol=${symbol}` : ''}`
                                 ).catch(() => ({ trades: [], count: 0 })),
  positions: ()               => get<{positions: Position[]; total_unrealized_pnl: number; as_of: string}>(
                                   '/positions'
                                 ).catch(() => ({ positions: [], total_unrealized_pnl: 0, as_of: '' })),

  // ── System ───────────────────────────────────────────────────────────────
  health:         ()           => get<HealthResponse>('/health'),
  regime:         (ticker: string) => get<RegimeResponse>(`/regime/${ticker}`),
  strategyStatus: ()           => get<StrategyStatusResponse>('/strategy/status'),
  strategyReset:  ()           => post<{ status: string }>('/strategy/reset', {}),
  signals:        (symbol: string, limit = 20) =>
    get<Signal[]>(`/signals/${symbol}?limit=${limit}`).catch(() => []),

  // ── Predict ──────────────────────────────────────────────────────────────
  predict: (body: PredictRequest) => post<PredictResponse>('/predict', body),

  // ── Backtest ─────────────────────────────────────────────────────────────
  backtest:          (body: BacktestRequest)          => post<BacktestResponse>('/backtest', body),
  portfolioBacktest: (body: PortfolioBacktestRequest) => post<PortfolioBacktestResponse>('/portfolio/backtest', body),
  benchmark:         (body: BenchmarkRequest)         => post<BenchmarkResponse>('/benchmark', body),

  // ── Train ─────────────────────────────────────────────────────────────────
  train:       (body: TrainRequest) => post<TrainResponse>('/train', body),
  trainStatus: (jobId: string)      => get<TrainResponse>(`/train/${jobId}`),
  trainList:   ()                   => get<{ jobs: TrainResponse[]; count: number }>('/train'),
  trainCancel: (jobId: string)      => req<TrainResponse>(`/train/${jobId}`, { method: 'DELETE' }),

  // ── Paper trading ─────────────────────────────────────────────────────────
  paperStart:  (body: PaperStartRequest) => post<PaperStatusResponse>('/paper/start', body),
  paperStatus: (ticker: string)          => get<PaperStatusResponse>(`/paper/${ticker}/status`),
  paperEquity: (ticker: string)          => get<PnLPoint[]>(`/paper/${ticker}/equity`),
  paperStop:   (ticker: string)          => post<{ running: boolean }>(`/paper/${ticker}/stop`, {}),
  paperList:   ()                        => get<{ engines: PaperListItem[]; count: number }>('/paper'),
}
