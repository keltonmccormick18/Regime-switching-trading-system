import { useState } from 'react'
import { BarChart2, Plus, Trash2 } from 'lucide-react'
import {
  ComposedChart, AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceArea,
} from 'recharts'
import { api } from '../api/client'
import type {
  BacktestResponse,
  PortfolioBacktestResponse,
  SimulationStats,
  TickerBacktestResult,
} from '../types'
import { Field, Btn, Panel, Badge, ErrorBox, Stat } from '../components/ui'

// ── Regime colour palette ──────────────────────────────────────────────────

const REGIME_FILL: Record<string, string> = {
  HIGH_VOL_BULL: 'rgba(251,191,36,0.13)',   // amber  — volatile uptrend
  HIGH_VOL_BEAR: 'rgba(248,113,113,0.13)',  // red    — volatile downtrend
  LOW_VOL_BULL:  'rgba(74,222,128,0.10)',   // green  — calm uptrend
  LOW_VOL_BEAR:  'rgba(147,197,253,0.10)',  // blue   — calm downtrend
}

const REGIME_LABEL: Record<string, string> = {
  HIGH_VOL_BULL: 'High Vol Bull',
  HIGH_VOL_BEAR: 'High Vol Bear',
  LOW_VOL_BULL:  'Low Vol Bull',
  LOW_VOL_BEAR:  'Low Vol Bear',
}

const REGIME_DOT: Record<string, string> = {
  HIGH_VOL_BULL: '#fbbf24',
  HIGH_VOL_BEAR: '#f87171',
  LOW_VOL_BULL:  '#4ade80',
  LOW_VOL_BEAR:  '#93c5fd',
}

function regimeSegments(curve: string[]) {
  const segs: { x1: number; x2: number; regime: string }[] = []
  if (!curve.length) return segs
  let start = 0
  for (let i = 1; i <= curve.length; i++) {
    if (i === curve.length || curve[i] !== curve[start]) {
      segs.push({ x1: start, x2: i - 1, regime: curve[start] })
      start = i
    }
  }
  return segs
}

function RegimeLegend({ curve }: { curve: string[] }) {
  const present = [...new Set(curve)]
  if (!present.length) return null
  return (
    <div className="flex flex-wrap gap-3 mt-2">
      {present.map(r => (
        <span key={r} className="flex items-center gap-1.5 text-xs text-slate-400">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: REGIME_DOT[r] }} />
          {REGIME_LABEL[r] ?? r}
        </span>
      ))}
    </div>
  )
}

const safeNum = (v: number | null | undefined) => (v != null && isFinite(v) ? v : 0)

const PRESETS = ['zero_cost', 'retail', 'institutional'] as const

// ── Trade marker dots ──────────────────────────────────────────────────────

function BuyDot(props: { cx?: number; cy?: number; value?: number }) {
  const { cx, cy, value } = props
  if (cx == null || cy == null || value == null) return null
  // Upward-pointing triangle below the line
  return <polygon points={`${cx},${cy + 3} ${cx - 8},${cy + 18} ${cx + 8},${cy + 18}`}
    fill="#4ade80" stroke="#166534" strokeWidth={1} opacity={1} />
}

function SellDot(props: { cx?: number; cy?: number; value?: number }) {
  const { cx, cy, value } = props
  if (cx == null || cy == null || value == null) return null
  // Downward-pointing triangle above the line
  return <polygon points={`${cx},${cy - 3} ${cx - 8},${cy - 18} ${cx + 8},${cy - 18}`}
    fill="#f87171" stroke="#991b1b" strokeWidth={1} opacity={1} />
}

function ExitDot(props: { cx?: number; cy?: number; value?: number }) {
  const { cx, cy, value } = props
  if (cx == null || cy == null || value == null) return null
  // Diamond (rotated square) — marks long → cash transition
  return <polygon points={`${cx},${cy - 10} ${cx + 8},${cy} ${cx},${cy + 10} ${cx - 8},${cy}`}
    fill="#94a3b8" stroke="#475569" strokeWidth={1} opacity={0.95} />
}

// Compute [start, end] ranges where signal === 0 from a signal series
function cashRanges(signals: number[]): { x1: number; x2: number }[] {
  const ranges: { x1: number; x2: number }[] = []
  let start: number | null = null
  for (let i = 0; i < signals.length; i++) {
    if (signals[i] === 0 && start === null) start = i
    if (signals[i] !== 0 && start !== null) { ranges.push({ x1: start, x2: i - 1 }); start = null }
  }
  if (start !== null) ranges.push({ x1: start, x2: signals.length - 1 })
  return ranges
}

// ── Shared form defaults ───────────────────────────────────────────────────

const SHARED_DEFAULTS = {
  start: '2010-01-01', end: '',
  use_tda: false, use_macro: false, lookback: 64, horizon: 16,
  position_size: 0.02, broker_preset: 'retail' as const,
  initial_capital: 100_000,
}

// ── Single-ticker backtest ─────────────────────────────────────────────────

function SingleBacktest() {
  const [form, setForm]       = useState({
    ...SHARED_DEFAULTS,
    ticker: 'AAPL', model_path: '', n_simulations: 1,
    retrain_every: 0, retrain_epochs: 10,
    signal_hold: 16, long_only: true,
    min_confidence: 0.0, stop_loss_pct: 0.0,
    low_vol_bull_buy_hold: true,
    regime_thresholds: {
      HIGH_VOL_BULL: 0.008,
      HIGH_VOL_BEAR: 0.005,
      LOW_VOL_BULL:  0.002,
      LOW_VOL_BEAR:  0.003,
    },
    regime_long_only: {
      HIGH_VOL_BULL: true,
      HIGH_VOL_BEAR: false,
      LOW_VOL_BULL:  true,
      LOW_VOL_BEAR:  false,
    },
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState<BacktestResponse | null>(null)
  const [error, setError]     = useState<string | null>(null)

  const set = (k: string, v: unknown) => setForm(f => ({ ...f, [k]: v }))

  async function run() {
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await api.backtest({
        ticker:          form.ticker.toUpperCase(),
        start:           form.start,
        end:             form.end || null,
        use_tda:         form.use_tda,
        use_macro:       form.use_macro,
        lookback:        form.lookback,
        horizon:         form.horizon,
        position_size:   form.position_size,
        broker_preset:   form.broker_preset,
        initial_capital: form.initial_capital,
        model_path:      form.model_path || null,
        n_simulations:   form.n_simulations,
        retrain_every:   form.retrain_every,
        retrain_epochs:  form.retrain_epochs,
        signal_hold:              form.signal_hold,
        long_only:                form.long_only,
        signal_threshold:         0.005,
        min_confidence:           form.min_confidence,
        stop_loss_pct:            form.stop_loss_pct,
        regime_signal_thresholds: form.regime_thresholds,
        regime_long_only:         form.regime_long_only,
        low_vol_bull_buy_hold:    form.low_vol_bull_buy_hold,
      })
      setResult(r)
    } catch (e: unknown) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const ss = result?.simulation_stats ?? null

  // Equity chart data — band when MC, simple curve when single run
  const buySet  = new Set(result?.buy_bars  ?? [])
  const sellSet = new Set(result?.sell_bars ?? [])
  const exitSet = new Set(result?.exit_bars ?? [])

  const equityData = result
    ? result.equity.map((v, i) => ({
        bar:      i,
        equity:   +safeNum(v).toFixed(2),
        bah:      +safeNum((result.bah_equity ?? [])[i]).toFixed(2),
        buyMark:  buySet.has(i)  ? +safeNum(v).toFixed(2) : undefined,
        sellMark: sellSet.has(i) ? +safeNum(v).toFixed(2) : undefined,
        exitMark: exitSet.has(i) ? +safeNum(v).toFixed(2) : undefined,
        ...(ss ? {
          p10: +safeNum(ss.equity_p10[i]).toFixed(2),
          p90: +safeNum(ss.equity_p90[i]).toFixed(2),
        } : {}),
      }))
    : []

  const cashZones = result?.signal_series ? cashRanges(result.signal_series) : []

  const ddData = result
    ? result.equity.map((v, i, arr) => {
        const peak = Math.max(...arr.slice(0, i + 1))
        const dd   = peak > 0 ? ((v - peak) / peak * 100) : 0
        return { bar: i, drawdown: +dd.toFixed(3) }
      })
    : []

  // Y-axis domain: must include strategy, B&H, and MC p90 band
  const equityYDomain: [number, number] | undefined = result
    ? (() => {
        const vals = [
          ...equityData.map(d => d.equity),
          ...equityData.map(d => d.bah).filter(v => v > 0),
          ...(ss ? equityData.map(d => d.p90 ?? 0) : []),
        ]
        const lo = Math.min(...vals)
        const hi = Math.max(...vals)
        const pad = (hi - lo) * 0.04
        return [Math.floor((lo - pad) / 1000) * 1000, Math.ceil((hi + pad) / 1000) * 1000]
      })()
    : undefined

  const pos = result && result.total_return >= 0

  // Helper: show "value ± std" when MC, plain value otherwise
  function mcStat(val: number, std: number | undefined, fmt: (v: number) => string) {
    if (!ss || std == null) return fmt(val)
    return <>{fmt(val)} <span className="text-xs text-slate-500 font-normal">±{fmt(std)}</span></>
  }

  return (
    <div className="space-y-5">
      <Panel title="Single Ticker Configuration" icon={<BarChart2 size={14} />}>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
          <Field label="Ticker">
            <input className="inp" value={form.ticker}
              onChange={e => set('ticker', e.target.value.toUpperCase())} />
          </Field>
          <Field label="Start Date">
            <input className="inp" type="date" value={form.start}
              onChange={e => set('start', e.target.value)} />
          </Field>
          <Field label="End Date (optional)">
            <input className="inp" type="date" value={form.end}
              onChange={e => set('end', e.target.value)} />
          </Field>
          <Field label="Broker">
            <select className="inp" value={form.broker_preset}
              onChange={e => set('broker_preset', e.target.value)}>
              {PRESETS.map(p => <option key={p} value={p}>{p.replace('_', ' ')}</option>)}
            </select>
          </Field>
          <Field label="Position Size (fraction of equity)">
            <input className="inp" type="number" step={0.05} min={0.01} max={0.5}
              value={form.position_size} onChange={e => set('position_size', +e.target.value)} />
            <p className="text-xs text-slate-500 mt-0.5">
              {form.long_only
                ? `${(form.position_size * 100).toFixed(0)}% deployed when bullish, cash otherwise`
                : `${(form.position_size * 100).toFixed(0)}% of equity per trade`}
            </p>
          </Field>
          <Field label="Initial Capital ($)">
            <input className="inp" type="number" step={1000} min={1000}
              value={form.initial_capital} onChange={e => set('initial_capital', +e.target.value)} />
          </Field>
          <Field label="Lookback">
            <input className="inp" type="number" min={10} max={512} value={form.lookback}
              onChange={e => set('lookback', +e.target.value)} />
          </Field>
          <Field label="Horizon">
            <input className="inp" type="number" min={1} max={252} value={form.horizon}
              onChange={e => set('horizon', +e.target.value)} />
          </Field>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
          <Field label="Artifact Path (optional)">
            <input className="inp" placeholder="Leave blank to train fresh"
              value={form.model_path} onChange={e => set('model_path', e.target.value)} />
          </Field>
          <Field label={`Monte Carlo Runs: ${form.n_simulations}`}>
            <div className="flex items-center gap-3 mt-2">
              <input type="range" min={1} max={20} step={1}
                className="w-full accent-blue-500"
                value={form.n_simulations}
                onChange={e => set('n_simulations', +e.target.value)} />
              <span className="text-sm font-mono text-slate-300 w-6 text-right">
                {form.n_simulations}
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              {form.n_simulations === 1 ? 'Single run — fast' : `${form.n_simulations} models trained in parallel — shows confidence band`}
            </p>
          </Field>
          <Field label="Use TDA features">
            <label className="flex items-center gap-2 mt-2 cursor-pointer">
              <input type="checkbox" className="w-4 h-4 accent-blue-500"
                checked={form.use_tda} onChange={e => set('use_tda', e.target.checked)} />
              <span className="text-sm text-slate-300">Enable TDA (slow)</span>
            </label>
          </Field>
          <Field label="Cross-asset macro">
            <label className="flex items-center gap-2 mt-2 cursor-pointer">
              <input type="checkbox" className="w-4 h-4 accent-emerald-500"
                checked={form.use_macro} onChange={e => set('use_macro', e.target.checked)} />
              <span className="text-sm text-slate-300">VIX + credit + USD features</span>
            </label>
          </Field>
        </div>

        {/* Signal structure */}
        <div className="border border-slate-700 rounded-lg p-3 mb-4">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
            Signal Structure
          </p>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-3">
            <Field label="Signal Hold (bars)">
              <input className="inp" type="number" min={1} max={252} step={1}
                value={form.signal_hold}
                onChange={e => set('signal_hold', +e.target.value)} />
            </Field>
            <Field label="Min Confidence">
              <input className="inp" type="number" min={0} max={1} step={0.05}
                value={form.min_confidence}
                onChange={e => set('min_confidence', +e.target.value)} />
            </Field>
            <Field label="Stop-Loss (0 = off)">
              <input className="inp" type="number" min={0} max={0.5} step={0.01}
                value={form.stop_loss_pct}
                onChange={e => set('stop_loss_pct', +e.target.value)} />
            </Field>
          </div>

          {/* Per-regime settings */}
          <div className="border-t border-slate-700/60 pt-3">
            {/* Low Vol Bull mode toggle */}
            <div className="mb-3 p-2.5 rounded-lg border border-slate-700/80 bg-slate-800/40">
              <label className="flex items-start gap-3 cursor-pointer">
                <input type="checkbox" className="w-4 h-4 mt-0.5 accent-emerald-500 shrink-0"
                  checked={form.low_vol_bull_buy_hold}
                  onChange={e => set('low_vol_bull_buy_hold', e.target.checked)} />
                <div>
                  <span className="flex items-center gap-1.5 text-sm text-slate-200 font-medium">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
                    Buy &amp; hold during Low Vol Bull
                  </span>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {form.low_vol_bull_buy_hold
                      ? 'Holds long unconditionally in calm uptrends — no model trained or consulted. Saves one training slot and avoids friction.'
                      : 'TCN model is trained and actively signals in Low Vol Bull periods. Threshold and direction controls below apply.'}
                  </p>
                </div>
              </label>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-3">
              {(Object.keys(form.regime_thresholds) as Array<keyof typeof form.regime_thresholds>).map(r => {
                const isBuyHold = r === 'LOW_VOL_BULL' && form.low_vol_bull_buy_hold
                return (
                  <div key={r} className={`space-y-2 ${isBuyHold ? 'opacity-40 pointer-events-none' : ''}`}>
                    <div className="flex items-center gap-1.5 text-xs text-slate-400 font-medium">
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: REGIME_DOT[r] }} />
                      {REGIME_LABEL[r]}
                      {isBuyHold && <span className="text-emerald-500 font-semibold ml-1">B&amp;H</span>}
                    </div>
                    <Field label="Threshold">
                      <input className="inp" type="number" min={0} max={0.2} step={0.001}
                        value={form.regime_thresholds[r]}
                        onChange={e => setForm(f => ({
                          ...f,
                          regime_thresholds: { ...f.regime_thresholds, [r]: +e.target.value },
                        }))} />
                    </Field>
                    <Field label="Direction">
                      <select className="inp"
                        value={form.regime_long_only[r] ? 'long_only' : 'long_short'}
                        onChange={e => setForm(f => ({
                          ...f,
                          regime_long_only: { ...f.regime_long_only, [r]: e.target.value === 'long_only' },
                        }))}>
                        <option value="long_only">Long / Flat</option>
                        <option value="long_short">Long / Short</option>
                      </select>
                    </Field>
                  </div>
                )
              })}
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Bear regimes default to Long / Short so the model can profit from correctly predicting downside.
            </p>
          </div>
        </div>

        {/* Walk-forward retraining */}
        <div className="border border-slate-700 rounded-lg p-3 mb-4">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
            Walk-Forward Retraining
          </p>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 items-end">
            <Field label="Retrain Every (bars)">
              <input className="inp" type="number" min={0} max={252} step={1}
                value={form.retrain_every}
                onChange={e => set('retrain_every', +e.target.value)} />
            </Field>
            <Field label="Retrain Epochs">
              <input className="inp" type="number" min={1} max={100} step={1}
                value={form.retrain_epochs}
                onChange={e => set('retrain_epochs', +e.target.value)} />
            </Field>
            <div className="col-span-2 text-xs text-slate-500 leading-relaxed self-end pb-2">
              {form.retrain_every === 0
                ? 'Disabled — single static model trained on train split only.'
                : `Models retrain every ${form.retrain_every} OOS bars (~${form.retrain_every === 21 ? 'monthly' : form.retrain_every === 63 ? 'quarterly' : `${form.retrain_every} days`}) on the expanding window. `
                  + `${form.retrain_epochs} epochs per retrain. `
                  + 'Significantly slower but adapts to regime shifts.'
              }
            </div>
          </div>
        </div>
        <Btn loading={loading} onClick={run}>
          {loading
            ? form.n_simulations > 1
              ? `Training ${form.n_simulations * (form.low_vol_bull_buy_hold ? 3 : 4)} models… (may take several minutes)`
              : form.retrain_every > 0
                ? `Training + walk-forward retraining every ${form.retrain_every} bars…`
                : form.low_vol_bull_buy_hold
                  ? 'Training 3 regime models + B&H for Low Vol Bull…'
                  : 'Training 4 regime models… (may take 1–3 min)'
            : form.n_simulations > 1
              ? `Run ${form.n_simulations}-Simulation Monte Carlo`
              : form.retrain_every > 0
                ? `Run Walk-Forward Backtest (retrain every ${form.retrain_every} bars)`
                : 'Run Backtest'}
        </Btn>
        {error && <ErrorBox msg={error} />}
      </Panel>

      {result && (
        <>
          {ss && (
            <div className="flex items-center gap-2 px-3 py-2 bg-blue-950/40 border border-blue-800/40 rounded-lg text-xs text-blue-300">
              <span className="font-semibold">Monte Carlo — {ss.n_simulations} runs</span>
              <span className="text-blue-400/60">·</span>
              <span>Metrics show mean ± 1σ across all trained models</span>
            </div>
          )}
          <div className="grid grid-cols-3 lg:grid-cols-8 gap-3">
            <Stat label="Strategy Return"
              value={<>
                {mcStat(result.total_return, ss?.total_return_std, v => `${v >= 0 ? '+' : ''}${(v*100).toFixed(1)}%`)}
                {result.bah_return != null && (
                  <span className="block text-xs font-normal mt-0.5"
                    style={{ color: result.bah_return >= 0 ? '#4ade80' : '#f87171' }}>
                    B&amp;H {result.bah_return >= 0 ? '+' : ''}{(result.bah_return * 100).toFixed(1)}%
                  </span>
                )}
              </>}
              color={pos ? 'text-green-400' : 'text-red-400'} />
            <Stat label="Sharpe"
              value={mcStat(result.sharpe, ss?.sharpe_std, v => v.toFixed(2))}
              color={result.sharpe >= 1 ? 'text-green-400' : result.sharpe >= 0 ? 'text-yellow-400' : 'text-red-400'} />
            <Stat label="Sortino"  value={result.sortino.toFixed(2)} color="text-slate-100" />
            <Stat label="Max DD"
              value={mcStat(result.max_drawdown, ss?.max_drawdown_std, v => `${(v*100).toFixed(1)}%`)}
              color="text-red-400" />
            <Stat label="Win Rate"
              value={mcStat(result.win_rate, ss?.win_rate_std, v => `${(v*100).toFixed(1)}%`)}
              color={result.win_rate >= 0.5 ? 'text-green-400' : 'text-yellow-400'} />
            <Stat label="Trades"   value={String(result.n_trades)} color="text-slate-100" />
            <Stat label="Alpha"
              value={mcStat(result.alpha, ss?.alpha_std, v => `${v >= 0 ? '+' : ''}${(v*100).toFixed(1)}%`)}
              color={(result.alpha ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'} />
            <Stat label="Beta"
              value={mcStat(result.beta, ss?.beta_std, v => v.toFixed(2))}
              color={Math.abs(result.beta ?? 0) < 0.5 ? 'text-green-400' : Math.abs(result.beta ?? 0) < 1 ? 'text-yellow-400' : 'text-slate-100'} />
          </div>

          <Panel title={ss ? `Equity Curve — mean ± p10/p90 band (${ss.n_simulations} runs)` : 'Equity Curve'}>
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={equityData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor={pos ? '#4ade80' : '#f87171'} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={pos ? '#4ade80' : '#f87171'} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="bar" tick={{ fill: '#94a3b8', fontSize: 10 }}
                  tickFormatter={(v: number) => v % 50 === 0 ? `Day ${v}` : ''} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }}
                  tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} width={55}
                  domain={equityYDomain} allowDataOverflow={false} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                  formatter={(v: number, name: string) => {
                    const labels: Record<string, string> = {
                      equity: ss ? 'Strategy (mean)' : 'Strategy',
                      p90: 'p90', p10: 'p10',
                      bah: 'Buy & Hold',
                    }
                    return [`$${Number(v).toLocaleString()}`, labels[name] ?? name]
                  }}
                  labelFormatter={(l: number) => {
                    const regime = result.regime_curve?.[l]
                    return `Day ${l}${regime ? `  ·  ${REGIME_LABEL[regime] ?? regime}` : ''}`
                  }}
                />
                {regimeSegments(result.regime_curve ?? []).map((seg, i) => (
                  <ReferenceArea key={i} x1={seg.x1} x2={seg.x2}
                    fill={REGIME_FILL[seg.regime]} strokeOpacity={0} />
                ))}
                {/* Cash / flat zones — gray diagonal hatch overlay */}
                {cashZones.map((z, i) => (
                  <ReferenceArea key={`cash-${i}`} x1={z.x1} x2={z.x2}
                    fill="rgba(148,163,184,0.18)" stroke="rgba(148,163,184,0.35)"
                    strokeWidth={1} strokeDasharray="3 3" />
                ))}
                {/* MC band */}
                {ss && <>
                  <Area type="monotone" dataKey="p90"
                    fill={pos ? '#4ade8022' : '#f8717122'} stroke="none" fillOpacity={1} dot={false} />
                  <Area type="monotone" dataKey="p10"
                    fill="#0f172a" stroke="none" fillOpacity={1} dot={false} />
                </>}
                {/* Strategy equity */}
                <Area type="monotone" dataKey="equity"
                  stroke={pos ? '#4ade80' : '#f87171'} strokeWidth={ss ? 2 : 2}
                  fill={ss ? 'none' : 'url(#eq)'} dot={false} />
                {/* Buy-and-hold — thick amber, renders properly as Line in ComposedChart */}
                <Line type="monotone" dataKey="bah"
                  stroke="#f59e0b" strokeWidth={3} strokeDasharray="10 5"
                  dot={false} legendType="none" strokeOpacity={1} connectNulls />
                {/* Trade markers — buy (▲), short (▼), exit to cash (◆) */}
                <Line dataKey="buyMark" stroke="none" dot={<BuyDot />}
                  activeDot={false} legendType="none" isAnimationActive={false} connectNulls={false} />
                <Line dataKey="sellMark" stroke="none" dot={<SellDot />}
                  activeDot={false} legendType="none" isAnimationActive={false} connectNulls={false} />
                <Line dataKey="exitMark" stroke="none" dot={<ExitDot />}
                  activeDot={false} legendType="none" isAnimationActive={false} connectNulls={false} />
              </ComposedChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap items-center gap-4 mt-2">
              <span className="flex items-center gap-1.5 text-xs text-amber-400 font-medium">
                <svg width="22" height="8" viewBox="0 0 22 8">
                  <line x1="0" y1="4" x2="22" y2="4" stroke="#f59e0b" strokeWidth="2.5" strokeDasharray="8 4" />
                </svg>
                Buy &amp; Hold
              </span>
              <span className="flex items-center gap-1.5 text-xs text-green-400 font-medium">
                <svg width="14" height="14" viewBox="0 0 14 14">
                  <polygon points="7,1 0,13 14,13" fill="#4ade80" stroke="#166534" strokeWidth="1" />
                </svg>
                Buy
              </span>
              <span className="flex items-center gap-1.5 text-xs text-red-400 font-medium">
                <svg width="14" height="14" viewBox="0 0 14 14">
                  <polygon points="7,13 0,1 14,1" fill="#f87171" stroke="#991b1b" strokeWidth="1" />
                </svg>
                Sell
              </span>
              <span className="flex items-center gap-1.5 text-xs text-slate-400 font-medium">
                <svg width="14" height="14" viewBox="0 0 14 14">
                  <polygon points="7,1 13,7 7,13 1,7" fill="#94a3b8" stroke="#475569" strokeWidth="1" />
                </svg>
                Exit to cash
              </span>
              <span className="flex items-center gap-1.5 text-xs text-slate-500">
                <span className="inline-block w-10 h-3 rounded-sm"
                  style={{ background: 'repeating-linear-gradient(135deg,rgba(148,163,184,0.35) 0px,rgba(148,163,184,0.35) 1px,rgba(148,163,184,0.1) 1px,rgba(148,163,184,0.1) 6px)' }} />
                In cash
              </span>
              <RegimeLegend curve={result.regime_curve ?? []} />
            </div>
          </Panel>

          <Panel title="Drawdown">
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={ddData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="dd" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#f87171" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#f87171" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="bar" tick={{ fill: '#94a3b8', fontSize: 10 }}
                  tickFormatter={(v: number) => v % 50 === 0 ? `Day ${v}` : ''} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }}
                  tickFormatter={(v: number) => `${v.toFixed(0)}%`} width={45} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                  formatter={(v: number) => [`${v.toFixed(2)}%`, 'Drawdown']}
                />
                <Area type="monotone" dataKey="drawdown"
                  stroke="#f87171" strokeWidth={1.5} fill="url(#dd)" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </Panel>

          <Panel title="Execution Costs">
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
              {[
                { label: 'Fills',          value: String(result.cost_summary.n_fills) },
                { label: 'Commission',     value: `$${result.cost_summary.total_commission.toFixed(2)}` },
                { label: 'Slippage',       value: `$${result.cost_summary.total_slippage.toFixed(2)}` },
                { label: 'Spread',         value: `$${result.cost_summary.total_spread.toFixed(2)}` },
                { label: 'Total Friction', value: `$${result.cost_summary.total_friction.toFixed(2)}` },
              ].map(({ label, value }) => (
                <div key={label} className="bg-slate-700 rounded-lg p-3 flex flex-col gap-1">
                  <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
                  <span className="text-lg font-mono font-bold text-slate-100">{value}</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-2">{result.notes}</p>
          </Panel>
        </>
      )}
    </div>
  )
}

// ── Colour palette for multi-line charts ───────────────────────────────────

const LINE_COLORS = [
  '#60a5fa', '#34d399', '#f472b6', '#fbbf24', '#a78bfa',
  '#fb923c', '#22d3ee', '#f87171', '#a3e635', '#e879f9',
]

// ── Correlation cell background ────────────────────────────────────────────

function corrColor(v: number): string {
  // v in [-1, 1]: red → gray → green
  if (v >= 0) {
    const g = Math.round(100 + v * 155)
    return `rgb(30,${g},60)`
  }
  const r = Math.round(100 + (-v) * 155)
  return `rgb(${r},30,50)`
}

// ── Portfolio backtest ─────────────────────────────────────────────────────

type TickerRow = { ticker: string; weight: string }

function PortfolioBacktest() {
  const [rows, setRows] = useState<TickerRow[]>([
    { ticker: 'AAPL', weight: '33' },
    { ticker: 'MSFT', weight: '33' },
    { ticker: 'NVDA', weight: '34' },
  ])
  const [form, setForm]       = useState({ ...SHARED_DEFAULTS })
  const [equalWeight, setEqualWeight] = useState(true)
  const [volWeight, setVolWeight]     = useState(true)
  const [regimeShift, setRegimeShift] = useState(true)
  const [rebalanceFreq, setRebalanceFreq] = useState<'never'|'monthly'|'quarterly'|'annual'>('never')
  const [longOnly, setLongOnly]       = useState(true)
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState<PortfolioBacktestResponse | null>(null)
  const [error, setError]     = useState<string | null>(null)

  const setShared = (k: string, v: unknown) => setForm(f => ({ ...f, [k]: v }))

  function distributeWeights(r: TickerRow[]): TickerRow[] {
    const n = r.length
    if (n === 0) return r
    const base = Math.floor(100 / n)
    const rem  = 100 - base * n
    return r.map((row, i) => ({ ...row, weight: String(i === n - 1 ? base + rem : base) }))
  }

  function addRow() {
    setRows(r => {
      const next = [...r, { ticker: '', weight: '0' }]
      return equalWeight ? next : distributeWeights(next)
    })
  }

  function removeRow(i: number) {
    setRows(r => {
      const next = r.filter((_, idx) => idx !== i)
      return (!equalWeight && next.length > 0) ? distributeWeights(next) : next
    })
  }

  function setRow(i: number, k: keyof TickerRow, v: string) {
    setRows(r => r.map((row, idx) => idx === i ? { ...row, [k]: v } : row))
  }

  function handleEqualWeightToggle(checked: boolean) {
    setEqualWeight(checked)
    if (!checked) setRows(r => distributeWeights(r))
  }

  async function run() {
    const tickers = rows.map(r => r.ticker.toUpperCase()).filter(Boolean)
    if (tickers.length === 0) { setError('Add at least one ticker'); return }

    const weights = equalWeight
      ? null
      : rows.map(r => parseFloat(r.weight) || 0).filter((_, i) => rows[i].ticker)

    setLoading(true); setError(null); setResult(null)
    try {
      const r = await api.portfolioBacktest({
        tickers,
        weights,
        start:           form.start,
        end:             form.end || null,
        use_tda:         form.use_tda,
        use_macro:       form.use_macro,
        lookback:        form.lookback,
        horizon:         form.horizon,
        broker_preset:   form.broker_preset,
        initial_capital: form.initial_capital,
        vol_weight:      volWeight,
        regime_shift:    regimeShift,
        rebalance_freq:  rebalanceFreq,
        long_only:       longOnly,
      })
      setResult(r)
    } catch (e: unknown) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  // Build combined chart data (portfolio + per-ticker curves)
  const chartData = result
    ? result.portfolio_equity.map((v, i) => {
        const pt: Record<string, number> = { bar: i, Portfolio: +safeNum(v).toFixed(2) }
        for (const tr of result.ticker_results) {
          pt[tr.ticker] = +safeNum(tr.equity[i]).toFixed(2)
        }
        return pt
      })
    : []

  const portPos = result && result.portfolio_return >= 0

  return (
    <div className="space-y-5">
      {/* ── Ticker list ── */}
      <Panel title="Portfolio Composition" icon={<BarChart2 size={14} />}>
        <div className="flex items-center gap-3 mb-3">
          <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300">
            <input type="checkbox" className="w-4 h-4 accent-blue-500"
              checked={equalWeight} onChange={e => handleEqualWeightToggle(e.target.checked)} />
            Equal weight
          </label>
        </div>

        <div className="space-y-2 mb-4">
          {rows.map((row, i) => (
            <div key={i} className="flex gap-2 items-center">
              <input
                className="inp flex-1"
                placeholder="Ticker"
                value={row.ticker}
                onChange={e => setRow(i, 'ticker', e.target.value.toUpperCase())}
              />
              {!equalWeight && (
                <input
                  className="inp w-24 text-right"
                  type="number" min={0} step={1}
                  placeholder="Weight %"
                  value={row.weight}
                  onChange={e => setRow(i, 'weight', e.target.value)}
                />
              )}
              <button
                className="p-2 text-slate-500 hover:text-red-400 transition-colors"
                onClick={() => removeRow(i)}
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        <button
          className="flex items-center gap-1 text-sm text-blue-400 hover:text-blue-300 mb-5"
          onClick={addRow}
        >
          <Plus size={14} /> Add ticker
        </button>

        {/* Allocation controls */}
        <div className="flex flex-wrap gap-4 mb-5 p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
          <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300">
            <input type="checkbox" className="w-4 h-4 accent-blue-500"
              checked={volWeight} onChange={e => setVolWeight(e.target.checked)} />
            <span>Inverse-vol weighting</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300">
            <input type="checkbox" className="w-4 h-4 accent-blue-500"
              checked={regimeShift} onChange={e => setRegimeShift(e.target.checked)} />
            <span>Regime shift (½ allocation in HIGH_VOL)</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300">
            <input type="checkbox" className="w-4 h-4 accent-blue-500"
              checked={longOnly} onChange={e => setLongOnly(e.target.checked)} />
            <span>Long only</span>
          </label>
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-400">Rebalance</span>
            <select
              className="inp py-1 text-sm w-32"
              value={rebalanceFreq}
              onChange={e => setRebalanceFreq(e.target.value as typeof rebalanceFreq)}
            >
              <option value="never">Never</option>
              <option value="monthly">Monthly</option>
              <option value="quarterly">Quarterly</option>
              <option value="annual">Annual</option>
            </select>
          </div>
        </div>

        {/* Shared params */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
          <Field label="Start Date">
            <input className="inp" type="date" value={form.start}
              onChange={e => setShared('start', e.target.value)} />
          </Field>
          <Field label="End Date (optional)">
            <input className="inp" type="date" value={form.end}
              onChange={e => setShared('end', e.target.value)} />
          </Field>
          <Field label="Broker">
            <select className="inp" value={form.broker_preset}
              onChange={e => setShared('broker_preset', e.target.value)}>
              {PRESETS.map(p => <option key={p} value={p}>{p.replace('_', ' ')}</option>)}
            </select>
          </Field>
          <Field label="Initial Capital ($)">
            <input className="inp" type="number" step={1000} min={1000}
              value={form.initial_capital} onChange={e => setShared('initial_capital', +e.target.value)} />
          </Field>
          <Field label="Lookback">
            <input className="inp" type="number" min={10} max={512} value={form.lookback}
              onChange={e => setShared('lookback', +e.target.value)} />
          </Field>
          <Field label="Horizon">
            <input className="inp" type="number" min={1} max={252} value={form.horizon}
              onChange={e => setShared('horizon', +e.target.value)} />
          </Field>
          <Field label="Use TDA">
            <label className="flex items-center gap-2 mt-2 cursor-pointer">
              <input type="checkbox" className="w-4 h-4 accent-blue-500"
                checked={form.use_tda} onChange={e => setShared('use_tda', e.target.checked)} />
              <span className="text-sm text-slate-300">Enable TDA (slow)</span>
            </label>
          </Field>
          <Field label="Cross-asset macro">
            <label className="flex items-center gap-2 mt-2 cursor-pointer">
              <input type="checkbox" className="w-4 h-4 accent-emerald-500"
                checked={form.use_macro} onChange={e => setShared('use_macro', e.target.checked)} />
              <span className="text-sm text-slate-300">VIX + credit + USD</span>
            </label>
          </Field>
        </div>

        <Btn loading={loading} onClick={run}>
          {loading ? `Running ${rows.length} backtests… (may take 1–3 min)` : 'Run Portfolio Backtest'}
        </Btn>
        {error && <ErrorBox msg={error} />}
      </Panel>

      {result && (
        <>
          {/* ── Portfolio-level metrics ── */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <Stat label="Portfolio Return"
              value={`${portPos ? '+' : ''}${(result.portfolio_return * 100).toFixed(1)}%`}
              color={portPos ? 'text-green-400' : 'text-red-400'} />
            <Stat label="Portfolio Sharpe" value={result.portfolio_sharpe.toFixed(2)}
              color={result.portfolio_sharpe >= 1 ? 'text-green-400' : result.portfolio_sharpe >= 0 ? 'text-yellow-400' : 'text-red-400'} />
            <Stat label="Max Drawdown"
              value={`${(result.portfolio_max_drawdown * 100).toFixed(1)}%`}
              color="text-red-400" />
            <Stat label="Tickers" value={String(result.tickers.length)} color="text-slate-100" />
          </div>

          {/* ── Combined equity curves ── */}
          <Panel title="Equity Curves (portfolio + tickers)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="bar" tick={{ fill: '#94a3b8', fontSize: 10 }}
                  tickFormatter={(v: number) => v % 50 === 0 ? `Day ${v}` : ''} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }}
                  tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} width={55} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                  formatter={(v: number, name: string) => [`$${v.toLocaleString()}`, name]}
                  labelFormatter={(l: number) => `Day ${l}`}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                {/* Portfolio curve first, bold */}
                <Line type="monotone" dataKey="Portfolio"
                  stroke="#ffffff" strokeWidth={2} dot={false} />
                {result.ticker_results.map((tr, i) => (
                  <Line key={tr.ticker} type="monotone" dataKey={tr.ticker}
                    stroke={LINE_COLORS[i % LINE_COLORS.length]}
                    strokeWidth={1} dot={false} strokeDasharray="4 2" />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Panel>

          {/* ── Per-ticker metrics table ── */}
          <Panel title="Per-Ticker Results">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-500 text-xs uppercase border-b border-slate-700">
                    <th className="pb-2 pr-4">Ticker</th>
                    <th className="pb-2 pr-4">Weight</th>
                    <th className="pb-2 pr-4">Return</th>
                    <th className="pb-2 pr-4">Alpha</th>
                    <th className="pb-2 pr-4">Beta</th>
                    <th className="pb-2 pr-4">Sharpe</th>
                    <th className="pb-2 pr-4">Max DD</th>
                    <th className="pb-2 pr-4">Win Rate</th>
                    <th className="pb-2">Trades</th>
                  </tr>
                </thead>
                <tbody>
                  {result.ticker_results.map((tr, i) => {
                    const pos = tr.total_return >= 0
                    return (
                      <tr key={tr.ticker} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                        <td className="py-2 pr-4 font-mono font-bold"
                          style={{ color: LINE_COLORS[i % LINE_COLORS.length] }}>
                          {tr.ticker}
                        </td>
                        <td className="py-2 pr-4 text-slate-400">
                          {(tr.weight * 100).toFixed(1)}%
                        </td>
                        <td className={`py-2 pr-4 font-mono ${pos ? 'text-green-400' : 'text-red-400'}`}>
                          {pos ? '+' : ''}{(tr.total_return * 100).toFixed(1)}%
                        </td>
                        <td className={`py-2 pr-4 font-mono ${tr.alpha >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {tr.alpha >= 0 ? '+' : ''}{(tr.alpha * 100).toFixed(1)}%
                        </td>
                        <td className={`py-2 pr-4 font-mono ${Math.abs(tr.beta) < 0.5 ? 'text-green-400' : 'text-yellow-400'}`}>
                          {tr.beta.toFixed(2)}
                        </td>
                        <td className={`py-2 pr-4 font-mono ${tr.sharpe >= 1 ? 'text-green-400' : tr.sharpe >= 0 ? 'text-yellow-400' : 'text-red-400'}`}>
                          {tr.sharpe.toFixed(2)}
                        </td>
                        <td className="py-2 pr-4 font-mono text-red-400">{(tr.max_drawdown * 100).toFixed(1)}%</td>
                        <td className={`py-2 pr-4 font-mono ${tr.win_rate >= 0.5 ? 'text-green-400' : 'text-yellow-400'}`}>
                          {(tr.win_rate * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 font-mono text-slate-300">{tr.n_trades}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {result.ticker_results.some(r => r.error) && (
              <div className="mt-3 space-y-1">
                {result.ticker_results.filter(r => r.error).map(r => (
                  <p key={r.ticker} className="text-xs text-red-400">
                    {r.ticker}: {r.error}
                  </p>
                ))}
              </div>
            )}
          </Panel>

          {/* ── Correlation heatmap ── */}
          <Panel title="Return Correlation Matrix">
            <div className="overflow-x-auto">
              <table className="text-xs font-mono">
                <thead>
                  <tr>
                    <th className="pr-3 text-slate-500" />
                    {result.tickers.map(t => (
                      <th key={t} className="pb-2 px-2 text-slate-400 text-center">{t}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.correlation_matrix.map((row, i) => (
                    <tr key={result.tickers[i]}>
                      <td className="pr-3 text-slate-400 text-right py-1">{result.tickers[i]}</td>
                      {row.map((v, j) => (
                        <td key={j} className="px-2 py-1 text-center rounded"
                          style={{ background: corrColor(safeNum(v)), color: '#e2e8f0', minWidth: 48 }}>
                          {safeNum(v).toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Green = positive correlation, red = negative correlation.
            </p>
          </Panel>
        </>
      )}
    </div>
  )
}

// ── Page root with mode toggle ─────────────────────────────────────────────

type Mode = 'single' | 'portfolio'

export function BacktestPage() {
  const [mode, setMode] = useState<Mode>('single')

  return (
    <div className="space-y-4">
      {/* Mode toggle */}
      <div className="flex gap-1 bg-slate-800 rounded-lg p-1 w-fit">
        {(['single', 'portfolio'] as Mode[]).map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors capitalize ${
              mode === m
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {m === 'single' ? 'Single Ticker' : 'Portfolio'}
          </button>
        ))}
      </div>

      {mode === 'single' ? <SingleBacktest /> : <PortfolioBacktest />}
    </div>
  )
}
