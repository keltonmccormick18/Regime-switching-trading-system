import { useState } from 'react'
import { FlaskConical } from 'lucide-react'
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceArea,
} from 'recharts'
import { api } from '../api/client'
import type { BenchmarkResponse, ModelName, RegimeMetrics } from '../types'
import { Field, Btn, Panel, Badge, ErrorBox, Stat } from '../components/ui'

// ── Model metadata ─────────────────────────────────────────────────────────

const MODEL_INFO: Record<ModelName, { label: string; native: string; desc: string; color: string }> = {
  TFTModel: {
    label:  'TFT (Temporal Fusion Transformer)',
    native: 'HIGH_VOL_BULL',
    desc:   'Variable-selection network + LSTM + multi-head attention + quantile heads. Excels at volatile uptrends.',
    color:  '#fbbf24',
  },
  OnlineModel: {
    label:  'Online ARF (Adaptive Random Forest)',
    native: 'HIGH_VOL_BEAR',
    desc:   'River ARF with ADWIN drift detection — continuously adapts to bear market regime shifts without full retraining.',
    color:  '#f87171',
  },
  TCNModel: {
    label:  'TCN (Temporal Convolutional Network)',
    native: 'LOW_VOL_BULL',
    desc:   'Causal dilated Conv1d with residual connections. Optimal for smooth, low-volatility uptrends.',
    color:  '#4ade80',
  },
  TCNLSTMModel: {
    label:  'TCN-LSTM',
    native: 'LOW_VOL_BEAR',
    desc:   'TCN feature extraction → LSTM sequential modelling. Handles mixed dynamics in calm bear regimes.',
    color:  '#93c5fd',
  },
}

// ── Regime display helpers ─────────────────────────────────────────────────

const REGIME_LABEL: Record<string, string> = {
  HIGH_VOL_BULL: 'High Vol Bull',
  HIGH_VOL_BEAR: 'High Vol Bear',
  LOW_VOL_BULL:  'Low Vol Bull',
  LOW_VOL_BEAR:  'Low Vol Bear',
}

const REGIME_FILL: Record<string, string> = {
  HIGH_VOL_BULL: 'rgba(251,191,36,0.13)',
  HIGH_VOL_BEAR: 'rgba(248,113,113,0.13)',
  LOW_VOL_BULL:  'rgba(74,222,128,0.10)',
  LOW_VOL_BEAR:  'rgba(147,197,253,0.10)',
}

const REGIME_DOT: Record<string, string> = {
  HIGH_VOL_BULL: '#fbbf24',
  HIGH_VOL_BEAR: '#f87171',
  LOW_VOL_BULL:  '#4ade80',
  LOW_VOL_BEAR:  '#93c5fd',
}

const REGIME_BADGE_COLOR: Record<string, string> = {
  HIGH_VOL_BULL: 'yellow',
  HIGH_VOL_BEAR: 'red',
  LOW_VOL_BULL:  'green',
  LOW_VOL_BEAR:  'blue',
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

const safeNum = (v: number | null | undefined) => (v != null && isFinite(v) ? v : 0)
const pct     = (v: number, decimals = 1) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(decimals)}%`
const num2    = (v: number) => v.toFixed(2)

// ── Regime breakdown table ─────────────────────────────────────────────────

function RegimeBreakdownTable({
  rows, nativeRegime, isMC,
}: {
  rows: RegimeMetrics[]
  nativeRegime: string
  isMC: boolean
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-slate-500 text-xs uppercase border-b border-slate-700">
            <th className="pb-2 pr-4">Regime</th>
            <th className="pb-2 pr-4">Role</th>
            <th className="pb-2 pr-4">Bars</th>
            <th className="pb-2 pr-4">Return{isMC && ' (mean ± σ)'}</th>
            <th className="pb-2 pr-4">Sharpe{isMC && ' (mean ± σ)'}</th>
            <th className="pb-2 pr-4">Max DD</th>
            <th className="pb-2">Win Rate</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => {
            const pos = r.total_return >= 0
            return (
              <tr
                key={r.regime}
                className={`border-b border-slate-700/50 ${r.is_native ? 'bg-slate-700/40' : 'hover:bg-slate-700/20'}`}
              >
                <td className="py-2.5 pr-4">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full inline-block shrink-0"
                      style={{ background: REGIME_DOT[r.regime] }}
                    />
                    <span className="font-mono text-slate-200">
                      {REGIME_LABEL[r.regime] ?? r.regime}
                    </span>
                  </div>
                </td>
                <td className="py-2.5 pr-4">
                  {r.is_native ? (
                    <Badge color={REGIME_BADGE_COLOR[r.regime] ?? 'slate'} small>Native</Badge>
                  ) : (
                    <span className="text-xs text-slate-500">Out-of-regime</span>
                  )}
                </td>
                <td className="py-2.5 pr-4 font-mono text-slate-300">
                  {r.n_bars > 0 ? r.n_bars : <span className="text-slate-600">—</span>}
                </td>
                <td className={`py-2.5 pr-4 font-mono font-semibold ${pos ? 'text-green-400' : 'text-red-400'}`}>
                  {r.n_bars > 0
                    ? <>{pct(r.total_return)}{isMC && r.total_return_std > 0 && <span className="text-xs text-slate-500 font-normal"> ±{pct(r.total_return_std)}</span>}</>
                    : <span className="text-slate-600">—</span>}
                </td>
                <td className={`py-2.5 pr-4 font-mono ${r.sharpe >= 1 ? 'text-green-400' : r.sharpe >= 0 ? 'text-yellow-400' : 'text-red-400'}`}>
                  {r.n_bars > 0
                    ? <>{num2(r.sharpe)}{isMC && r.sharpe_std > 0 && <span className="text-xs text-slate-500 font-normal"> ±{num2(r.sharpe_std)}</span>}</>
                    : <span className="text-slate-600">—</span>}
                </td>
                <td className="py-2.5 pr-4 font-mono text-red-400">
                  {r.n_bars > 0 ? `${(r.max_drawdown * 100).toFixed(1)}%` : <span className="text-slate-600">—</span>}
                </td>
                <td className={`py-2.5 font-mono ${r.win_rate >= 0.5 ? 'text-green-400' : 'text-yellow-400'}`}>
                  {r.n_bars > 0 ? `${(r.win_rate * 100).toFixed(1)}%` : <span className="text-slate-600">—</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="text-xs text-slate-500 mt-2">
        Native regime = the market condition this model is specifically designed for.
        Metrics are computed only from bars where the market was in that regime.
        {isMC && ' Mean ± 1σ across all seeds — large σ means the result is seed-sensitive.'}
      </p>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────

const MODELS: ModelName[] = ['TFTModel', 'OnlineModel', 'TCNModel', 'TCNLSTMModel']
const PRESETS = ['zero_cost', 'retail', 'institutional'] as const

export function BenchmarkPage() {
  const [form, setForm] = useState({
    ticker:           'AAPL',
    model_name:       'TCNModel' as ModelName,
    start:            '2010-01-01',
    end:              '',
    use_tda:          false,
    use_macro:        false,
    lookback:         64,
    horizon:          16,
    position_size:    0.9,
    broker_preset:    'retail' as const,
    initial_capital:  100_000,
    signal_hold:      16,
    long_only:        !MODEL_INFO['TCNModel'].native.includes('BEAR'),
    min_confidence:   0.0,
    stop_loss_pct:    0.0,
    n_simulations:    1,
    regime_thresholds: {
      HIGH_VOL_BULL: 0.008,
      HIGH_VOL_BEAR: 0.005,
      LOW_VOL_BULL:  0.002,
      LOW_VOL_BEAR:  0.003,
    },
  })

  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState<BenchmarkResponse | null>(null)
  const [error,   setError]   = useState<string | null>(null)

  const set = (k: string, v: unknown) => setForm(f => ({ ...f, [k]: v }))

  const info       = MODEL_INFO[form.model_name]
  const modelColor = info.color

  async function run() {
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await api.benchmark({
        ticker:           form.ticker.toUpperCase(),
        model_name:       form.model_name,
        start:            form.start,
        end:              form.end || null,
        use_tda:          form.use_tda,
        use_macro:        form.use_macro,
        lookback:         form.lookback,
        horizon:          form.horizon,
        position_size:    form.position_size,
        broker_preset:    form.broker_preset,
        initial_capital:  form.initial_capital,
        signal_hold:              form.signal_hold,
        long_only:                form.long_only,
        signal_threshold:         0.005,
        min_confidence:           form.min_confidence,
        stop_loss_pct:            form.stop_loss_pct,
        regime_signal_thresholds: form.regime_thresholds,
        n_simulations:            form.n_simulations,
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
  const equityData = result
    ? result.equity.map((v, i) => ({
        bar:    i,
        equity: +safeNum(v).toFixed(2),
        bah:    +safeNum((result.bah_equity ?? [])[i]).toFixed(2),
        ...(ss ? {
          p10: +safeNum(ss.equity_p10[i]).toFixed(2),
          p90: +safeNum(ss.equity_p90[i]).toFixed(2),
        } : {}),
      }))
    : []

  const ddData = result
    ? result.equity.map((v, i, arr) => {
        const peak = Math.max(...arr.slice(0, i + 1))
        const dd   = peak > 0 ? ((v - peak) / peak * 100) : 0
        return { bar: i, drawdown: +dd.toFixed(3) }
      })
    : []

  const pos = result && result.overall_return >= 0

  const nativeRow = result?.regime_breakdown.find(r => r.is_native)

  function mcStat(val: number, std: number | undefined, fmt: (v: number) => string) {
    if (!ss || !std) return fmt(val)
    return <>{fmt(val)} <span className="text-xs text-slate-500 font-normal">±{fmt(std)}</span></>
  }

  return (
    <div className="space-y-5">
      {/* ── Configuration ── */}
      <Panel title="Model Regime Benchmark" icon={<FlaskConical size={14} />}>

        {/* Model picker */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-4">
          {MODELS.map(m => {
            const mi = MODEL_INFO[m]
            const active = form.model_name === m
            return (
              <button
                key={m}
                onClick={() => setForm(f => ({
                  ...f,
                  model_name: m,
                  long_only: !MODEL_INFO[m].native.includes('BEAR'),
                }))}
                className={`text-left p-3 rounded-lg border transition-all ${
                  active
                    ? 'border-blue-500 bg-blue-950/40'
                    : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ background: mi.color }}
                  />
                  <span className="text-sm font-semibold text-slate-100">{mi.label}</span>
                </div>
                <div className="flex items-center gap-2 mb-1.5 ml-4">
                  <Badge color={REGIME_BADGE_COLOR[mi.native] ?? 'slate'} small>
                    {REGIME_LABEL[mi.native]}
                  </Badge>
                  <span className="text-xs text-slate-500">native regime</span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed ml-4">{mi.desc}</p>
              </button>
            )
          })}
        </div>

        {/* Config fields */}
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
          <Field label="Position Size">
            <input className="inp" type="number" step={0.01} min={0.01} max={0.5}
              value={form.position_size}
              onChange={e => set('position_size', +e.target.value)} />
          </Field>
          <Field label="Initial Capital ($)">
            <input className="inp" type="number" step={1000} min={1000}
              value={form.initial_capital}
              onChange={e => set('initial_capital', +e.target.value)} />
          </Field>
          <Field label="Lookback">
            <input className="inp" type="number" min={10} max={512}
              value={form.lookback}
              onChange={e => set('lookback', +e.target.value)} />
          </Field>
          <Field label="Horizon">
            <input className="inp" type="number" min={1} max={252}
              value={form.horizon}
              onChange={e => set('horizon', +e.target.value)} />
          </Field>
        </div>

        {/* Signal structure */}
        <div className="border border-slate-700 rounded-lg p-3 mb-4">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">Signal Structure</p>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-3">
            <Field label="Signal Hold (bars)">
              <input className="inp" type="number" min={1} max={252}
                value={form.signal_hold}
                onChange={e => set('signal_hold', +e.target.value)} />
            </Field>
            <Field label="Direction">
              <select className="inp" value={form.long_only ? 'long_only' : 'long_short'}
                onChange={e => set('long_only', e.target.value === 'long_only')}>
                <option value="long_only">Long / Flat</option>
                <option value="long_short">Long / Short</option>
              </select>
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

          {/* Per-regime signal thresholds */}
          <div className="border-t border-slate-700/60 pt-3">
            <p className="text-xs text-slate-500 mb-2">
              Signal threshold by regime — min |predicted return| to generate a trade.
              Low-vol regimes need a lower bar than high-vol ones.
            </p>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              {(Object.keys(form.regime_thresholds) as Array<keyof typeof form.regime_thresholds>).map(r => (
                <Field key={r} label={
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full inline-block shrink-0" style={{ background: REGIME_DOT[r] }} />
                    {REGIME_LABEL[r]}
                  </span>
                }>
                  <input className="inp" type="number" min={0} max={0.2} step={0.001}
                    value={form.regime_thresholds[r]}
                    onChange={e => setForm(f => ({
                      ...f,
                      regime_thresholds: { ...f.regime_thresholds, [r]: +e.target.value },
                    }))} />
                </Field>
              ))}
            </div>
          </div>

          <div className="mt-3 flex gap-6 flex-wrap">
            <Field label="Use TDA features">
              <label className="flex items-center gap-2 mt-2 cursor-pointer">
                <input type="checkbox" className="w-4 h-4 accent-blue-500"
                  checked={form.use_tda}
                  onChange={e => set('use_tda', e.target.checked)} />
                <span className="text-sm text-slate-300">Enable TDA (slow)</span>
              </label>
            </Field>
            <Field label="Cross-asset macro">
              <label className="flex items-center gap-2 mt-2 cursor-pointer">
                <input type="checkbox" className="w-4 h-4 accent-emerald-500"
                  checked={form.use_macro}
                  onChange={e => set('use_macro', e.target.checked)} />
                <span className="text-sm text-slate-300">VIX + credit + USD</span>
              </label>
            </Field>
          </div>
        </div>

        {/* Monte Carlo */}
        <div className="border border-slate-700 rounded-lg p-3 mb-4">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">Monte Carlo Seeds</p>
          <div className="flex items-center gap-3">
            <input
              type="range" min={1} max={10} step={1}
              className="w-48 accent-blue-500"
              value={form.n_simulations}
              onChange={e => set('n_simulations', +e.target.value)}
            />
            <span className="text-sm font-mono text-slate-300 w-4">{form.n_simulations}</span>
          </div>
          <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
            {form.n_simulations === 1
              ? 'Single run — fast, but one seed may be lucky or unlucky.'
              : `${form.n_simulations} independent weight initialisations — per-regime metrics show mean ± std so you can tell if the native-regime edge is real.`}
          </p>
        </div>

        <Btn loading={loading} onClick={run}>
          {loading
            ? form.n_simulations > 1
              ? `Training ${form.n_simulations} × ${form.model_name}… (may take a few minutes)`
              : `Training ${form.model_name} and running benchmark…`
            : form.n_simulations > 1
              ? `Run ${form.n_simulations}-seed Monte Carlo — ${form.model_name}`
              : `Benchmark ${form.model_name} on ${form.ticker || '…'}`}
        </Btn>
        {error && <ErrorBox msg={error} />}
      </Panel>

      {/* ── Results ── */}
      {result && (
        <>
          {ss && (
            <div className="flex items-center gap-2 px-3 py-2 bg-blue-950/40 border border-blue-800/40 rounded-lg text-xs text-blue-300">
              <span className="font-semibold">Monte Carlo — {ss.n_simulations} seeds</span>
              <span className="text-blue-400/60">·</span>
              <span>Overall metrics are means across all seeds. Per-regime table shows mean ± 1σ.</span>
            </div>
          )}

          {/* Header strip */}
          <div
            className="flex items-center gap-3 px-4 py-3 rounded-lg border"
            style={{ borderColor: `${modelColor}55`, background: `${modelColor}11` }}
          >
            <span className="w-3 h-3 rounded-full shrink-0" style={{ background: modelColor }} />
            <span className="font-semibold text-slate-100">{result.model_name}</span>
            <span className="text-slate-500 text-xs">·</span>
            <Badge color={REGIME_BADGE_COLOR[result.native_regime] ?? 'slate'}>
              Native: {REGIME_LABEL[result.native_regime] ?? result.native_regime}
            </Badge>
            <span className="text-slate-500 text-xs ml-auto">{result.ticker}</span>
          </div>

          {/* Overall metrics */}
          <div className="grid grid-cols-3 lg:grid-cols-8 gap-3">
            <Stat label="Return"
              value={<>
                {mcStat(result.overall_return, ss?.total_return_std, v => pct(v))}
                {result.bah_return != null && (
                  <span className="block text-xs font-normal mt-0.5"
                    style={{ color: result.bah_return >= 0 ? '#4ade80' : '#f87171' }}>
                    B&amp;H {pct(result.bah_return)}
                  </span>
                )}
              </>}
              color={pos ? 'text-green-400' : 'text-red-400'} />
            <Stat label="Sharpe"
              value={mcStat(result.overall_sharpe, ss?.sharpe_std, num2)}
              color={result.overall_sharpe >= 1 ? 'text-green-400' : result.overall_sharpe >= 0 ? 'text-yellow-400' : 'text-red-400'} />
            <Stat label="Sortino"  value={num2(result.overall_sortino)} color="text-slate-100" />
            <Stat label="Max DD"
              value={mcStat(result.overall_max_drawdown, ss?.max_drawdown_std, v => `${(v*100).toFixed(1)}%`)}
              color="text-red-400" />
            <Stat label="Win Rate"
              value={mcStat(result.overall_win_rate, ss?.win_rate_std, v => `${(v*100).toFixed(1)}%`)}
              color={result.overall_win_rate >= 0.5 ? 'text-green-400' : 'text-yellow-400'} />
            <Stat label="Trades"   value={String(result.n_trades)} color="text-slate-100" />
            <Stat label="Alpha"
              value={mcStat(result.alpha, ss?.alpha_std, v => pct(v))}
              color={result.alpha >= 0 ? 'text-green-400' : 'text-red-400'} />
            <Stat label="Beta"
              value={mcStat(result.beta, ss?.beta_std, num2)}
              color={Math.abs(result.beta) < 0.5 ? 'text-green-400' : Math.abs(result.beta) < 1 ? 'text-yellow-400' : 'text-slate-100'} />
          </div>

          {/* Native regime spotlight */}
          {nativeRow && nativeRow.n_bars > 0 && (
            <div
              className="rounded-lg border p-4"
              style={{ borderColor: `${modelColor}44`, background: `${modelColor}0d` }}
            >
              <p className="text-xs font-semibold uppercase tracking-widest mb-2"
                style={{ color: modelColor }}>
                Native Regime Performance — {REGIME_LABEL[result.native_regime]}
              </p>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <Stat label="Native Return"
                  value={pct(nativeRow.total_return)}
                  color={nativeRow.total_return >= 0 ? 'text-green-400' : 'text-red-400'} />
                <Stat label="Native Sharpe"
                  value={num2(nativeRow.sharpe)}
                  color={nativeRow.sharpe >= 1 ? 'text-green-400' : nativeRow.sharpe >= 0 ? 'text-yellow-400' : 'text-red-400'} />
                <Stat label="Native Max DD"
                  value={`${(nativeRow.max_drawdown * 100).toFixed(1)}%`}
                  color="text-red-400" />
                <Stat label="Native Win Rate"
                  value={`${(nativeRow.win_rate * 100).toFixed(1)}%`}
                  color={nativeRow.win_rate >= 0.5 ? 'text-green-400' : 'text-yellow-400'} />
              </div>
              <p className="text-xs text-slate-500 mt-2">
                {nativeRow.n_bars} bars ({((nativeRow.n_bars / result.equity.length) * 100).toFixed(0)}%
                of OOS period) were in the native regime.
              </p>
            </div>
          )}

          {/* Equity curve */}
          <Panel title={ss
            ? `Equity Curve — mean ± p10/p90 band (${ss.n_simulations} seeds, regime bands overlaid)`
            : 'Equity Curve — strategy vs Buy & Hold (regime bands overlaid)'}>
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={equityData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="bmeq" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor={pos ? '#4ade80' : '#f87171'} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={pos ? '#4ade80' : '#f87171'} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="bar" tick={{ fill: '#94a3b8', fontSize: 10 }}
                  tickFormatter={(v: number) => v % 50 === 0 ? `Day ${v}` : ''} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }}
                  tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} width={55} />
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
                {/* MC band: p90 filled, p10 layered on top in background colour */}
                {ss && <>
                  <Area type="monotone" dataKey="p90"
                    fill={pos ? '#4ade8022' : '#f8717122'} stroke="none" fillOpacity={1} dot={false} />
                  <Area type="monotone" dataKey="p10"
                    fill="#0f172a" stroke="none" fillOpacity={1} dot={false} />
                </>}
                <Area type="monotone" dataKey="equity"
                  stroke={pos ? '#4ade80' : '#f87171'} strokeWidth={ss ? 2 : 1.5}
                  fill={ss ? 'none' : 'url(#bmeq)'} dot={false} />
                <Line type="monotone" dataKey="bah"
                  stroke="#e2e8f0" strokeWidth={2} strokeDasharray="6 3"
                  dot={false} legendType="none" strokeOpacity={0.9} />
              </AreaChart>
            </ResponsiveContainer>
            {/* Regime legend */}
            <div className="flex flex-wrap items-center gap-4 mt-2">
              <span className="flex items-center gap-1.5 text-xs text-slate-300">
                <svg width="18" height="8" viewBox="0 0 18 8">
                  <line x1="0" y1="4" x2="18" y2="4" stroke="#e2e8f0" strokeWidth="2" strokeDasharray="6 3" />
                </svg>
                Buy &amp; Hold
              </span>
              {[...new Set(result.regime_curve)].map(r => (
                <span key={r} className="flex items-center gap-1.5 text-xs text-slate-400">
                  <span className="w-3 h-3 rounded-sm inline-block" style={{ background: REGIME_DOT[r] }} />
                  {REGIME_LABEL[r] ?? r}
                  {r === result.native_regime && (
                    <Badge color={REGIME_BADGE_COLOR[r] ?? 'slate'} small>native</Badge>
                  )}
                </span>
              ))}
            </div>
          </Panel>

          {/* Drawdown */}
          <Panel title="Drawdown">
            <ResponsiveContainer width="100%" height={150}>
              <AreaChart data={ddData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="bmdd" x1="0" y1="0" x2="0" y2="1">
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
                {regimeSegments(result.regime_curve ?? []).map((seg, i) => (
                  <ReferenceArea key={i} x1={seg.x1} x2={seg.x2}
                    fill={REGIME_FILL[seg.regime]} strokeOpacity={0} />
                ))}
                <Area type="monotone" dataKey="drawdown"
                  stroke="#f87171" strokeWidth={1.5} fill="url(#bmdd)" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </Panel>

          {/* Per-regime breakdown */}
          <Panel title={ss ? `Performance by Regime — mean ± 1σ across ${ss.n_simulations} seeds` : 'Performance by Regime'}>
            <RegimeBreakdownTable
              rows={result.regime_breakdown}
              nativeRegime={result.native_regime}
              isMC={!!ss}
            />
          </Panel>

          {/* Regime time allocation chart */}
          <Panel title="Regime Time Allocation">
            <div className="flex gap-2 flex-wrap">
              {result.regime_breakdown
                .filter(r => r.n_bars > 0)
                .sort((a, b) => b.n_bars - a.n_bars)
                .map(r => {
                  const pctBars = ((r.n_bars / result.equity.length) * 100)
                  return (
                    <div
                      key={r.regime}
                      className="flex flex-col gap-1 flex-1 min-w-[120px]"
                    >
                      <div className="flex items-center justify-between text-xs text-slate-400">
                        <span className="flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full" style={{ background: REGIME_DOT[r.regime] }} />
                          {REGIME_LABEL[r.regime] ?? r.regime}
                        </span>
                        {r.is_native && <Badge color={REGIME_BADGE_COLOR[r.regime] ?? 'slate'} small>N</Badge>}
                      </div>
                      <div className="h-2 rounded-full bg-slate-700 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${pctBars}%`, background: REGIME_DOT[r.regime] }}
                        />
                      </div>
                      <span className="text-xs font-mono text-slate-400">
                        {pctBars.toFixed(0)}% ({r.n_bars} bars)
                      </span>
                    </div>
                  )
                })}
            </div>
          </Panel>

          {/* Execution costs */}
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
