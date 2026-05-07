import { useState } from 'react'
import { PlayCircle, StopCircle, RefreshCw, AlertTriangle } from 'lucide-react'
import { api } from '../api/client'
import type { PaperStatusResponse } from '../types'
import { Field, Btn, Panel, ErrorBox } from '../components/ui'
import { useActiveOps } from '../contexts/ActiveOpsContext'

const DEFAULTS = {
  ticker: 'AAPL', interval: '1d' as const,
  broker_preset: 'retail' as const,
  initial_capital: 100_000, prediction_interval: 1,
  max_drawdown_limit: 0.15, stop_loss_pct: 0.05,
  latency_ms: 100, model_path: '',
  max_position_pct: 0.40, max_leverage: 1.5, rebalance_threshold: 0.05,
}

function pct(v: number, digits = 2) {
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`
}

function EngineCard({
  engine, onStop, onRefresh,
}: {
  engine: PaperStatusResponse
  onStop: (t: string) => void
  onRefresh: (t: string) => void
}) {
  const p      = engine.portfolio
  const r      = engine.risk_status
  const c      = engine.cost_summary
  const posRet = p.total_return >= 0

  return (
    <div className={`bg-slate-800 rounded-xl border p-4 space-y-4 ${
      r.halted ? 'border-red-700' : engine.running ? 'border-blue-700' : 'border-slate-700'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${engine.running ? 'bg-green-400 animate-pulse' : 'bg-slate-600'}`} />
          <span className="font-mono font-bold text-slate-100 text-lg">{engine.ticker}</span>
          {r.halted && (
            <span className="flex items-center gap-1 text-xs text-red-400 bg-red-950 px-2 py-0.5 rounded-full">
              <AlertTriangle size={10} /> HALTED
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onRefresh(engine.ticker)}
            className="p-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-400 hover:text-slate-200 transition-colors"
            title="Refresh"
          >
            <RefreshCw size={13} />
          </button>
          {engine.running && (
            <button
              onClick={() => onStop(engine.ticker)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-900 hover:bg-red-800 text-red-300 text-xs font-semibold transition-colors"
            >
              <StopCircle size={12} /> Stop
            </button>
          )}
        </div>
      </div>

      {/* Portfolio summary */}
      <div className="grid grid-cols-3 lg:grid-cols-6 gap-2">
        {[
          { label: 'Equity',    value: `$${p.equity.toLocaleString(undefined,{maximumFractionDigits:0})}` },
          { label: 'Return',    value: pct(p.total_return), color: posRet ? 'text-green-400' : 'text-red-400' },
          { label: 'Drawdown',  value: pct(p.drawdown),    color: 'text-red-400' },
          { label: 'Sharpe',    value: p.sharpe.toFixed(2) },
          { label: 'Bars',      value: String(engine.bar_count) },
          { label: 'Orders',    value: String(engine.n_orders) },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-slate-700 rounded-lg p-2">
            <div className="text-xs text-slate-500 mb-0.5">{label}</div>
            <div className={`font-mono text-sm font-bold ${color ?? 'text-slate-100'}`}>{value}</div>
          </div>
        ))}
      </div>

      {/* Risk status */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 text-xs">
        <div className="text-slate-500">Stop loss <span className="text-slate-300">{(r.stop_loss_pct*100).toFixed(0)}%</span></div>
        <div className="text-slate-500">Max DD <span className="text-slate-300">{(r.max_drawdown_limit*100).toFixed(0)}%</span></div>
        <div className="text-slate-500">Vol target <span className="text-slate-300">{(r.vol_target*100).toFixed(0)}%</span></div>
        <div className="text-slate-500">Trailing stop <span className="text-slate-300">{r.trailing_stop ? `${(r.trailing_stop_pct*100).toFixed(0)}%` : 'off'}</span></div>
      </div>

      {/* Costs */}
      {c.n_fills > 0 && (
        <div className="text-xs text-slate-500 border-t border-slate-700 pt-2">
          {c.n_fills} fills · commission ${c.total_commission.toFixed(2)} ·
          slippage ${c.total_slippage.toFixed(2)} ·
          lat avg {c.avg_latency_ms.toFixed(0)}ms
        </div>
      )}

      {r.halt_reason && (
        <div className="text-xs text-red-400 bg-red-950 rounded p-2">{r.halt_reason}</div>
      )}
    </div>
  )
}

export function PaperPage() {
  const [form, setForm]       = useState(DEFAULTS)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const { engines, addEngine, stopEngine, refreshEngine } = useActiveOps()

  const set = (k: string, v: unknown) => setForm(f => ({ ...f, [k]: v }))

  async function start() {
    setLoading(true); setError(null)
    try {
      const status = await api.paperStart({
        ticker:              form.ticker.toUpperCase(),
        interval:            form.interval,
        broker_preset:       form.broker_preset,
        initial_capital:     form.initial_capital,
        prediction_interval: form.prediction_interval,
        max_drawdown_limit:  form.max_drawdown_limit,
        stop_loss_pct:       form.stop_loss_pct,
        latency_ms:          form.latency_ms,
        max_position_pct:    form.max_position_pct,
        max_leverage:        form.max_leverage,
        rebalance_threshold: form.rebalance_threshold,
        model_path:          form.model_path || null,
      })
      addEngine(status)
    } catch (e: unknown) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-5">
      <Panel title="Start Paper Engine" icon={<PlayCircle size={14} />}>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
          <Field label="Ticker">
            <input className="inp" value={form.ticker}
              onChange={e => set('ticker', e.target.value.toUpperCase())} />
          </Field>
          <Field label="Interval">
            <select className="inp" value={form.interval}
              onChange={e => set('interval', e.target.value)}>
              {['1m','5m','15m','1h','1d'].map(v => <option key={v}>{v}</option>)}
            </select>
          </Field>
          <Field label="Broker">
            <select className="inp" value={form.broker_preset}
              onChange={e => set('broker_preset', e.target.value)}>
              {['zero_cost','retail','institutional'].map(v =>
                <option key={v} value={v}>{v.replace('_',' ')}</option>
              )}
            </select>
          </Field>
          <Field label="Initial Capital ($)">
            <input className="inp" type="number" step={1000} value={form.initial_capital}
              onChange={e => set('initial_capital', +e.target.value)} />
          </Field>
          <Field label="Predict Every N Bars">
            <input className="inp" type="number" min={1} max={100} value={form.prediction_interval}
              onChange={e => set('prediction_interval', +e.target.value)} />
          </Field>
          <Field label="Max DD Limit">
            <input className="inp" type="number" step={0.01} min={0.01} max={1} value={form.max_drawdown_limit}
              onChange={e => set('max_drawdown_limit', +e.target.value)} />
          </Field>
          <Field label="Stop Loss %">
            <input className="inp" type="number" step={0.01} min={0.01} max={1} value={form.stop_loss_pct}
              onChange={e => set('stop_loss_pct', +e.target.value)} />
          </Field>
          <Field label="Simulated Latency (ms)">
            <input className="inp" type="number" step={10} min={0} value={form.latency_ms}
              onChange={e => set('latency_ms', +e.target.value)} />
          </Field>
          <Field label="Max Position %">
            <input className="inp" type="number" step={0.05} min={0.05} max={1} value={form.max_position_pct}
              onChange={e => set('max_position_pct', +e.target.value)} />
          </Field>
          <Field label="Max Leverage">
            <input className="inp" type="number" step={0.1} min={0.1} max={5} value={form.max_leverage}
              onChange={e => set('max_leverage', +e.target.value)} />
          </Field>
          <Field label="Rebalance Threshold">
            <input className="inp" type="number" step={0.01} min={0.01} max={0.5} value={form.rebalance_threshold}
              onChange={e => set('rebalance_threshold', +e.target.value)} />
          </Field>
        </div>
        <Field label="Artifact Path (optional)">
          <input className="inp" placeholder="Use pre-trained model (blank = train fresh)"
            value={form.model_path} onChange={e => set('model_path', e.target.value)} />
        </Field>
        <div className="mt-4">
          <Btn loading={loading} onClick={start}>
            <PlayCircle size={13} className="inline mr-1.5" />
            {loading ? 'Starting…' : 'Start Paper Trading'}
          </Btn>
        </div>
        {error && <ErrorBox msg={error} />}
      </Panel>

      {engines.length > 0 && (
        <div>
          <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">
            Active Engines ({engines.length})
          </h2>
          <div className="space-y-3">
            {engines.map(e => (
              <EngineCard key={e.ticker} engine={e} onStop={stopEngine} onRefresh={refreshEngine} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
