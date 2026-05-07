import { useState } from 'react'
import { Cpu, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { api } from '../api/client'
import type { PredictResponse } from '../types'
import { Field, Btn, Panel, Badge, ErrorBox, regimeColor } from '../components/ui'

const DEFAULTS = {
  ticker: 'AAPL', start: '2010-01-01', horizon: 16,
  lookback: 64, use_tda: false, use_macro: false, model_path: '',
}

export function PredictPage() {
  const [form, setForm]       = useState(DEFAULTS)
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState<PredictResponse | null>(null)
  const [error, setError]     = useState<string | null>(null)

  const set = (k: string, v: unknown) => setForm(f => ({ ...f, [k]: v }))

  async function run() {
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await api.predict({
        ticker:     form.ticker.toUpperCase(),
        start:      form.start,
        horizon:    form.horizon,
        lookback:   form.lookback,
        use_tda:    form.use_tda,
        use_macro:  form.use_macro,
        model_path: form.model_path || null,
      })
      setResult(r)
    } catch (e: unknown) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const chartData = result
    ? result.prediction.map((p, i) => ({ bar: `+${i + 1}`, price: +p.toFixed(2) }))
    : []

  const first = result?.prediction[0]
  const last  = result?.prediction[result.prediction.length - 1]
  const pct   = first && last ? ((last - first) / first * 100) : null
  const bull  = pct != null && pct > 0

  return (
    <div className="space-y-5">
      <Panel title="Run Prediction" icon={<Cpu size={14} />}>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
          <Field label="Ticker">
            <input className="inp" value={form.ticker}
              onChange={e => set('ticker', e.target.value.toUpperCase())} />
          </Field>
          <Field label="History Start">
            <input className="inp" type="date" value={form.start}
              onChange={e => set('start', e.target.value)} />
          </Field>
          <Field label="Horizon (bars)">
            <input className="inp" type="number" min={1} max={252} value={form.horizon}
              onChange={e => set('horizon', +e.target.value)} />
          </Field>
          <Field label="Lookback (bars)">
            <input className="inp" type="number" min={10} max={512} value={form.lookback}
              onChange={e => set('lookback', +e.target.value)} />
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-3 mb-4">
          <Field label="Artifact Path (optional)">
            <input className="inp" placeholder="artifacts/models/SPY_LOW_VOL_BULL_....pt"
              value={form.model_path} onChange={e => set('model_path', e.target.value)} />
          </Field>
          <Field label="Use TDA features">
            <label className="flex items-center gap-2 mt-2 cursor-pointer select-none">
              <input type="checkbox" className="w-4 h-4 accent-blue-500"
                checked={form.use_tda} onChange={e => set('use_tda', e.target.checked)} />
              <span className="text-sm text-slate-300">
                Enable (slow — requires gudhi)
              </span>
            </label>
          </Field>
          <Field label="Cross-asset macro">
            <label className="flex items-center gap-2 mt-2 cursor-pointer select-none">
              <input type="checkbox" className="w-4 h-4 accent-emerald-500"
                checked={form.use_macro} onChange={e => set('use_macro', e.target.checked)} />
              <span className="text-sm text-slate-300">VIX + credit + USD</span>
            </label>
          </Field>
        </div>
        <Btn loading={loading} onClick={run}>
          {loading ? 'Running model…' : 'Predict'}
        </Btn>
        {error && <ErrorBox msg={error} />}
      </Panel>

      {result && (
        <>
          {/* ── Summary badges ── */}
          <div className="flex flex-wrap gap-3">
            <Badge color="blue">{result.ticker}</Badge>
            <Badge color={regimeColor(result.regime)}>{result.regime.replace(/_/g,' ')}</Badge>
            <Badge color="slate">{result.model_used}</Badge>
            <Badge color={bull ? 'green' : 'red'}>
              {bull ? <TrendingUp size={12} className="inline mr-1" />
                    : <TrendingDown size={12} className="inline mr-1" />}
              {pct != null ? `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%` : '—'}
              {' '}over {result.horizon} bars
            </Badge>
          </div>

          {/* ── Forecast chart ── */}
          <Panel title={`${result.horizon}-Bar Price Forecast — ${result.ticker}`}>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="bar" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis domain={['auto', 'auto']} tick={{ fill: '#94a3b8', fontSize: 11 }}
                  tickFormatter={(v: number) => `$${v.toFixed(0)}`} width={60} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                  labelStyle={{ color: '#94a3b8' }}
                  formatter={(v: number) => [`$${v.toFixed(2)}`, 'Forecast']}
                />
                <ReferenceLine y={first} stroke="#475569" strokeDasharray="4 4" label="" />
                <Line
                  type="monotone" dataKey="price"
                  stroke={bull ? '#4ade80' : '#f87171'}
                  strokeWidth={2} dot={{ r: 3, fill: bull ? '#4ade80' : '#f87171' }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Panel>

          {/* ── Raw values ── */}
          <Panel title="Raw Prediction Array">
            <div className="flex flex-wrap gap-2">
              {result.prediction.map((p, i) => (
                <span key={i}
                  className="font-mono text-xs px-2 py-1 rounded bg-slate-700 text-slate-300">
                  <span className="text-slate-500 mr-1">+{i+1}</span>
                  ${p.toFixed(2)}
                </span>
              ))}
            </div>
          </Panel>
        </>
      )}
    </div>
  )
}
