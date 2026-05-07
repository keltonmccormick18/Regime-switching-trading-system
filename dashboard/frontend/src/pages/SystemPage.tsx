import { useEffect, useState } from 'react'
import { Server, RefreshCw, ShieldAlert, ShieldCheck } from 'lucide-react'
import { api } from '../api/client'
import type { HealthResponse, RegimeResponse, StrategyStatusResponse } from '../types'
import { Btn, Panel, Badge, ErrorBox, Stat, regimeColor } from '../components/ui'

function Dot({ ok }: { ok: boolean }) {
  return (
    <span className={`inline-block w-2.5 h-2.5 rounded-full mr-2 ${
      ok ? 'bg-green-400' : 'bg-red-500'
    }`} />
  )
}

export function SystemPage() {
  const [health,   setHealth]   = useState<HealthResponse | null>(null)
  const [strategy, setStrategy] = useState<StrategyStatusResponse | null>(null)
  const [regimeTicker, setRegimeTicker] = useState('AAPL')
  const [regime,   setRegime]   = useState<RegimeResponse | null>(null)
  const [loading,  setLoading]  = useState(false)
  const [regLoading, setRegLoading] = useState(false)
  const [resetMsg, setResetMsg]    = useState<string | null>(null)
  const [error,    setError]    = useState<string | null>(null)

  async function loadAll() {
    setLoading(true); setError(null)
    try {
      const [h, s] = await Promise.all([api.health(), api.strategyStatus()])
      setHealth(h)
      setStrategy(s)
    } catch (e: unknown) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadAll() }, [])

  async function checkRegime() {
    setRegLoading(true); setRegime(null)
    try {
      const r = await api.regime(regimeTicker.toUpperCase())
      setRegime(r)
    } catch (e: unknown) {
      setError(String(e))
    } finally {
      setRegLoading(false)
    }
  }

  async function resetHalt() {
    setResetMsg(null)
    try {
      const r = await api.strategyReset()
      setResetMsg(r.status === 'ok' ? 'Halt cleared. Trading resumed.' : JSON.stringify(r))
      await loadAll()
    } catch (e: unknown) {
      setError(String(e))
    }
  }

  const p = strategy?.portfolio
  const r = strategy?.risk_status

  return (
    <div className="space-y-5">
      {/* ── Health ── */}
      <Panel title="System Health" icon={<Server size={14} />}>
        <div className="flex items-center gap-4 mb-3">
          <button
            onClick={loadAll}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-700
                       hover:bg-slate-600 text-slate-300 text-xs font-semibold transition-colors disabled:opacity-50"
          >
            <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
          {health && (
            <span className={`text-sm font-semibold ${
              health.status === 'ok' ? 'text-green-400' : 'text-yellow-400'
            }`}>
              {health.status.toUpperCase()}
            </span>
          )}
        </div>

        {health ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="bg-slate-700 rounded-lg p-3">
              <div className="text-xs text-slate-500 mb-1">PostgreSQL</div>
              <div className="flex items-center">
                <Dot ok={health.postgres} />
                <span className={health.postgres ? 'text-green-400' : 'text-red-400'}>
                  {health.postgres ? 'Connected' : 'Offline'}
                </span>
              </div>
            </div>
            <div className="bg-slate-700 rounded-lg p-3">
              <div className="text-xs text-slate-500 mb-1">Redis</div>
              <div className="flex items-center">
                <Dot ok={health.redis} />
                <span className={health.redis ? 'text-green-400' : 'text-red-400'}>
                  {health.redis ? 'Connected' : 'Offline'}
                </span>
              </div>
            </div>
            <div className="bg-slate-700 rounded-lg p-3">
              <div className="text-xs text-slate-500 mb-1">API Version</div>
              <div className="text-slate-200 font-mono">{health.version}</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-3">
              <div className="text-xs text-slate-500 mb-1">Checked</div>
              <div className="text-slate-400 text-xs font-mono">
                {new Date(health.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ) : (
          <p className="text-slate-500 text-sm">Loading…</p>
        )}
        {error && <ErrorBox msg={error} />}
      </Panel>

      {/* ── Regime checker ── */}
      <Panel title="Regime Detection">
        <div className="flex gap-3 mb-4">
          <input
            className="inp flex-1 max-w-xs"
            value={regimeTicker}
            onChange={e => setRegimeTicker(e.target.value.toUpperCase())}
            placeholder="AAPL"
            onKeyDown={e => e.key === 'Enter' && checkRegime()}
          />
          <Btn loading={regLoading} onClick={checkRegime} compact>
            Detect Regime
          </Btn>
        </div>
        {regime && (
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
            <div className="lg:col-span-2 bg-slate-700 rounded-lg p-3 flex flex-col gap-1">
              <span className="text-xs text-slate-500">Regime</span>
              <Badge color={regimeColor(regime.regime)} large>
                {regime.regime.replace(/_/g, ' ')}
              </Badge>
            </div>
            <div className="bg-slate-700 rounded-lg p-3">
              <div className="text-xs text-slate-500 mb-1">20-Day Vol</div>
              <div className="font-mono text-slate-100">
                {regime.vol_20 != null ? `${(regime.vol_20 * 100).toFixed(2)}%` : '—'}
              </div>
            </div>
            <div className="bg-slate-700 rounded-lg p-3">
              <div className="text-xs text-slate-500 mb-1">SMA 200 Ratio</div>
              <div className={`font-mono ${
                (regime.sma_ratio_200 ?? 0) > 1 ? 'text-green-400' : 'text-red-400'
              }`}>
                {regime.sma_ratio_200 != null ? regime.sma_ratio_200.toFixed(4) : '—'}
              </div>
            </div>
            <div className="bg-slate-700 rounded-lg p-3">
              <div className="text-xs text-slate-500 mb-1">TDA L1</div>
              <div className="font-mono text-slate-100">
                {regime.tda_l1 != null ? regime.tda_l1.toFixed(4) : '—'}
              </div>
            </div>
          </div>
        )}
      </Panel>

      {/* ── Strategy status ── */}
      {strategy && p && r && (
        <Panel title="Strategy Engine">
          {/* Risk halt banner */}
          {r.halted ? (
            <div className="flex items-center gap-3 bg-red-950 border border-red-800 rounded-xl p-3 mb-4">
              <ShieldAlert size={18} className="text-red-400 shrink-0" />
              <div className="flex-1">
                <p className="text-red-300 font-semibold text-sm">Trading Halted</p>
                <p className="text-red-400 text-xs mt-0.5">{r.halt_reason}</p>
              </div>
              <button
                onClick={resetHalt}
                className="px-3 py-1.5 rounded-lg bg-red-800 hover:bg-red-700 text-red-200 text-xs font-semibold transition-colors"
              >
                Reset Halt
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-green-400 text-sm mb-4">
              <ShieldCheck size={16} /> Trading Active
              {resetMsg && <span className="text-xs text-slate-400 ml-2">{resetMsg}</span>}
            </div>
          )}

          {/* Portfolio metrics */}
          <div className="grid grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
            <Stat label="Equity"
              value={`$${p.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
              color="text-slate-100" />
            <Stat label="Return"    value={`${p.total_return >= 0 ? '+' : ''}${(p.total_return*100).toFixed(2)}%`}
              color={p.total_return >= 0 ? 'text-green-400' : 'text-red-400'} />
            <Stat label="Drawdown"  value={`${(p.drawdown*100).toFixed(2)}%`}          color="text-red-400" />
            <Stat label="Sharpe"    value={p.sharpe.toFixed(2)}                          color="text-slate-100" />
            <Stat label="Signals"   value={String(strategy.n_signals)}                  color="text-slate-100" />
            <Stat label="Orders"    value={String(strategy.n_orders)}                   color="text-slate-100" />
          </div>

          {/* Risk config */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 text-xs">
            {[
              { label: 'Max Drawdown Limit', value: `${(r.max_drawdown_limit*100).toFixed(0)}%` },
              { label: 'Stop Loss',          value: `${(r.stop_loss_pct*100).toFixed(0)}%` },
              { label: 'Trailing Stop',      value: r.trailing_stop ? `${(r.trailing_stop_pct*100).toFixed(0)}%` : 'off' },
              { label: 'Vol Target',         value: `${(r.vol_target*100).toFixed(0)}%` },
            ].map(({ label, value }) => (
              <div key={label} className="bg-slate-700 rounded-lg p-2">
                <div className="text-slate-500 mb-0.5">{label}</div>
                <div className="text-slate-200 font-mono">{value}</div>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  )
}
