import { useEffect, useRef, useState } from 'react'
import { RefreshCw, Radio } from 'lucide-react'
import { api } from '../api/client'
import type {
  MetricsSummary, PnLPoint, Position, Signal, PaperStatusResponse,
} from '../types'
import { MetricsCards }   from '../components/MetricsCards'
import { PnLChart }       from '../components/PnLChart'
import { DrawdownChart }  from '../components/DrawdownChart'
import { SignalChart }    from '../components/SignalChart'
import { SignalFeed }     from '../components/SignalFeed'
import { PositionsTable } from '../components/PositionsTable'
import { Panel } from '../components/ui'

const MAX_SIGNALS = 120
const POLL_MS     = 15_000   // refresh paper engine data every 15 s

function mergeSignals(a: Signal[], b: Signal[]): Signal[] {
  const seen = new Set<string>()
  return [...a, ...b]
    .filter(s => {
      const k = `${s.symbol}:${s.timestamp}`
      if (seen.has(k)) return false
      seen.add(k); return true
    })
    .sort((x, y) => new Date(y.timestamp).getTime() - new Date(x.timestamp).getTime())
    .slice(0, MAX_SIGNALS)
}

/** Convert PaperStatusResponse portfolio into the MetricsSummary shape MetricsCards expects. */
function paperToSummary(pe: PaperStatusResponse): MetricsSummary {
  const p = pe.portfolio
  return {
    run_id:       `paper:${pe.ticker}`,
    ticker:       pe.ticker,
    sharpe:       p.sharpe,
    total_return: p.total_return,
    max_drawdown: p.max_drawdown,
    win_rate:     0,
    n_trades:     pe.n_orders,
    regime:       '',
    model_used:   '',
  }
}

/** Convert paper portfolio positions into the Position[] shape PositionsTable expects. */
function paperToPositions(pe: PaperStatusResponse): Position[] {
  // PaperStatusResponse doesn't include individual positions — the backend
  // exposes them via the equity_history field; we surface what's available.
  return []
}

export function DashboardPage({ signals: wsSignals }: { signals: Signal[] }) {
  // ── Core dashboard state ──────────────────────────────────────────────────
  const [summary,    setSummary]    = useState<MetricsSummary | null>(null)
  const [pnlData,    setPnlData]    = useState<PnLPoint[]>([])
  const [positions,  setPositions]  = useState<Position[]>([])
  const [allSignals, setAllSignals] = useState<Signal[]>([])
  const [histSymbol, setHistSymbol] = useState('AAPL')
  const [histInput,  setHistInput]  = useState('AAPL')
  const [histLoad,   setHistLoad]   = useState(false)

  // ── Paper engine state ────────────────────────────────────────────────────
  const [paperEngines,  setPaperEngines]  = useState<PaperStatusResponse[]>([])
  const [activeTicker,  setActiveTicker]  = useState<string | null>(null)

  const prevWsLen = useRef(0)

  // ── Helpers ───────────────────────────────────────────────────────────────
  async function loadPaperEngines() {
    try {
      const list = await api.paperList()
      if (!list.engines.length) { setPaperEngines([]); return }

      const statuses = await Promise.all(
        list.engines.map(e => api.paperStatus(e.ticker).catch(() => null))
      )
      const live = statuses.filter((s): s is PaperStatusResponse => s !== null)
      setPaperEngines(live)

      // Pick the engine with the most bars as the "active" one for charts
      const running = live.filter(e => e.running)
      const best = running.sort((a, b) => b.bar_count - a.bar_count)[0] ?? live[0]
      if (best) {
        setActiveTicker(best.ticker)
        setSummary(paperToSummary(best))
        // Fetch its equity curve
        const eq = await api.paperEquity(best.ticker).catch(() => null)
        if (eq && eq.length) setPnlData(eq)
      }
    } catch { /* non-fatal */ }
  }

  async function loadFallback() {
    // Only use generic /summary + /pnl when no paper engine is active
    if (activeTicker) return
    api.summary()  .then(s => { if (s) setSummary(s) })           .catch(() => {})
    api.pnl()      .then(d => { if (d?.length) setPnlData(d) })   .catch(() => {})
  }

  async function loadPositions() {
    api.positions().then(r => setPositions(r?.positions ?? [])).catch(() => {})
  }

  async function loadHistory(symbol: string) {
    setHistLoad(true)
    try {
      const hist = await api.signals(symbol, 60)
      setAllSignals(prev => mergeSignals(prev, hist ?? []))
    } catch { /* Redis may be offline */ }
    finally { setHistLoad(false) }
  }

  // ── Effects ───────────────────────────────────────────────────────────────
  useEffect(() => {
    loadPaperEngines()
    loadPositions()
    const id = setInterval(() => { loadPaperEngines(); loadFallback(); loadPositions() }, POLL_MS)
    return () => clearInterval(id)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { loadFallback() }, [activeTicker]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { loadHistory(histSymbol) }, [histSymbol])

  // Merge incoming WebSocket signals
  useEffect(() => {
    if (wsSignals.length > prevWsLen.current) {
      const incoming = wsSignals.slice(0, wsSignals.length - prevWsLen.current)
      setAllSignals(prev => mergeSignals(incoming, prev))
    }
    prevWsLen.current = wsSignals.length
  }, [wsSignals])

  const runningEngines = paperEngines.filter(e => e.running)

  return (
    <div className="space-y-4">

      {/* ── Paper engine live banner ── */}
      {runningEngines.length > 0 && (
        <div className="flex items-center gap-3 flex-wrap">
          {runningEngines.map(e => (
            <button
              key={e.ticker}
              onClick={() => {
                setActiveTicker(e.ticker)
                setSummary(paperToSummary(e))
                api.paperEquity(e.ticker).then(eq => { if (eq?.length) setPnlData(eq) }).catch(() => {})
              }}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-semibold
                          transition-colors cursor-pointer
                          ${activeTicker === e.ticker
                            ? 'bg-blue-900/60 border-blue-600 text-blue-300'
                            : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500'}`}
            >
              <Radio size={11} className="text-green-400 animate-pulse" />
              {e.ticker}
              <span className="text-slate-500 font-normal">
                {e.bar_count} bars · {e.portfolio.total_return >= 0 ? '+' : ''}
                {(e.portfolio.total_return * 100).toFixed(2)}%
              </span>
            </button>
          ))}
          {activeTicker && (
            <span className="text-xs text-slate-600">
              showing live data from <span className="text-slate-400">{activeTicker}</span>
            </span>
          )}
        </div>
      )}

      {/* ── Metrics row ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricsCards summary={summary} />
      </div>

      {/* ── Equity + Drawdown ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title={activeTicker ? `Equity — ${activeTicker} (paper)` : 'Equity Curve'}>
          <PnLChart data={pnlData} />
        </Panel>
        <Panel title="Drawdown">
          <DrawdownChart data={pnlData} />
        </Panel>
      </div>

      {/* ── Live signals ── */}
      <Panel title={`Live Signals${allSignals.length > 0 ? ` · ${allSignals.length}` : ''}`}>
        <div className="flex items-center gap-2 mb-3">
          <input
            className="inp w-28 text-xs py-1"
            value={histInput}
            onChange={e => setHistInput(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && setHistSymbol(histInput)}
            placeholder="AAPL"
          />
          <button
            onClick={() => setHistSymbol(histInput)}
            disabled={histLoad}
            className="flex items-center gap-1 px-2 py-1 rounded bg-slate-700 hover:bg-slate-600
                       text-slate-300 text-xs font-semibold transition-colors disabled:opacity-50"
          >
            <RefreshCw size={11} className={histLoad ? 'animate-spin' : ''} />
            Load history
          </button>
          <span className="text-xs text-slate-600">· new signals appear via WebSocket</span>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <SignalChart signals={allSignals} />
          <SignalFeed  signals={allSignals} />
        </div>
      </Panel>

      {/* ── Positions ── */}
      <Panel title="Open Positions">
        <PositionsTable positions={positions} />
      </Panel>

    </div>
  )
}
