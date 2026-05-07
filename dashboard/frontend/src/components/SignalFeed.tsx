import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import type { Signal } from '../types'

function relTime(ts: string): string {
  const diff = (Date.now() - new Date(ts).getTime()) / 1000
  if (diff < 5)    return 'just now'
  if (diff < 60)   return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
}

function DirectionChip({ signal }: { signal: number }) {
  if (signal === 1)
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold
                       bg-green-900/70 text-green-300 border border-green-800">
        <TrendingUp size={10} /> LONG
      </span>
    )
  if (signal === -1)
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold
                       bg-red-900/70 text-red-300 border border-red-800">
        <TrendingDown size={10} /> SHORT
      </span>
    )
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold
                     bg-slate-700 text-slate-400 border border-slate-600">
      <Minus size={10} /> FLAT
    </span>
  )
}

function ConfBar({ pct }: { pct: number }) {
  const w = Math.round(pct * 100)
  const color = pct >= 0.7 ? 'bg-green-500' : pct >= 0.45 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${w}%` }} />
      </div>
      <span className="text-xs text-slate-500 tabular-nums">{w}%</span>
    </div>
  )
}

interface Props {
  signals: Signal[]
  maxRows?: number
}

export function SignalFeed({ signals, maxRows = 40 }: Props) {
  // Tick every 15s so relative timestamps stay fresh
  const [, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 15_000)
    return () => clearInterval(id)
  }, [])

  if (signals.length === 0) {
    return (
      <div className="h-48 flex flex-col items-center justify-center gap-2 text-slate-600">
        <span className="text-2xl">📡</span>
        <span className="text-sm">Waiting for signals…</span>
        <span className="text-xs">Run a prediction to see live data</span>
      </div>
    )
  }

  const rows = signals.slice(0, maxRows)

  return (
    <div className="divide-y divide-slate-700/50 overflow-y-auto max-h-72">
      {rows.map((s, i) => (
        <div
          key={`${s.symbol}-${s.timestamp}-${i}`}
          className={`flex items-center gap-3 px-1 py-2.5 transition-colors
            ${i === 0 ? 'bg-slate-700/30' : 'hover:bg-slate-700/20'}`}
        >
          {/* New signal pulse dot */}
          {i === 0 && (
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse shrink-0" />
          )}
          {i !== 0 && <span className="w-1.5 shrink-0" />}

          {/* Symbol */}
          <span className="font-mono font-bold text-slate-200 text-sm w-12 shrink-0">
            {s.symbol}
          </span>

          {/* Direction */}
          <div className="w-20 shrink-0">
            <DirectionChip signal={s.signal} />
          </div>

          {/* Confidence bar */}
          <div className="flex-1 hidden sm:block">
            <ConfBar pct={s.confidence} />
          </div>

          {/* Regime */}
          <span className="text-xs text-slate-500 w-24 truncate hidden md:block">
            {s.regime.replace(/_/g, ' ')}
          </span>

          {/* Time */}
          <span className="text-xs text-slate-600 tabular-nums shrink-0 w-16 text-right">
            {relTime(s.timestamp)}
          </span>
        </div>
      ))}
    </div>
  )
}
