import {
  Bar, Cell, ComposedChart, CartesianGrid, Line,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { Signal } from '../types'

interface Props { signals: Signal[] }

function sigColor(s: number) {
  if (s ===  1) return '#22c55e'
  if (s === -1) return '#ef4444'
  return '#475569'
}
function sigLabel(s: number) {
  if (s ===  1) return 'LONG'
  if (s === -1) return 'SHORT'
  return 'FLAT'
}

const fmtTime = (ts: string) =>
  new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })

const TOOLTIP = {
  backgroundColor: '#1e293b',
  border: '1px solid #334155',
  borderRadius: 6,
  color: '#f1f5f9',
  fontSize: 11,
}

export function SignalChart({ signals }: Props) {
  // Oldest → newest for left-to-right time axis
  const data = [...signals]
    .reverse()
    .slice(-50)
    .map(s => ({
      time:       fmtTime(s.timestamp),
      signal:     s.signal,
      confidence: s.confidence,
      symbol:     s.symbol,
    }))

  if (data.length === 0) {
    return (
      <div className="h-[200px] flex items-center justify-center text-slate-600 text-sm">
        Waiting for signals…
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />

        {/* Signal direction axis  (-1 / 0 / 1) */}
        <YAxis
          yAxisId="sig"
          domain={[-1.6, 1.6]}
          ticks={[-1, 0, 1]}
          tickFormatter={v => v === 1 ? 'L' : v === -1 ? 'S' : '—'}
          tick={{ fill: '#475569', fontSize: 10 }}
          tickLine={false} axisLine={false}
          width={18}
        />
        {/* Confidence axis (0 – 1) */}
        <YAxis
          yAxisId="conf"
          orientation="right"
          domain={[0, 1]}
          tickFormatter={v => `${(v * 100).toFixed(0)}%`}
          tick={{ fill: '#475569', fontSize: 10 }}
          tickLine={false} axisLine={false}
          width={36}
        />
        <XAxis
          dataKey="time"
          tick={{ fill: '#475569', fontSize: 9 }}
          tickLine={false} axisLine={false}
          interval={Math.max(1, Math.floor(data.length / 6))}
        />

        <ReferenceLine yAxisId="sig" y={0} stroke="#334155" strokeDasharray="4 4" />

        <Tooltip
          contentStyle={TOOLTIP}
          formatter={(v: unknown, name: string) => {
            if (name === 'signal')     return [sigLabel(v as number), 'Signal']
            if (name === 'confidence') return [`${((v as number) * 100).toFixed(0)}%`, 'Confidence']
            return [String(v), name]
          }}
        />

        {/* Signal bars — colored by direction */}
        <Bar yAxisId="sig" dataKey="signal" maxBarSize={14} radius={[2, 2, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={sigColor(d.signal)} fillOpacity={0.85} />
          ))}
        </Bar>

        {/* Confidence line */}
        <Line
          yAxisId="conf"
          type="monotone"
          dataKey="confidence"
          stroke="#3b82f6"
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
