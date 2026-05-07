import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer,
  Tooltip, XAxis, YAxis, ReferenceLine,
} from 'recharts'
import type { PnLPoint } from '../types'

interface Props { data: PnLPoint[] }

const fmtDate = (ts: string) =>
  new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })

const fmtDollar = (v: number) => {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`
  if (v >= 1_000)     return `$${(v / 1_000).toFixed(1)}k`
  return `$${v.toFixed(0)}`
}

const TOOLTIP = {
  backgroundColor: '#1e293b',
  border: '1px solid #334155',
  borderRadius: 6,
  color: '#f1f5f9',
  fontSize: 12,
}

export function PnLChart({ data }: Props) {
  if (!data || data.length === 0) {
    return (
      <div className="h-[220px] flex items-center justify-center text-slate-600 text-sm">
        No equity data yet
      </div>
    )
  }

  // Thin to max 300 points for render performance
  const step    = Math.max(1, Math.floor(data.length / 300))
  const thinned = data.filter((_, i) => i % step === 0)

  const first = thinned[0]?.equity ?? 0
  const last  = thinned[thinned.length - 1]?.equity ?? 0
  const pos   = last >= first

  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={thinned} margin={{ top: 4, right: 4, left: 8, bottom: 0 }}>
        <defs>
          <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={pos ? '#22c55e' : '#ef4444'} stopOpacity={0.28} />
            <stop offset="95%" stopColor={pos ? '#22c55e' : '#ef4444'} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="timestamp"
          tickFormatter={fmtDate}
          tick={{ fill: '#475569', fontSize: 10 }}
          tickLine={false} axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tickFormatter={fmtDollar}
          tick={{ fill: '#475569', fontSize: 10 }}
          tickLine={false} axisLine={false}
          domain={['auto', 'auto']}
          width={56}
        />
        {/* Starting equity reference line */}
        {first > 0 && (
          <ReferenceLine y={first} stroke="#334155" strokeDasharray="4 4" />
        )}
        <Tooltip
          labelFormatter={fmtDate}
          formatter={(v: number, name: string) => {
            if (name === 'equity') {
              const ret = first > 0 ? ((v - first) / first * 100) : 0
              return [`${fmtDollar(v)}  (${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%)`, 'Equity']
            }
            return [String(v), name]
          }}
          contentStyle={TOOLTIP}
        />
        <Area
          type="monotone"
          dataKey="equity"
          stroke={pos ? '#22c55e' : '#ef4444'}
          strokeWidth={2}
          fill="url(#pnlGrad)"
          dot={data.length === 1 ? { r: 4, fill: pos ? '#22c55e' : '#ef4444' } : false}
          activeDot={{ r: 3 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
