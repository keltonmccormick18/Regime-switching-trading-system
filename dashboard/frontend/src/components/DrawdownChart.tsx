import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer,
  ReferenceLine, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { PnLPoint } from '../types'

interface Props { data: PnLPoint[] }

const fmtDate = (ts: string) =>
  new Date(ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })

const TOOLTIP = {
  backgroundColor: '#1e293b',
  border: '1px solid #334155',
  borderRadius: 6,
  color: '#f1f5f9',
  fontSize: 12,
}

export function DrawdownChart({ data }: Props) {
  const step    = Math.max(1, Math.floor(data.length / 300))
  const thinned = data.filter((_, i) => i % step === 0)

  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={thinned} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
        <defs>
          <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.05} />
            <stop offset="95%" stopColor="#ef4444" stopOpacity={0.35} />
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
          tickFormatter={v => `${(v * 100).toFixed(0)}%`}
          tick={{ fill: '#475569', fontSize: 10 }}
          tickLine={false} axisLine={false}
          width={44}
          domain={['auto', 0]}
        />
        <ReferenceLine y={0} stroke="#334155" strokeDasharray="4 4" />
        <Tooltip
          labelFormatter={fmtDate}
          formatter={(v: number) => [`${(v * 100).toFixed(2)}%`, 'Drawdown']}
          contentStyle={TOOLTIP}
        />
        <Area
          type="monotone"
          dataKey="drawdown"
          stroke="#ef4444"
          strokeWidth={1.5}
          fill="url(#ddGrad)"
          dot={false}
          activeDot={{ r: 3, fill: '#ef4444' }}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
