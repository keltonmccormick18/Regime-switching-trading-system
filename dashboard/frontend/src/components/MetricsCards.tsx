import { Activity, ArrowDown, Target, TrendingUp } from 'lucide-react'
import type { MetricsSummary } from '../types'

interface CardProps {
  icon:  React.ElementType
  label: string
  value: string
  sub?:  string
  color: string
}

function Card({ icon: Icon, label, value, sub, color }: CardProps) {
  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500 uppercase tracking-widest">{label}</span>
        <Icon size={14} className={color} />
      </div>
      <p className={`text-2xl font-bold font-mono ${color}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500">{sub}</p>}
    </div>
  )
}

interface Props { summary: MetricsSummary | null }

export function MetricsCards({ summary: s }: Props) {
  const retColor    = !s ? 'text-slate-500'
    : s.total_return >= 0   ? 'text-green-400' : 'text-red-400'
  const sharpeColor = !s ? 'text-slate-500'
    : s.sharpe >= 1         ? 'text-green-400'
    : s.sharpe >= 0.5       ? 'text-yellow-400' : 'text-red-400'
  const winColor    = !s ? 'text-slate-500'
    : s.win_rate >= 0.5     ? 'text-green-400' : 'text-yellow-400'

  return (
    <>
      <Card
        icon={TrendingUp}
        label="Total Return"
        value={s ? `${s.total_return >= 0 ? '+' : ''}${(s.total_return * 100).toFixed(1)}%` : '—'}
        sub={s ? `${s.n_trades} trades` : undefined}
        color={retColor}
      />
      <Card
        icon={Activity}
        label="Sharpe Ratio"
        value={s ? s.sharpe.toFixed(2) : '—'}
        sub={s ? `annualised` : undefined}
        color={sharpeColor}
      />
      <Card
        icon={ArrowDown}
        label="Max Drawdown"
        value={s ? `${(s.max_drawdown * 100).toFixed(1)}%` : '—'}
        sub={s ? `peak-to-trough` : undefined}
        color="text-red-400"
      />
      <Card
        icon={Target}
        label="Win Rate"
        value={s ? `${(s.win_rate * 100).toFixed(1)}%` : '—'}
        sub={s ? s.model_used : undefined}
        color={winColor}
      />
    </>
  )
}
