import type { Position } from '../types'

interface Props { positions: Position[] }

function SideBadge({ side }: { side: string }) {
  const isLong = side === 'long'
  const cls = isLong
    ? 'bg-green-900/60 text-green-300 border border-green-800'
    : 'bg-red-900/60   text-red-300   border border-red-800'
  return (
    <span className={`text-xs px-2 py-0.5 rounded font-semibold ${cls}`}>
      {side.toUpperCase()}
    </span>
  )
}

function PnLCell({ pnl, pct }: { pnl: number; pct: number }) {
  const pos = pnl >= 0
  return (
    <span className={`font-mono text-sm ${pos ? 'text-green-400' : 'text-red-400'}`}>
      {pos ? '+' : ''}${Math.abs(pnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
      <span className="text-xs opacity-60 ml-1">
        ({pos ? '+' : ''}{(pct * 100).toFixed(2)}%)
      </span>
    </span>
  )
}

export function PositionsTable({ positions }: Props) {
  if (!positions || positions.length === 0) {
    return (
      <p className="text-slate-600 text-sm py-6 text-center">No open positions</p>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700 text-left">
            {['Symbol', 'Side', 'Qty', 'Entry', 'Current', 'Unrealized P&L'].map(h => (
              <th key={h} className="pb-2 pr-4 text-xs text-slate-500 uppercase tracking-widest font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {positions.map((p, i) => (
            <tr
              key={i}
              className="border-b border-slate-700/40 hover:bg-slate-700/25 transition-colors"
            >
              <td className="py-3 pr-4 font-semibold text-slate-100">{p.symbol}</td>
              <td className="py-3 pr-4"><SideBadge side={p.side} /></td>
              <td className="py-3 pr-4 font-mono text-slate-300 text-right">{p.quantity}</td>
              <td className="py-3 pr-4 font-mono text-slate-400 text-right">
                ${p.avg_entry.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </td>
              <td className="py-3 pr-4 font-mono text-slate-300 text-right">
                ${p.current_price.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </td>
              <td className="py-3 text-right">
                <PnLCell pnl={p.unrealized_pnl} pct={p.unrealized_pnl_pct} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
