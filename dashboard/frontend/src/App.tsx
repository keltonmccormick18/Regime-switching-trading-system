import { useState } from 'react'
import {
  LayoutDashboard, Cpu, BarChart2, BrainCircuit,
  Server, Wifi, WifiOff, FlaskConical,
} from 'lucide-react'
import { useSignalStream } from './hooks/useSignalStream'
import { DashboardPage }   from './pages/DashboardPage'
import { PredictPage }     from './pages/PredictPage'
import { BacktestPage }    from './pages/BacktestPage'
import { BenchmarkPage }   from './pages/BenchmarkPage'
import { TrainPage }       from './pages/TrainPage'
import { SystemPage }      from './pages/SystemPage'
import { ActiveOpsProvider } from './contexts/ActiveOpsContext'
import { ActiveOpsBar }      from './components/ActiveOpsBar'

type Tab = 'dashboard' | 'predict' | 'backtest' | 'benchmark' | 'train' | 'system'

const NAV: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'dashboard',  label: 'Dashboard',    icon: <LayoutDashboard size={15} /> },
  { id: 'predict',    label: 'Predict',      icon: <Cpu             size={15} /> },
  { id: 'backtest',   label: 'Backtest',     icon: <BarChart2       size={15} /> },
  { id: 'benchmark',  label: 'Benchmark',    icon: <FlaskConical    size={15} /> },
  { id: 'train',      label: 'Train',        icon: <BrainCircuit    size={15} /> },
  { id: 'system',     label: 'System',       icon: <Server          size={15} /> },
]

function AppShell() {
  const [tab, setTab]           = useState<Tab>('dashboard')
  const { connected, signals }  = useSignalStream()

  return (
    <div className="min-h-screen bg-slate-900 flex flex-col font-mono text-slate-100">

      {/* ── Top bar ─────────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-slate-800 shrink-0">
        <div>
          <span className="text-sm font-bold tracking-tight text-slate-100">
            QUANT TRADING SYSTEM
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          {connected
            ? <Wifi   size={13} className="text-green-400" />
            : <WifiOff size={13} className="text-red-500"  />}
          <span>{connected ? 'LIVE' : 'DISCONNECTED'}</span>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">

        {/* ── Sidebar ───────────────────────────────────────────────────────── */}
        <nav className="w-44 shrink-0 border-r border-slate-800 py-3 flex flex-col px-2">
          <div className="flex flex-col gap-0.5">
            {NAV.map(({ id, label, icon }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={`flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-xs font-semibold text-left transition-colors
                  ${tab === id
                    ? 'bg-blue-700 text-white'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                  }`}
              >
                {icon}
                {label}
              </button>
            ))}
          </div>
          <ActiveOpsBar onNavigate={setTab} />
        </nav>

        {/* ── Page content ─────────────────────────────────────────────────── */}
        <main className="flex-1 overflow-y-auto p-5">
          {tab === 'dashboard'  && <DashboardPage />}
          {tab === 'predict'    && <PredictPage   />}
          {tab === 'backtest'   && <BacktestPage  />}
          {tab === 'benchmark'  && <BenchmarkPage />}
          {tab === 'train'      && <TrainPage     />}
          {tab === 'system'     && <SystemPage    />}
        </main>

      </div>
    </div>
  )
}

export default function App() {
  return (
    <ActiveOpsProvider>
      <AppShell />
    </ActiveOpsProvider>
  )
}
