import { BrainCircuit, Clock, Loader, PlayCircle, StopCircle, XCircle } from 'lucide-react'
import { useActiveOps } from '../contexts/ActiveOpsContext'

type Tab = 'dashboard' | 'predict' | 'backtest' | 'train' | 'paper' | 'system'

export function ActiveOpsBar({ onNavigate }: { onNavigate: (tab: Tab) => void }) {
  const { jobs, engines, cancelJob, stopEngine } = useActiveOps()

  const activeJobs    = jobs.filter(j => j.status === 'queued' || j.status === 'running')
  const runningEngines = engines.filter(e => e.running)

  if (activeJobs.length === 0 && runningEngines.length === 0) return null

  return (
    <div className="mt-auto border-t border-slate-800 pt-2 pb-2 px-2 space-y-1">
      <div className="text-xs text-slate-600 uppercase tracking-widest px-1 mb-1.5">Active</div>

      {activeJobs.map(job => (
        <div
          key={job.job_id}
          className="group flex items-center gap-1.5 px-2 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-750"
        >
          <button
            onClick={() => onNavigate('train')}
            className="flex items-center gap-1.5 flex-1 min-w-0 text-left"
            title={`Training ${job.ticker} — ${job.status}`}
          >
            {job.status === 'running'
              ? <Loader size={11} className="text-blue-400 animate-spin shrink-0" />
              : <Clock  size={11} className="text-slate-500 shrink-0" />
            }
            <span className="text-xs font-mono text-slate-300 truncate">{job.ticker}</span>
            <BrainCircuit size={10} className="text-slate-600 shrink-0" />
          </button>
          <button
            onClick={() => cancelJob(job.job_id)}
            className="opacity-0 group-hover:opacity-100 p-0.5 text-slate-500 hover:text-red-400 transition-all"
            title="Cancel training"
          >
            <XCircle size={12} />
          </button>
        </div>
      ))}

      {runningEngines.map(engine => (
        <div
          key={engine.ticker}
          className="group flex items-center gap-1.5 px-2 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-750"
        >
          <button
            onClick={() => onNavigate('paper')}
            className="flex items-center gap-1.5 flex-1 min-w-0 text-left"
            title={`Paper trading ${engine.ticker}`}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse shrink-0" />
            <span className="text-xs font-mono text-slate-300 truncate">{engine.ticker}</span>
            <PlayCircle size={10} className="text-slate-600 shrink-0" />
          </button>
          <button
            onClick={() => stopEngine(engine.ticker)}
            className="opacity-0 group-hover:opacity-100 p-0.5 text-slate-500 hover:text-red-400 transition-all"
            title="Stop paper trading"
          >
            <StopCircle size={12} />
          </button>
        </div>
      ))}
    </div>
  )
}
