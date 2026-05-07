import { useState } from 'react'
import { BrainCircuit, CheckCircle, XCircle, Loader, Clock } from 'lucide-react'
import { api } from '../api/client'
import type { TrainResponse } from '../types'
import { Field, Btn, Panel, Badge, ErrorBox, regimeColor } from '../components/ui'
import { useActiveOps } from '../contexts/ActiveOpsContext'

const DEFAULTS = {
  ticker: 'AAPL', start: '2010-01-01', use_tda: false,
  lookback: 64, horizon: 16, epochs: 30,
  save_artifact: true, artifact_name: '',
}

function JobCard({ job, onCancel }: { job: TrainResponse; onCancel: (id: string) => void }) {
  const icon = {
    queued:  <Clock size={14} className="text-slate-400" />,
    running: <Loader size={14} className="text-blue-400 animate-spin" />,
    done:    <CheckCircle size={14} className="text-green-400" />,
    error:   <XCircle size={14} className="text-red-400" />,
  }[job.status]

  const borderColor = {
    queued: 'border-slate-600', running: 'border-blue-600',
    done: 'border-green-600',   error: 'border-red-600',
  }[job.status]

  const elapsed = job.finished_at
    ? `${((new Date(job.finished_at).getTime() - new Date(job.started_at).getTime()) / 1000).toFixed(0)}s`
    : job.status === 'running' ? 'running…' : '—'

  const canCancel = job.status === 'queued' || job.status === 'running'

  return (
    <div className={`bg-slate-800 rounded-xl border ${borderColor} p-4 space-y-3`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon}
          <span className="font-mono font-bold text-slate-100">{job.ticker}</span>
          <span className="text-xs text-slate-500">{job.job_id.slice(0, 8)}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-xs font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full ${
            job.status === 'done'    ? 'bg-green-900 text-green-300' :
            job.status === 'error'   ? 'bg-red-900 text-red-300' :
            job.status === 'running' ? 'bg-blue-900 text-blue-300' :
            'bg-slate-700 text-slate-400'
          }`}>{job.status === 'error' && job.message === 'Cancelled by user' ? 'cancelled' : job.status}</span>
          {canCancel && (
            <button
              onClick={() => onCancel(job.job_id)}
              className="text-xs px-2 py-0.5 rounded-full bg-slate-700 hover:bg-red-900 text-slate-400 hover:text-red-300 transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-xs">
        <div><span className="text-slate-500">Elapsed</span><br/>
          <span className="text-slate-200 font-mono">{elapsed}</span></div>
        {job.regime && (
          <div><span className="text-slate-500">Regime</span><br/>
            <Badge color={regimeColor(job.regime)} small>
              {job.regime.replace(/_/g,' ')}
            </Badge>
          </div>
        )}
        {job.model_used && (
          <div><span className="text-slate-500">Model</span><br/>
            <span className="text-slate-200 font-mono">{job.model_used}</span></div>
        )}
      </div>

      {job.artifact_path && (
        <div className="text-xs">
          <span className="text-slate-500">Artifact → </span>
          <span className="text-blue-400 font-mono break-all">{job.artifact_path}</span>
        </div>
      )}
      {job.message && job.message !== 'Cancelled by user' && (
        <div className="text-xs text-red-400 font-mono bg-red-950 rounded p-2 break-all">
          {job.message}
        </div>
      )}
    </div>
  )
}

export function TrainPage() {
  const [form, setForm]       = useState(DEFAULTS)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const { jobs, addJob, cancelJob } = useActiveOps()

  const set = (k: string, v: unknown) => setForm(f => ({ ...f, [k]: v }))

  async function run() {
    setLoading(true); setError(null)
    try {
      const job = await api.train({
        ticker:        form.ticker.toUpperCase(),
        start:         form.start,
        use_tda:       form.use_tda,
        lookback:      form.lookback,
        horizon:       form.horizon,
        epochs:        form.epochs,
        save_artifact: form.save_artifact,
        artifact_name: form.artifact_name || null,
      })
      addJob(job)
    } catch (e: unknown) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-5">
      <Panel title="Train Model" icon={<BrainCircuit size={14} />}>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
          <Field label="Ticker">
            <input className="inp" value={form.ticker}
              onChange={e => set('ticker', e.target.value.toUpperCase())} />
          </Field>
          <Field label="History Start">
            <input className="inp" type="date" value={form.start}
              onChange={e => set('start', e.target.value)} />
          </Field>
          <Field label="Epochs">
            <input className="inp" type="number" min={1} max={500} value={form.epochs}
              onChange={e => set('epochs', +e.target.value)} />
          </Field>
          <Field label="Lookback">
            <input className="inp" type="number" min={10} max={512} value={form.lookback}
              onChange={e => set('lookback', +e.target.value)} />
          </Field>
          <Field label="Horizon">
            <input className="inp" type="number" min={1} max={252} value={form.horizon}
              onChange={e => set('horizon', +e.target.value)} />
          </Field>
          <Field label="Artifact Name (optional)">
            <input className="inp" placeholder="auto-generated if blank"
              value={form.artifact_name} onChange={e => set('artifact_name', e.target.value)} />
          </Field>
          <Field label="Save artifact">
            <label className="flex items-center gap-2 mt-2 cursor-pointer">
              <input type="checkbox" className="w-4 h-4 accent-blue-500"
                checked={form.save_artifact} onChange={e => set('save_artifact', e.target.checked)} />
              <span className="text-sm text-slate-300">Save .pt file</span>
            </label>
          </Field>
          <Field label="Use TDA">
            <label className="flex items-center gap-2 mt-2 cursor-pointer">
              <input type="checkbox" className="w-4 h-4 accent-blue-500"
                checked={form.use_tda} onChange={e => set('use_tda', e.target.checked)} />
              <span className="text-sm text-slate-300">Enable TDA features</span>
            </label>
          </Field>
        </div>
        <p className="text-xs text-slate-500 mb-3">
          Training runs in the background. Status persists while navigating between pages.
        </p>
        <Btn loading={loading} onClick={run}>
          {loading ? 'Queuing job…' : 'Start Training'}
        </Btn>
        {error && <ErrorBox msg={error} />}
      </Panel>

      {jobs.length > 0 && (
        <div>
          <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">
            Training Jobs ({jobs.length})
          </h2>
          <div className="space-y-3">
            {jobs.map(j => (
              <JobCard key={j.job_id} job={j} onCancel={cancelJob} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
