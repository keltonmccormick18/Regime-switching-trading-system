import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api } from '../api/client'
import type { PaperStatusResponse, TrainResponse } from '../types'

interface ActiveOpsCtx {
  jobs:         TrainResponse[]
  engines:      PaperStatusResponse[]
  addJob:       (job: TrainResponse) => void
  cancelJob:    (jobId: string) => Promise<void>
  addEngine:    (engine: PaperStatusResponse) => void
  stopEngine:   (ticker: string) => Promise<void>
  refreshEngine:(ticker: string) => Promise<void>
}

const Ctx = createContext<ActiveOpsCtx | null>(null)

export function useActiveOps() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useActiveOps must be inside ActiveOpsProvider')
  return ctx
}

export function ActiveOpsProvider({ children }: { children: React.ReactNode }) {
  const [jobs,    setJobs]    = useState<TrainResponse[]>([])
  const [engines, setEngines] = useState<PaperStatusResponse[]>([])

  // On mount: restore any paper engines already running on the server
  useEffect(() => {
    api.paperList()
      .then(({ engines: list }) => {
        const running = list.filter(e => e.running)
        if (running.length === 0) return
        Promise.all(running.map(e => api.paperStatus(e.ticker).catch(() => null)))
          .then(statuses => {
            const valid = statuses.filter(Boolean) as PaperStatusResponse[]
            if (valid.length > 0) setEngines(valid)
          })
      })
      .catch(() => {})

    // On mount: restore in-progress training jobs from server
    api.trainList()
      .then(({ jobs: list }) => {
        const active = list.filter(j => j.status === 'queued' || j.status === 'running')
        if (active.length > 0) setJobs(active)
      })
      .catch(() => {})
  }, [])

  // Poll active training jobs every 3s
  useEffect(() => {
    const active = jobs.filter(j => j.status === 'queued' || j.status === 'running')
    if (active.length === 0) return
    const id = setInterval(() => {
      active.forEach(j => {
        api.trainStatus(j.job_id)
          .then(updated => setJobs(prev => prev.map(x => x.job_id === updated.job_id ? updated : x)))
          .catch(() => {})
      })
    }, 3_000)
    return () => clearInterval(id)
  }, [jobs])

  // Poll running paper engines every 30s
  useEffect(() => {
    const running = engines.filter(e => e.running)
    if (running.length === 0) return
    const id = setInterval(() => {
      running.forEach(e => {
        api.paperStatus(e.ticker)
          .then(updated => setEngines(prev => prev.map(x => x.ticker === updated.ticker ? updated : x)))
          .catch(() => {})
      })
    }, 30_000)
    return () => clearInterval(id)
  }, [engines])

  const addJob = useCallback((job: TrainResponse) => {
    setJobs(prev => [job, ...prev])
  }, [])

  const cancelJob = useCallback(async (jobId: string) => {
    await api.trainCancel(jobId).catch(() => {})
    setJobs(prev => prev.map(j =>
      j.job_id === jobId
        ? { ...j, status: 'error' as const, message: 'Cancelled by user', finished_at: new Date().toISOString() }
        : j
    ))
  }, [])

  const addEngine = useCallback((engine: PaperStatusResponse) => {
    setEngines(prev => [engine, ...prev.filter(e => e.ticker !== engine.ticker)])
  }, [])

  const stopEngine = useCallback(async (ticker: string) => {
    await api.paperStop(ticker).catch(() => {})
    const updated = await api.paperStatus(ticker).catch(() => null)
    if (updated) setEngines(prev => prev.map(e => e.ticker === ticker ? updated : e))
  }, [])

  const refreshEngine = useCallback(async (ticker: string) => {
    const updated = await api.paperStatus(ticker).catch(() => null)
    if (updated) setEngines(prev => prev.map(e => e.ticker === ticker ? updated : e))
  }, [])

  return (
    <Ctx.Provider value={{ jobs, engines, addJob, cancelJob, addEngine, stopEngine, refreshEngine }}>
      {children}
    </Ctx.Provider>
  )
}
