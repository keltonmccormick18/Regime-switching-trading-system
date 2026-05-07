import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api } from '../api/client'
import type { TrainResponse } from '../types'

interface ActiveOpsCtx {
  jobs:      TrainResponse[]
  addJob:    (job: TrainResponse) => void
  cancelJob: (jobId: string) => Promise<void>
}

const Ctx = createContext<ActiveOpsCtx | null>(null)

export function useActiveOps() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useActiveOps must be inside ActiveOpsProvider')
  return ctx
}

export function ActiveOpsProvider({ children }: { children: React.ReactNode }) {
  const [jobs, setJobs] = useState<TrainResponse[]>([])

  // On mount: restore any in-progress training jobs from the server
  useEffect(() => {
    api.trainList()
      .then(({ jobs: list }) => {
        const active = list.filter(j => j.status === 'queued' || j.status === 'running')
        if (active.length > 0) setJobs(active)
      })
      .catch(() => {})
  }, [])

  // Poll active training jobs every 3 s
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

  return (
    <Ctx.Provider value={{ jobs, addJob, cancelJob }}>
      {children}
    </Ctx.Provider>
  )
}
