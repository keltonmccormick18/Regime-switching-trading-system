import { useEffect, useRef, useState } from 'react'
import type { Signal } from '../types'

export function useSignalStream(maxSignals = 120) {
  const [signals,   setSignals]   = useState<Signal[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>

    function connect() {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws    = new WebSocket(`${proto}//${window.location.host}/ws/signals`)
      wsRef.current = ws

      ws.onopen    = () => setConnected(true)
      ws.onclose   = () => { setConnected(false); timer = setTimeout(connect, 3_000) }
      ws.onerror   = () => ws.close()
      ws.onmessage = (e) => {
        try {
          const sig = JSON.parse(e.data) as Signal
          setSignals(prev => [sig, ...prev].slice(0, maxSignals))
        } catch { /* ignore malformed messages */ }
      }
    }

    connect()
    return () => { clearTimeout(timer); wsRef.current?.close() }
  }, [maxSignals])

  return { signals, connected }
}
