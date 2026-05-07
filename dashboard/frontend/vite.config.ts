import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// __dirname is not available in ES-module configs; derive it from import.meta.url
const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Cert files live at <repo-root>/certs/  (two levels up from dashboard/frontend/)
const repoRoot = path.resolve(__dirname, '../..')
const keyPath  = path.join(repoRoot, 'certs', 'server.key')
const crtPath  = path.join(repoRoot, 'certs', 'server.crt')
// Set VITE_HTTPS=1 in the environment to enable HTTPS with local certs
const useHttps  = process.env.VITE_HTTPS === '1'
const haveCerts = fs.existsSync(keyPath) && fs.existsSync(crtPath)

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    https: useHttps && haveCerts
      ? { key: fs.readFileSync(keyPath), cert: fs.readFileSync(crtPath) }
      : undefined,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (p: string) => p.replace(/^\/api/, ''),
        // Return a clean 503 instead of crashing when the API is not running
        onError(err, _req, res) {
          if ('writeHead' in res) {
            res.writeHead(503, { 'Content-Type': 'application/json' })
            res.end(JSON.stringify({ detail: 'API server is not running' }))
          }
        },
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
        // Silence ECONNREFUSED noise when the API is not running
        onError() {},
      },
    },
  },
})
