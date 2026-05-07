import React from 'react'
import type { ReactNode } from 'react'
import { Loader2 } from 'lucide-react'

// ── Colour helpers ────────────────────────────────────────────────────────────

export function regimeColor(regime: string): string {
  if (!regime) return 'slate'
  const r = regime.toLowerCase()
  if (r.includes('bull'))      return 'green'
  if (r.includes('bear'))      return 'red'
  if (r.includes('low_vol'))   return 'blue'
  if (r.includes('high_vol'))  return 'yellow'
  if (r.includes('crisis'))    return 'red'
  return 'slate'
}

// ── Panel ─────────────────────────────────────────────────────────────────────

export function Panel({
  title, icon, children,
}: {
  title: string
  icon?: ReactNode
  children: ReactNode
}) {
  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
      <h2 className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">
        {icon}
        {title}
      </h2>
      {children}
    </div>
  )
}

// ── Field ─────────────────────────────────────────────────────────────────────

export function Field({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-slate-500 font-medium">{label}</label>
      {children}
    </div>
  )
}

// ── Btn ───────────────────────────────────────────────────────────────────────

export function Btn({
  children, onClick, loading = false, compact = false, variant = 'primary', disabled,
}: {
  children: ReactNode
  onClick?: () => void
  loading?: boolean
  compact?: boolean
  variant?: 'primary' | 'danger' | 'ghost'
  disabled?: boolean
}) {
  const base =
    'inline-flex items-center justify-center gap-1.5 font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed'
  const size = compact ? 'px-3 py-1.5 text-xs' : 'px-4 py-2 text-sm'
  const color =
    variant === 'danger'
      ? 'bg-red-800 hover:bg-red-700 text-red-200'
      : variant === 'ghost'
      ? 'bg-slate-700 hover:bg-slate-600 text-slate-300'
      : 'bg-blue-700 hover:bg-blue-600 text-white'

  return (
    <button
      className={`${base} ${size} ${color}`}
      onClick={onClick}
      disabled={loading || disabled}
    >
      {loading && <Loader2 size={12} className="animate-spin" />}
      {children}
    </button>
  )
}

// ── Badge ─────────────────────────────────────────────────────────────────────

const BADGE_COLORS: Record<string, string> = {
  green:  'bg-green-900  text-green-300',
  red:    'bg-red-900    text-red-300',
  blue:   'bg-blue-900   text-blue-300',
  yellow: 'bg-yellow-900 text-yellow-300',
  slate:  'bg-slate-700  text-slate-300',
  purple: 'bg-purple-900 text-purple-300',
}

export function Badge({
  children, color = 'slate', small = false, large = false,
}: {
  children: ReactNode
  color?: string
  small?: boolean
  large?: boolean
}) {
  const cls = BADGE_COLORS[color] ?? BADGE_COLORS.slate
  const size = large ? 'text-sm px-3 py-1' : small ? 'text-xs px-1.5 py-0.5' : 'text-xs px-2 py-0.5'
  return (
    <span className={`inline-flex items-center rounded-full font-semibold uppercase tracking-wide ${cls} ${size}`}>
      {children}
    </span>
  )
}

// ── ErrorBox ──────────────────────────────────────────────────────────────────

export function ErrorBox({ msg }: { msg: string }) {
  return (
    <div className="mt-3 flex items-start gap-2 rounded-lg bg-red-950 border border-red-800 px-3 py-2 text-xs text-red-300 font-mono break-all">
      <span className="shrink-0 font-bold text-red-400">Error:</span>
      {msg}
    </div>
  )
}

// ── Stat ──────────────────────────────────────────────────────────────────────

export function Stat({
  label, value, color = 'text-slate-100',
}: {
  label: string
  value: React.ReactNode
  color?: string
}) {
  return (
    <div className="bg-slate-700 rounded-lg p-3 flex flex-col gap-1">
      <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
      <span className={`text-lg font-mono font-bold ${color}`}>{value}</span>
    </div>
  )
}
