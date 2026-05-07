import {
  BarChart2, BrainCircuit, Cpu, FlaskConical,
  TrendingUp, Shield, Layers, Activity,
} from 'lucide-react'
import { Panel } from '../components/ui'

// ─── Data ─────────────────────────────────────────────────────────────────

const REGIMES = [
  {
    name:  'LOW_VOL_BULL',
    model: 'TCNModel',
    color: 'text-green-400',
    bg:    'bg-green-900/20 border-green-800/40',
    desc:  'Low realised volatility, price above 200-day SMA. Trend-following conditions — the TCN captures multi-scale momentum patterns across short and long receptive fields.',
  },
  {
    name:  'LOW_VOL_BEAR',
    model: 'TCNLSTMModel',
    color: 'text-yellow-400',
    bg:    'bg-yellow-900/20 border-yellow-800/40',
    desc:  'Low volatility but price below 200-day SMA. The TCN-LSTM hybrid uses sequential memory to model slow mean-reversion dynamics typical of grinding bear markets.',
  },
  {
    name:  'HIGH_VOL_BULL',
    model: 'TFTModel',
    color: 'text-blue-400',
    bg:    'bg-blue-900/20 border-blue-800/40',
    desc:  "Elevated volatility, price above 200-day SMA. The Temporal Fusion Transformer's multi-head attention excels at non-linear, spike-driven price action in volatile bull runs.",
  },
  {
    name:  'HIGH_VOL_BEAR',
    model: 'OnlineModel',
    color: 'text-red-400',
    bg:    'bg-red-900/20 border-red-800/40',
    desc:  'High volatility, price below 200-day SMA — the most difficult regime. The River online learner adapts in real time to rapidly shifting distributions where static models degrade fastest.',
  },
]

const FEATURES = [
  { name: 'logret',        desc: 'Daily log return — primary price signal fed to all models.' },
  { name: 'vol_20',        desc: '20-bar realised volatility — used for regime detection and inverse-vol position sizing.' },
  { name: 'rsi_14',        desc: '14-period RSI — momentum oscillator, normalised to [0, 1].' },
  { name: 'sma_ratio_50',  desc: 'Close / 50-day SMA — short-term trend proxy.' },
  { name: 'sma_ratio_200', desc: 'Close / 200-day SMA — long-term trend proxy; below 1.0 blocks allocation entirely.' },
  { name: 'tda_l1',        desc: 'TDA persistence L1 — topological loop lifetime from sliding-window point cloud. Captures cycle structure invisible to linear indicators.' },
  { name: 'tda_l2',        desc: 'TDA persistence L2 — second-order topological feature for multi-scale shape analysis.' },
  { name: 'vix_pct',       desc: 'VIX expanding percentile [0,1] — cross-asset fear gauge (macro mode).' },
  { name: 'credit_spread', desc: 'log(HYG/LQD) — high-yield vs investment-grade spread; widens in risk-off (macro mode).' },
  { name: 'dxy_ret',       desc: 'UUP log return — USD strength signal (macro mode).' },
]

const PORTFOLIO_RULES = [
  {
    icon: <TrendingUp size={16} className="text-blue-400 shrink-0 mt-0.5" />,
    title: 'Inverse-volatility weighting',
    desc:  'Each ticker\'s target allocation is proportional to 1/vol_20. Low-volatility assets (e.g. TLT, GLD) naturally receive larger shares than high-volatility ones (e.g. NVDA), reducing uncompensated risk concentration.',
  },
  {
    icon: <Shield size={16} className="text-yellow-400 shrink-0 mt-0.5" />,
    title: '200-SMA trend filter',
    desc:  'Any ticker trading below its 200-day SMA is excluded from allocation entirely — its capital stays as cash. This prevented TLT allocation during the 2022–2023 rate-hike drawdown.',
  },
  {
    icon: <Activity size={16} className="text-red-400 shrink-0 mt-0.5" />,
    title: 'Regime-conditional allocation shift',
    desc:  'Tickers currently in a HIGH_VOL regime receive a 0.5× weight multiplier before re-normalisation. Capital flows toward lower-volatility assets during turbulent markets without any hard overrides.',
  },
  {
    icon: <Layers size={16} className="text-green-400 shrink-0 mt-0.5" />,
    title: 'Shared capital pool',
    desc:  'All tickers share one cash balance. Fills carry real cost consequences — a commission paid for NVDA reduces cash available for SPY. Inactive tickers\' allocations remain as cash rather than being redistributed.',
  },
]

// ─── Component ────────────────────────────────────────────────────────────

export function DashboardPage() {
  return (
    <div className="space-y-6 max-w-5xl">

      {/* ── Hero ── */}
      <div className="rounded-xl border border-slate-700/60 bg-slate-800/40 p-6">
        <h1 className="text-lg font-bold text-slate-100 tracking-tight mb-1">
          Regime-Switching Trading System
        </h1>
        <p className="text-sm text-slate-400 leading-relaxed">
          A research platform for developing and evaluating regime-conditioned predictive models
          on equity and crypto price series. The system detects the current market regime in real time,
          routes predictions to the model trained for that regime, and executes a shared-capital
          portfolio strategy with realistic broker simulation.
        </p>
      </div>

      {/* ── Regime → Model routing ── */}
      <Panel title="Regime Detection & Model Routing">
        <p className="text-xs text-slate-500 mb-4 leading-relaxed">
          Regime is determined at every bar using three signals: realised vol_20 expanding percentile
          (above median → HIGH_VOL), 200-day SMA ratio (above 1.0 → BULL), and optionally TDA L1
          persistence. Each regime routes to a dedicated model trained exclusively on windows from
          that regime.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {REGIMES.map(r => (
            <div key={r.name} className={`rounded-lg border p-4 ${r.bg}`}>
              <div className="flex items-center justify-between mb-2">
                <span className={`text-xs font-bold font-mono ${r.color}`}>{r.name}</span>
                <span className="text-xs text-slate-400 font-mono bg-slate-800/60 px-2 py-0.5 rounded">
                  {r.model}
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">{r.desc}</p>
            </div>
          ))}
        </div>
      </Panel>

      {/* ── Features ── */}
      <Panel title="Feature Pipeline">
        <p className="text-xs text-slate-500 mb-4 leading-relaxed">
          Features are computed bar-by-bar from raw OHLCV data. TDA features require{' '}
          <span className="text-slate-300 font-mono">use_tda=true</span>; macro features require{' '}
          <span className="text-slate-300 font-mono">use_macro=true</span>. All features are
          z-scored over a rolling window before being fed to the model.
        </p>
        <div className="divide-y divide-slate-700/50">
          {FEATURES.map(f => (
            <div key={f.name} className="flex gap-3 py-2.5">
              <span className="text-xs font-mono text-blue-400 w-36 shrink-0 pt-px">{f.name}</span>
              <span className="text-xs text-slate-400 leading-relaxed">{f.desc}</span>
            </div>
          ))}
        </div>
      </Panel>

      {/* ── Portfolio engine ── */}
      <Panel title="Portfolio Engine">
        <p className="text-xs text-slate-500 mb-4 leading-relaxed">
          The shared-capital portfolio engine runs a single bar-by-bar simulation across all tickers.
          All four allocation rules operate simultaneously; their combined effect determines how much
          capital each ticker receives on any given bar.
        </p>
        <div className="space-y-3">
          {PORTFOLIO_RULES.map(r => (
            <div key={r.title} className="flex gap-3 rounded-lg bg-slate-800/50 border border-slate-700/40 p-3">
              {r.icon}
              <div>
                <div className="text-xs font-semibold text-slate-200 mb-0.5">{r.title}</div>
                <div className="text-xs text-slate-400 leading-relaxed">{r.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {/* ── Navigation guide ── */}
      <Panel title="Navigation">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[
            { icon: <Cpu size={14} className="text-slate-400" />,        label: 'Predict',   desc: 'Run a single-ticker prediction for a given date range and horizon.' },
            { icon: <BarChart2 size={14} className="text-slate-400" />,  label: 'Backtest',  desc: 'Single-asset or multi-asset shared-capital portfolio backtest with Monte Carlo simulation.' },
            { icon: <FlaskConical size={14} className="text-slate-400" />,label: 'Benchmark', desc: 'Head-to-head comparison of all four models on the same ticker and date range.' },
            { icon: <BrainCircuit size={14} className="text-slate-400" />,label: 'Train',     desc: 'Train and save a regime-conditioned model artifact for use in backtests or predictions.' },
          ].map(n => (
            <div key={n.label} className="flex gap-3 items-start rounded-lg bg-slate-800/50 border border-slate-700/40 p-3">
              <span className="mt-0.5">{n.icon}</span>
              <div>
                <div className="text-xs font-semibold text-slate-200 mb-0.5">{n.label}</div>
                <div className="text-xs text-slate-400 leading-relaxed">{n.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </Panel>

    </div>
  )
}
