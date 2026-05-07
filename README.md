# Quant Trading System

A full-stack algorithmic trading research platform built in Python and React.

**FastAPI backend** — backtesting, paper trading, model training, regime detection  
**React dashboard** — equity curves, portfolio analytics, signal visualization  
**Shared-capital portfolio engine** — inverse-vol weighting, regime-conditional allocation, periodic rebalancing

---

## Architecture

```
src/
  api/           FastAPI routes, schemas, dependency injection
  execution/     Backtest engine, portfolio engine, paper trading, broker simulation
  features/      Feature pipeline (returns, vol, RSI, SMA, TDA topology)
  ingestion/     yfinance + Binance REST historical data loaders
  models/        TCN, TCN-LSTM, TFT, Online (River) regime-conditioned models
  strategy/      Signal generation, risk management
  storage/       Postgres + Redis persistence
dashboard/
  frontend/      Vite + React + Recharts dashboard
pipelines/
  training_pipeline.py   Sequence prep + regime-conditioned model training
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker + Docker Compose (for Postgres and Redis)

---

## Setup

### 1. Clone and configure environment

```bash
git clone https://github.com/keltonmccormick18/Regime-switching-trading-system.git
cd Regime-switching-trading-system
cp .env.example .env
# Edit .env — set POSTGRES_PASSWORD
```

### 2. Start infrastructure

```bash
docker compose up -d
```

### 3. Python environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start the API

```bash
./start_api.sh --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

With TLS:
```bash
./start_api.sh --ssl
```

### 5. Start the dashboard

```bash
./start_dashboard.sh
# Dashboard: http://localhost:5173
```

---

## Key Features

### Backtesting
- Single-asset backtest with regime-conditioned model selection
- Shared-capital portfolio backtest across multiple tickers
- Inverse-volatility weighting with 200-SMA trend filter (no allocation to assets below their 200-day SMA)
- Regime-conditional allocation shifts (HIGH_VOL regimes receive 50% weight reduction)
- Periodic rebalancing (monthly / quarterly / annual)
- Realistic broker simulation: commissions, slippage, spread, execution lag

### Models
| Model | Description |
|-------|-------------|
| `TCNModel` | Temporal Convolutional Network |
| `TCNLSTMModel` | TCN + LSTM hybrid |
| `TFTModel` | Temporal Fusion Transformer |
| `OnlineModel` | River online learning |

Each model is trained per-regime (`LOW_VOL_BULL`, `LOW_VOL_BEAR`, `HIGH_VOL_BULL`, `HIGH_VOL_BEAR`) using TDA-based topological features and optional cross-asset macro features (VIX, credit spread, USD).

### Paper Trading
Simulated paper trading engine that replays historical bars with the same broker simulation and risk controls as the backtest engine.

---

## Environment Variables

See `.env.example` for all supported variables.

| Variable | Required | Description |
|----------|----------|-------------|
| `POSTGRES_PASSWORD` | Yes | Postgres password (matches docker-compose) |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Recommended (Apple Silicon) | Avoids MPS SIGSEGV in attention ops |

---

## TLS (optional)

Self-signed certs are generated automatically when you pass `--ssl` to either start script:

```bash
./make_certs.sh        # generates certs/server.key + certs/server.crt
./start_api.sh --ssl
./start_dashboard.sh   # connects to https API automatically
```

---

## Running Tests

```bash
pytest tests/
```

---

## License

MIT
