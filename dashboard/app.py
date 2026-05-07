"""FastAPI backend for the quant trading dashboard.

Tries to connect to PostgreSQL (trades, metrics) and Redis (signals) from the
storage layer. Falls back to generated mock data when services are unavailable,
so the dashboard works stand-alone without any infrastructure running.

Start:
    cd quant-trading-system:
    uvicorn dashboard.app:app --reload --port 8000

Endpoints:
    GET  /api/summary         latest performance summary
    GET  /api/pnl             equity curve + drawdown time series
    GET  /api/trades          recent filled orders
    GET  /api/signals/{sym}   signal history for a symbol
    GET  /api/positions       current open positions
    WS   /ws/signals          real-time signal stream (broadcast to all clients)
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Optional storage layer — fall back to mock data if unavailable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_db    = None
_cache = None
_HAS_STORAGE = False

try:
    from src.storage.db    import PostgresDB
    from src.storage.cache import SignalCache
    _db    = PostgresDB()
    _cache = SignalCache()
    _HAS_STORAGE = bool(_cache.ping())
except Exception:
    pass

# ---------------------------------------------------------------------------
# Connected WebSocket clients
# ---------------------------------------------------------------------------
_clients: set[WebSocket] = set()


async def _broadcast(payload: str) -> None:
    dead: set[WebSocket] = set()
    for ws in _clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


# ---------------------------------------------------------------------------
# Background task: push signals to all WebSocket clients
# ---------------------------------------------------------------------------
async def _signal_loop() -> None:
    _symbols = ["SPY", "QQQ", "BTCUSDT", "AAPL"]
    _regimes = ["high_vol_bull", "high_vol_bear", "low_vol_bull", "low_vol_bear"]
    _models  = ["TCNModel", "TCNLSTMModel", "TransformerModel"]

    if _HAS_STORAGE:
        # Forward messages from Redis pub/sub
        loop = asyncio.get_event_loop()
        ps   = _cache._client.pubsub()
        ps.subscribe("signals")
        while True:
            msg = await loop.run_in_executor(None, ps.get_message, True, 1.0)
            if msg and msg["type"] == "message":
                data = msg["data"]
                await _broadcast(data if isinstance(data, str) else data.decode())
    else:
        # Generate realistic mock signals every ~2.5 s
        base = 450.0
        while True:
            sym = random.choice(_symbols)
            jitter = random.gauss(0, 0.3)
            sig = {
                "symbol":     sym,
                "signal":     random.choices([-1, 0, 1], weights=[2, 3, 2])[0],
                "timestamp":  datetime.now(tz=timezone.utc).isoformat(),
                "regime":     random.choice(_regimes),
                "model_used": random.choice(_models),
                "prediction": [round(base + jitter + random.gauss(0, 0.8) * (i + 1) * 0.05, 2)
                               for i in range(16)],
                "confidence": round(random.uniform(0.45, 0.88), 3),
            }
            await _broadcast(json.dumps(sig))
            await asyncio.sleep(2.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_signal_loop())
    yield
    task.cancel()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Quant Dashboard API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Mock data generators  (used when storage layer is unavailable)
# ---------------------------------------------------------------------------
_PNL_CACHE: list[dict] | None = None


def _build_pnl(n_days: int = 365) -> list[dict]:
    global _PNL_CACHE
    if _PNL_CACHE:
        return _PNL_CACHE
    rng = np.random.default_rng(42)
    rets  = rng.normal(0.00045, 0.011, n_days)
    eq    = np.cumprod(1 + rets)
    peak  = np.maximum.accumulate(eq)
    dd    = (eq - peak) / peak
    end   = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    dates = [end - timedelta(days=n_days - i - 1) for i in range(n_days)]
    _PNL_CACHE = [
        {"timestamp": d.isoformat(), "equity": float(e), "drawdown": float(d_)}
        for d, e, d_ in zip(dates, eq, dd)
    ]
    return _PNL_CACHE


def _build_summary() -> dict:
    pnl    = _build_pnl()
    eq     = [p["equity"] for p in pnl]
    rets   = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))]
    mean_r = float(np.mean(rets))
    std_r  = float(np.std(rets)) or 1e-9
    return {
        "run_id":       "run-demo-001",
        "ticker":       "SPY",
        "sharpe":       round(mean_r / std_r * math.sqrt(252), 3),
        "total_return": round(float(eq[-1]) - 1, 4),
        "max_drawdown": round(float(min(p["drawdown"] for p in pnl)), 4),
        "win_rate":     0.542,
        "n_trades":     87,
        "regime":       "low_vol_bull",
        "model_used":   "TCNLSTMModel",
    }


def _build_trades(n: int = 30) -> list[dict]:
    rng      = random.Random(7)
    symbols  = ["SPY", "QQQ", "AAPL", "MSFT", "BTCUSDT"]
    regimes  = ["high_vol_bull", "high_vol_bear", "low_vol_bull", "low_vol_bear"]
    models   = ["TCNModel", "TCNLSTMModel", "TransformerModel"]
    now      = datetime.now(tz=timezone.utc)
    trades   = [
        {
            "id":         i + 1,
            "created_at": (now - timedelta(hours=rng.randint(1, 720))).isoformat(),
            "symbol":     rng.choice(symbols),
            "side":       rng.choice(["buy", "sell"]),
            "price":      round(rng.uniform(380, 510), 2),
            "quantity":   rng.choice([10, 25, 50, 100]),
            "strategy":   "backtest",
            "regime":     rng.choice(regimes),
            "model_used": rng.choice(models),
            "notes":      "",
        }
        for i in range(n)
    ]
    return sorted(trades, key=lambda t: t["created_at"], reverse=True)


def _build_positions() -> list[dict]:
    return [
        {
            "symbol": "SPY",   "side": "long",  "quantity": 100,
            "entry_price": 450.20, "current_price": 452.14,
            "pnl": 194.0,  "pnl_pct":  0.0043, "regime": "low_vol_bull",
        },
        {
            "symbol": "QQQ",   "side": "long",  "quantity": 50,
            "entry_price": 378.60, "current_price": 376.20,
            "pnl": -120.0, "pnl_pct": -0.0063, "regime": "low_vol_bear",
        },
        {
            "symbol": "BTCUSDT", "side": "short", "quantity": 0.5,
            "entry_price": 62_400.0, "current_price": 61_820.0,
            "pnl": 290.0,  "pnl_pct":  0.0093, "regime": "high_vol_bear",
        },
    ]


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.get("/api/summary")
def get_summary() -> dict:
    if _HAS_STORAGE:
        rows = _db.get_metrics(limit=1)
        if rows:
            m = rows[0]
            return {
                "run_id": m.run_id, "ticker": m.ticker,
                "sharpe": m.sharpe, "total_return": m.total_return,
                "max_drawdown": m.max_drawdown, "win_rate": m.win_rate,
                "n_trades": m.n_trades, "regime": m.regime, "model_used": m.model_used,
            }
    return _build_summary()


@app.get("/api/pnl")
def get_pnl() -> list[dict]:
    return _build_pnl()


@app.get("/api/trades")
def get_trades(symbol: str | None = None, limit: int = 50) -> list[dict]:
    if _HAS_STORAGE:
        trades = _db.get_trades(symbol=symbol, limit=limit)
        return [
            {
                "id": t.id,
                "created_at": t.created_at.isoformat() if t.created_at else "",
                "symbol": t.symbol, "side": t.side,
                "price": t.price,   "quantity": t.quantity,
                "strategy": t.strategy, "regime": t.regime,
                "model_used": t.model_used, "notes": t.notes,
            }
            for t in trades
        ]
    return _build_trades()


@app.get("/api/signals/{symbol}")
def get_signals(symbol: str, n: int = 60) -> list[dict]:
    if _HAS_STORAGE:
        sigs = _cache.get_signal_history(symbol, n=n)
        return [s.to_dict() for s in sigs]
    rng = random.Random(symbol)
    now = datetime.now(tz=timezone.utc)
    return [
        {
            "symbol":     symbol.upper(),
            "signal":     rng.choices([-1, 0, 1], weights=[2, 3, 2])[0],
            "timestamp":  (now - timedelta(minutes=(n - i) * 5)).isoformat(),
            "regime":     rng.choice(["high_vol_bull", "low_vol_bull"]),
            "model_used": "TCNLSTMModel",
            "prediction": [round(450 + rng.gauss(0, 1.5), 2) for _ in range(16)],
            "confidence": round(rng.uniform(0.42, 0.87), 3),
        }
        for i in range(n)
    ]


@app.get("/api/positions")
def get_positions() -> list[dict]:
    return _build_positions()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients.add(websocket)
    try:
        # Hold connection open; _signal_loop pushes messages via _broadcast
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)
