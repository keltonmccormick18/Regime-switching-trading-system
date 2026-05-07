"""Quant Trading System — FastAPI application.

Endpoints
---------
GET  /health
POST /predict
POST /train
GET  /train/{job_id}
POST /trade
GET  /trades
GET  /metrics
POST /metrics
GET  /positions
GET  /signals/{symbol}
GET  /regime/{ticker}
GET  /artifacts
GET  /strategy/status
POST /strategy/reset
POST /strategy/bar
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import (
    get_artifacts,
    get_artifacts_optional,
    get_cache,
    get_cache_optional,
    get_db,
    get_db_optional,
    get_engine,
    get_engine_optional,
    get_tracker,
)
from src.api.schemas import (
    ArtifactRecord,
    ArtifactsResponse,
    BacktestRequest,
    BacktestResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    CostSummary,
    HealthResponse,
    MetricRecord,
    MetricsResponse,
    PaperStartRequest,
    PaperStatusResponse,
    PortfolioBacktestRequest,
    PortfolioBacktestResponse,
    PositionRecord,
    PositionsResponse,
    PredictRequest,
    PredictResponse,
    RegimeMetrics,
    RegimeResponse,
    SignalResponse,
    TickerBacktestResult,
    TradeRequest,
    TradeResponse,
    TrainRequest,
    TrainResponse,
)

# ──────────────────────────── In-memory registries ──────────────────────────

_jobs:            dict[str, TrainResponse] = {}
_cancelled_jobs:  set[str] = set()
_paper_engines:   dict[str, object] = {}   # ticker → PaperEngine


# ──────────────────────────── Lifespan ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up singletons (non-fatal if backends are down)
    from src.api.dependencies import (
        _get_db_singleton,
        _get_cache_singleton,
        _get_artifacts_singleton,
        _get_tracker_singleton,
    )
    _get_db_singleton()
    _get_cache_singleton()
    _get_artifacts_singleton()
    _get_tracker_singleton()
    yield


# ──────────────────────────── App ──────────────────────────────────────────

app = FastAPI(
    title="Quant Trading System API",
    version="1.0.0",
    description="Regime-switching ML trading system: predict, trade, backtest.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────── WebSocket manager ────────────────────────────

class _WSManager:
    """Broadcast hub for live signal updates."""

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients = [c for c in self._clients if c is not ws]

    async def broadcast(self, payload: dict) -> None:
        text = json.dumps(payload)
        dead: list[WebSocket] = []
        for client in list(self._clients):
            try:
                await client.send_text(text)
            except Exception:
                dead.append(client)
        for ws in dead:
            self.disconnect(ws)


_ws_manager = _WSManager()


@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    """Stream live signals to connected dashboard clients."""
    await _ws_manager.connect(websocket)
    try:
        # Keep the connection alive; client sends pings, we ignore them
        while True:
            await asyncio.wait_for(websocket.receive_text(), timeout=30)
    except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
        pass
    finally:
        _ws_manager.disconnect(websocket)


# ──────────────────────────── Summary / PnL (dashboard widgets) ─────────────

def _best_paper_engine():
    """Return the running paper engine with the most bars, or None."""
    best = None
    for pe in _paper_engines.values():
        s = pe.status()
        if s["running"] and (best is None or s["bar_count"] > best.status()["bar_count"]):
            best = pe
    return best


@app.get("/summary", tags=["analytics"])
def get_summary(db=Depends(get_db_optional)):
    """Live paper engine portfolio → DB metrics → zeros fallback."""
    # 1. Prefer a running paper engine
    pe = _best_paper_engine()
    if pe is not None:
        s = pe.status()
        p = s["portfolio"]
        return {
            "run_id":       f"paper:{pe.ticker}",
            "ticker":       pe.ticker,
            "sharpe":       p.get("sharpe", 0.0),
            "total_return": p.get("total_return", 0.0),
            "max_drawdown": p.get("max_drawdown", 0.0),
            "win_rate":     0.0,      # not tracked in portfolio
            "n_trades":     s["n_orders"],
            "regime":       "",
            "model_used":   "",
        }
    # 2. Fall back to latest DB metric
    if db is not None:
        try:
            records = db.get_metrics(limit=1)
            if records:
                r = records[0]
                return {
                    "run_id":       r.get("run_id", ""),
                    "ticker":       r.get("ticker", ""),
                    "sharpe":       r.get("sharpe", 0.0),
                    "total_return": r.get("total_return", 0.0),
                    "max_drawdown": r.get("max_drawdown", 0.0),
                    "win_rate":     r.get("win_rate", 0.0),
                    "n_trades":     r.get("n_trades", 0),
                    "regime":       r.get("regime", ""),
                    "model_used":   r.get("model_used", ""),
                }
        except Exception:
            pass
    return {
        "run_id": "", "ticker": "", "sharpe": 0.0,
        "total_return": 0.0, "max_drawdown": 0.0,
        "win_rate": 0.0, "n_trades": 0,
        "regime": "", "model_used": "",
    }


@app.get("/pnl", tags=["analytics"])
def get_pnl(ticker: Optional[str] = Query(None), db=Depends(get_db_optional)):
    """Live paper engine equity curve → DB metric curve → empty fallback."""
    # 1. Prefer paper engine equity history
    pe = _paper_engines.get(ticker.upper()) if ticker else _best_paper_engine()
    if pe is not None:
        history = pe.status().get("equity_history", [])
        if history:
            return history
    # 2. Fall back to synthetic curve from DB metrics
    if db is not None:
        try:
            records = db.get_metrics(limit=200)
            if records:
                points, peak = [], 100_000.0
                for r in reversed(records):
                    equity = 100_000.0 * (1 + r.get("total_return", 0.0))
                    peak   = max(peak, equity)
                    dd     = (equity - peak) / peak if peak > 0 else 0.0
                    ts     = r.get("created_at") or datetime.now(timezone.utc).isoformat()
                    points.append({"timestamp": str(ts), "equity": round(equity, 2),
                                   "drawdown": round(dd, 4)})
                return points
        except Exception:
            pass
    return []


# ──────────────────────────── Health ───────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    db = get_db_optional()
    cache = get_cache_optional()

    pg_ok = False
    if db is not None:
        try:
            db.health_check()
            pg_ok = True
        except Exception:
            pass

    redis_ok = False
    if cache is not None:
        try:
            cache.ping()
            redis_ok = True
        except Exception:
            pass

    status = "ok" if pg_ok and redis_ok else "degraded"
    return HealthResponse(
        status=status,
        postgres=pg_ok,
        redis=redis_ok,
        timestamp=datetime.now(timezone.utc),
    )


# ──────────────────────────── Predict ──────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(
    req: PredictRequest,
    artifacts=Depends(get_artifacts_optional),
    cache=Depends(get_cache_optional),
):
    from src.features.pipeline import build_features, FEATURE_COLS
    from src.models.regime import detect_regime, REGIME_MODELS
    from src.ingestion.historical import load_data
    from pipelines.training_pipeline import prepare_sequences

    # ── load data & features ────────────────────────────────────────────
    try:
        raw_df = load_data(req.ticker, req.start, datetime.now().strftime("%Y-%m-%d"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data load failed: {exc}")

    try:
        _today = datetime.now().strftime("%Y-%m-%d")
        _macro_df = None
        if req.use_macro:
            from src.ingestion.historical import load_macro_data as _lmd
            _macro_df = _lmd(req.start, _today)
        feat_df = build_features(raw_df, use_tda=req.use_tda, macro_df=_macro_df).dropna()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature build failed: {exc}")

    if len(feat_df) < req.lookback:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough data: need {req.lookback} rows, got {len(feat_df)}",
        )

    # ── detect regime ───────────────────────────────────────────────────
    regime = detect_regime(feat_df)
    model_cls = REGIME_MODELS[regime]

    # ── load or train model ─────────────────────────────────────────────
    if req.model_path and artifacts:
        try:
            model = artifacts.load(req.model_path)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Artifact load failed: {exc}")
    else:
        # Quick train on full history
        X, y = prepare_sequences(feat_df, lookback=req.lookback, horizon=req.horizon)
        model = model_cls(horizon=req.horizon)
        model.fit(X, y)

    # ── predict on the most recent window ───────────────────────────────
    feature_data = feat_df[FEATURE_COLS].values       # (T, n_features)
    window = feature_data[-req.lookback:]             # (lookback, n_features)
    X_pred = window.T[np.newaxis, :, :]               # (1, n_features, lookback)
    current_price = float(feat_df["close"].iloc[-1])
    pred_returns = model.predict(X_pred)[0]           # (horizon,) forward returns
    # Convert returns → predicted prices for backward-compatible API response
    prediction = (current_price * (1.0 + pred_returns)).tolist()

    # ── run through strategy engine (optional — non-fatal if unavailable) ─
    engine = get_engine_optional()
    engine_signal = None
    if engine is not None:
        try:
            import numpy as _np
            engine_signal = engine.sig_gen.from_price_prediction(
                symbol        = req.ticker,
                prediction    = _np.array(prediction),
                current_price = current_price,
                regime        = regime.name,
                model_used    = model_cls.__name__,
            )
        except Exception:
            pass

    # ── publish to Redis + WebSocket clients ────────────────────────────
    direction  = int(engine_signal.direction)  if engine_signal else int(
        np.sign(prediction[-1] - current_price)
    )
    confidence = float(engine_signal.confidence) if engine_signal else 0.5

    if cache is not None:
        try:
            from src.storage.cache import Signal as CacheSignal
            cs = CacheSignal(
                symbol     = req.ticker,
                signal     = direction,
                confidence = confidence,
                regime     = regime.name,
                model_used = model_cls.__name__,
                timestamp  = datetime.now(timezone.utc).isoformat(),
            )
            cache.broadcast(cs)
        except Exception:
            pass

    # Push to any connected dashboard WebSocket clients (fire-and-forget)
    try:
        asyncio.get_event_loop().call_soon_threadsafe(
            lambda: asyncio.ensure_future(_ws_manager.broadcast({
                "symbol":     req.ticker,
                "signal":     direction,
                "confidence": confidence,
                "regime":     regime.name,
                "model_used": model_cls.__name__,
                "prediction": prediction,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
            }))
        )
    except Exception:
        pass

    return PredictResponse(
        ticker=req.ticker,
        regime=regime.name,
        model_used=model_cls.__name__,
        prediction=prediction,
        horizon=req.horizon,
        timestamp=datetime.now(timezone.utc),
    )


# ──────────────────────────── Train ────────────────────────────────────────

def _run_training_job(job_id: str, req: TrainRequest) -> None:
    """Background task: train → backtest → persist artifact + metrics."""
    if job_id in _cancelled_jobs:
        return
    _jobs[job_id].status = "running"

    try:
        from src.features.pipeline import build_features
        from src.models.regime import detect_regime, REGIME_MODELS
        from src.ingestion.historical import load_data
        from pipelines.training_pipeline import prepare_sequences
        from src.execution.backtest import backtest
        import pandas as pd

        raw_df = load_data(req.ticker, req.start, datetime.now().strftime("%Y-%m-%d"))
        if job_id in _cancelled_jobs:
            return
        _macro_df = None
        if getattr(req, "use_macro", False):
            from src.ingestion.historical import load_macro_data as _lmd
            _macro_df = _lmd(req.start, datetime.now().strftime("%Y-%m-%d"))
        feat_df = build_features(raw_df, use_tda=req.use_tda, macro_df=_macro_df).dropna()

        regime = detect_regime(feat_df)
        model_cls = REGIME_MODELS[regime]

        X, y = prepare_sequences(feat_df, lookback=req.lookback, horizon=req.horizon)
        split = int(len(X) * 0.8)
        model = model_cls(horizon=req.horizon)
        model.fit(X[:split], y[:split])

        # Backtest on held-out set
        preds = model.predict(X[split:])   # (n_test, horizon) forward returns
        signals = pd.Series(np.sign(preds[:, 0]))
        bt_df = pd.DataFrame({"close": closes[: len(preds)]})
        result = backtest(bt_df, signals=signals)

        # Persist artifact
        artifact_path = None
        artifacts = get_artifacts_optional()
        if artifacts and req.save_artifact:
            name = req.artifact_name or f"{req.ticker}_{regime.name}_{job_id[:8]}"
            artifact_path = artifacts.save(
                model,
                name,
                metadata={
                    "ticker": req.ticker,
                    "regime": regime.name,
                    "sharpe": result.sharpe,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "job_id": job_id,
                },
            )

        # Write metrics to DB
        db = get_db_optional()
        if db is not None:
            from src.storage.db import Metric
            db.insert_metric(
                Metric(
                    run_id=job_id,
                    ticker=req.ticker,
                    sharpe=result.sharpe,
                    total_return=result.total_return,
                    max_drawdown=result.max_drawdown,
                    win_rate=result.win_rate,
                    n_trades=result.n_trades,
                    regime=regime.name,
                    model_used=model_cls.__name__,
                    extra={"notes": f"auto-train job {job_id}"},
                )
            )

        _jobs[job_id].status = "done"
        _jobs[job_id].regime = regime.name
        _jobs[job_id].model_used = model_cls.__name__
        _jobs[job_id].artifact_path = artifact_path
        _jobs[job_id].finished_at = datetime.now(timezone.utc)

    except Exception as exc:
        _jobs[job_id].status = "error"
        _jobs[job_id].message = str(exc)
        _jobs[job_id].finished_at = datetime.now(timezone.utc)


@app.post("/train", response_model=TrainResponse, status_code=202, tags=["training"])
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job = TrainResponse(
        job_id=job_id,
        ticker=req.ticker,
        status="queued",
        started_at=datetime.now(timezone.utc),
    )
    _jobs[job_id] = job
    background_tasks.add_task(_run_training_job, job_id, req)
    return job


@app.get("/train", tags=["training"])
def list_jobs():
    return {"jobs": list(_jobs.values()), "count": len(_jobs)}


@app.get("/train/{job_id}", response_model=TrainResponse, tags=["training"])
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.delete("/train/{job_id}", response_model=TrainResponse, tags=["training"])
def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status in ("queued", "running"):
        _cancelled_jobs.add(job_id)
        job.status = "error"
        job.message = "Cancelled by user"
        job.finished_at = datetime.now(timezone.utc)
    return job


# ──────────────────────────── Trade ────────────────────────────────────────

@app.post("/trade", response_model=TradeResponse, tags=["trading"])
def trade(
    req: TradeRequest,
    db=Depends(get_db),
    tracker=Depends(get_tracker),
    cache=Depends(get_cache_optional),
):
    from src.storage.db import Trade

    t = Trade(
        symbol=req.symbol,
        side=req.side,
        quantity=req.quantity,
        price=req.price,
        strategy=req.strategy,
        regime=req.regime,
        model_used=req.model_used,
        notes=req.notes,
    )

    try:
        trade_id = db.insert_trade(t)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB insert failed: {exc}")

    tracker.apply_trade(req.symbol, req.side, req.quantity, req.price)

    if cache is not None:
        try:
            from src.storage.cache import Signal
            sig = Signal(
                symbol=req.symbol,
                signal=1.0 if req.side == "buy" else -1.0,
                confidence=1.0,
                regime=req.regime,
                model_used=req.model_used,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            cache.broadcast(sig)
        except Exception:
            pass

    return TradeResponse(
        trade_id=trade_id,
        symbol=req.symbol,
        side=req.side,
        price=req.price,
        quantity=req.quantity,
        strategy=req.strategy,
        regime=req.regime,
        model_used=req.model_used,
        created_at=datetime.now(timezone.utc),
    )


@app.get("/trades", tags=["trading"])
def get_trades(
    symbol: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db=Depends(get_db),
):
    try:
        trades = db.get_trades(symbol=symbol, strategy=strategy, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"trades": trades, "count": len(trades)}


# ──────────────────────────── Metrics ──────────────────────────────────────

@app.get("/metrics", response_model=MetricsResponse, tags=["analytics"])
def get_metrics(
    ticker: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db=Depends(get_db),
):
    try:
        records = db.get_metrics(run_id=run_id, ticker=ticker, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return MetricsResponse(
        ticker=ticker,
        run_id=run_id,
        records=[MetricRecord(**r) for r in records],
        count=len(records),
    )


@app.post("/metrics", response_model=MetricRecord, status_code=201, tags=["analytics"])
def post_metric(record: MetricRecord, db=Depends(get_db)):
    from src.storage.db import Metric

    m = Metric(
        run_id=record.run_id,
        ticker=record.ticker,
        sharpe=record.sharpe or 0.0,
        total_return=record.total_return or 0.0,
        max_drawdown=record.max_drawdown or 0.0,
        win_rate=record.win_rate or 0.0,
        n_trades=record.n_trades or 0,
        regime=record.regime or "",
        model_used=record.model_used or "",
        extra={"notes": record.notes} if record.notes else {},
    )
    try:
        inserted_id = db.insert_metric(m)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    record.id = inserted_id
    record.created_at = datetime.now(timezone.utc)
    return record


# ──────────────────────────── Positions ────────────────────────────────────

@app.get("/positions", response_model=PositionsResponse, tags=["trading"])
def get_positions(tracker=Depends(get_tracker)):
    import yfinance as yf

    open_pos = tracker.open_positions()

    if not open_pos:
        return PositionsResponse(
            positions=[],
            total_unrealized_pnl=0.0,
            as_of=datetime.now(timezone.utc),
        )

    # Batch fetch current prices
    symbols = [p.symbol for p in open_pos]
    prices: dict[str, float] = {}
    try:
        tickers_obj = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            hist = tickers_obj.tickers[sym].history(period="1d")
            if not hist.empty:
                prices[sym] = float(hist["Close"].iloc[-1])
    except Exception:
        pass  # fall back to avg_entry if price fetch fails

    records: list[PositionRecord] = []
    total_upnl = 0.0

    for pos in open_pos:
        cp = prices.get(pos.symbol, pos.avg_entry)
        upnl = pos.unrealized_pnl(cp)
        total_upnl += upnl
        records.append(
            PositionRecord(
                symbol=pos.symbol,
                quantity=pos.quantity,
                avg_entry=pos.avg_entry,
                current_price=cp,
                unrealized_pnl=upnl,
                unrealized_pnl_pct=pos.unrealized_pnl_pct(cp),
                side=pos.side,
            )
        )

    return PositionsResponse(
        positions=records,
        total_unrealized_pnl=total_upnl,
        as_of=datetime.now(timezone.utc),
    )


# ──────────────────────────── Signals ──────────────────────────────────────

@app.get("/signals/{symbol}", response_model=list[SignalResponse], tags=["inference"])
def get_signals(
    symbol: str,
    limit: int = Query(20, ge=1, le=200),
    cache=Depends(get_cache),
):
    try:
        raw = cache.get_signal_history(symbol.upper(), n=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return [
        SignalResponse(
            symbol=s.symbol,
            signal=s.signal,
            confidence=s.confidence,
            regime=s.regime,
            model_used=s.model_used,
            timestamp=datetime.fromisoformat(s.timestamp),
        )
        for s in raw
    ]


# ──────────────────────────── Regime ───────────────────────────────────────

@app.get("/regime/{ticker}", response_model=RegimeResponse, tags=["inference"])
def get_regime(ticker: str, use_tda: bool = Query(False)):
    from src.features.pipeline import build_features
    from src.models.regime import detect_regime
    from src.ingestion.historical import load_data

    try:
        raw_df = load_data(
            ticker.upper(),
            "2022-01-01",
            datetime.now().strftime("%Y-%m-%d"),
        )
        feat_df = build_features(raw_df, use_tda=use_tda).dropna()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data/feature error: {exc}")

    regime = detect_regime(feat_df)
    last = feat_df.iloc[-1]

    return RegimeResponse(
        ticker=ticker.upper(),
        regime=regime.name,
        tda_l1=float(last.get("tda_l1", 0.0)),
        tda_l2=float(last.get("tda_l2", 0.0)),
        vol_20=float(last.get("vol_20", 0.0)),
        sma_ratio_200=float(last.get("sma_ratio_200", 0.0)),
        timestamp=datetime.now(timezone.utc),
    )


# ──────────────────────────── Artifacts ────────────────────────────────────

@app.get("/artifacts", response_model=ArtifactsResponse, tags=["system"])
def list_artifacts(artifacts=Depends(get_artifacts)):
    try:
        records = artifacts.list_artifacts()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ArtifactsResponse(
        artifacts=[
            ArtifactRecord(
                name=r.get("name", ""),
                path=r.get("path", ""),
                class_name=r.get("class_name", ""),
                config=r.get("config", {}),
                metadata=r.get("metadata", {}),
                created_at=r.get("created_at"),
            )
            for r in records
        ],
        count=len(records),
    )


# ──────────────────────────── Strategy Engine ──────────────────────────────

@app.get("/strategy/status", tags=["strategy"])
def strategy_status(engine=Depends(get_engine)):
    """Return portfolio summary + risk status for the live strategy engine."""
    return engine.summary()


@app.post("/strategy/reset", tags=["strategy"])
def strategy_reset(engine=Depends(get_engine)):
    """Clear the circuit breaker halt so trading can resume after manual review."""
    was_halted = engine.risk.halted
    engine.risk.reset_halt()
    return {
        "was_halted":  was_halted,
        "halt_reason": engine.risk.halt_reason,
        "status":      "ok",
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }


@app.post("/strategy/bar", tags=["strategy"])
def strategy_bar(
    prices: dict,
    engine=Depends(get_engine),
    db=Depends(get_db_optional),
    cache=Depends(get_cache_optional),
):
    """Push a new price bar into the strategy engine.

    Triggers mark-to-market and stop-loss checks.  Returns any close orders
    generated by risk rules.  Intended for live-trading loops where prices
    arrive bar-by-bar.

    Body: ``{"AAPL": 178.5, "TSLA": 241.0, ...}``
    """
    close_orders = engine.on_bar(prices)

    persisted = []
    for order in close_orders:
        if db is not None:
            try:
                from src.storage.db import Trade
                t = Trade(
                    symbol     = order.symbol,
                    side       = order.side,
                    quantity   = order.quantity,
                    price      = order.price,
                    strategy   = "strategy_engine",
                    regime     = order.regime,
                    model_used = order.model_used,
                    notes      = order.reason,
                )
                db.insert_trade(t)
            except Exception:
                pass
        if cache is not None:
            try:
                from src.storage.cache import Signal as CacheSignal
                cs = CacheSignal(
                    symbol     = order.symbol,
                    signal     = -1 if order.side == "sell" else 1,
                    confidence = 1.0,
                    regime     = order.regime,
                    model_used = order.model_used,
                    timestamp  = order.timestamp,
                )
                cache.broadcast(cs)
            except Exception:
                pass
        persisted.append(order.to_dict())

    return {
        "close_orders": persisted,
        "n_stops_triggered": len(close_orders),
        "portfolio": engine.portfolio.summary(),
        "risk_status": engine.risk.status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────── Backtest ─────────────────────────────────────

@app.post("/backtest", response_model=BacktestResponse, tags=["execution"])
def run_backtest(req: BacktestRequest, background_tasks: BackgroundTasks):
    """Event-driven backtest with optional Monte Carlo simulation.

    When n_simulations=1 (default) a single run with a random seed is performed.
    When n_simulations>1 the model is retrained with seeds 0..N-1 and results
    are aggregated; simulation_stats in the response carries the distribution.
    """
    from concurrent.futures import ThreadPoolExecutor
    from src.ingestion.historical import load_data
    from src.features.pipeline import build_features
    from src.models.regime import detect_regime, REGIME_MODELS
    from pipelines.training_pipeline import prepare_sequences
    from src.execution.backtest_engine import BacktestEngine
    from src.execution.broker import SimulatedBroker
    from src.api.schemas import SimulationStats

    end = req.end or datetime.now().strftime("%Y-%m-%d")

    # ── Load data + features (shared across all simulations) ──────────────
    try:
        raw_df = load_data(req.ticker, req.start, end)
        _macro_df = None
        if req.use_macro:
            from src.ingestion.historical import load_macro_data as _lmd
            _macro_df = _lmd(req.start, end)
        feat_df = build_features(raw_df, use_tda=req.use_tda, macro_df=_macro_df).dropna()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data/feature error: {exc}")

    if len(feat_df) < req.lookback + 50:
        raise HTTPException(status_code=422,
                            detail=f"Not enough data: {len(feat_df)} rows")

    preset = req.broker_preset

    # ── Pre-fitted artifact path — single run, full history ────────────────
    artifacts = get_artifacts_optional()
    if req.model_path and artifacts:
        try:
            model = artifacts.load(req.model_path)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Artifact load failed: {exc}")

        if preset == "zero_cost":   broker = SimulatedBroker.zero_cost()
        elif preset == "institutional": broker = SimulatedBroker.institutional()
        else:                           broker = SimulatedBroker.retail()

        be = BacktestEngine(broker=broker, initial_capital=req.initial_capital)
        try:
            result = be.run_model(req.ticker, feat_df, model,
                                  req.lookback, req.horizon,
                                  position_size=req.position_size)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")

        cs_raw = be.cost_summary()
        sim_stats = None
        all_results = [result]

    else:
        # ── Train-fresh path: Monte Carlo over N seeds ─────────────────────
        regime    = detect_regime(feat_df)
        model_cls = REGIME_MODELS[regime]
        X, y      = prepare_sequences(feat_df, req.lookback, req.horizon)
        split     = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]

        test_df = feat_df.iloc[split:].reset_index(drop=True)
        if len(test_df) < req.lookback + 10:
            raise HTTPException(status_code=422,
                                detail="Test split too short — use a longer date range or smaller lookback")

        # ── Per-window regime labels (computed once, shared across all sims) ─
        # X[j] uses the lookback window ending at feat_df row (j + lookback - 1).
        # We label each window by the regime at its last bar so each model only
        # trains on the dynamics it will actually be called to predict.
        _regime_cols_ok = {"tda_l1", "vol_20", "sma_ratio_200"}.issubset(feat_df.columns)
        if _regime_cols_ok:
            from src.execution.backtest_engine import _regime_series as _rs
            _bar_regimes = _rs(feat_df)   # one label per feat_df row
            window_regimes: list = [
                _bar_regimes[j + req.lookback - 1]
                for j in range(len(X))
            ]
        else:
            window_regimes = []

        _MIN_REGIME_SAMPLES = 10   # fall back to full set if a regime is too rare

        def _one_sim(seed: int):
            import random as _r
            import traceback as _tb
            import numpy as _np
            from src.models.regime import REGIME_MODELS as _REGIME_MODELS
            _r.seed(seed); _np.random.seed(seed)
            try:
                import torch as _t
                _t.manual_seed(seed)
                _t.backends.cudnn.deterministic = True
            except Exception:
                pass
            try:
                # Train one model per regime, filtered to windows in that regime.
                # Each model sees only the dynamics it will be called to predict OOS.
                # When low_vol_bull_buy_hold is True, skip LOW_VOL_BULL entirely —
                # no model needed since the engine will hold long unconditionally.
                n_features = X_train.shape[1]
                fitted: dict = {}
                for _regime_enum, _model_cls in _REGIME_MODELS.items():
                    _rname = _regime_enum.name
                    if req.low_vol_bull_buy_hold and _rname == "LOW_VOL_BULL":
                        continue  # buy-and-hold mode: skip training this slot
                    if window_regimes:
                        _idx = [i for i, r in enumerate(window_regimes[:split])
                                if r == _rname]
                        # Persistence filter: keep only windows where ≥50% of
                        # the next `horizon` bars stay in the same regime.
                        # Removes near-bottom reversal noise from training.
                        _bar_regimes_inner = _bar_regimes  # captured from outer scope
                        def _bt_persistent(idx: int, rname: str = _rname) -> bool:
                            bar_end = idx + req.lookback - 1
                            fwd = [
                                _bar_regimes_inner[bar_end + k]
                                for k in range(1, req.horizon + 1)
                                if bar_end + k < len(_bar_regimes_inner)
                            ]
                            return bool(fwd) and sum(1 for b in fwd if b == rname) / len(fwd) >= 0.5
                        _pidx = [i for i in _idx if _bt_persistent(i)]
                        if len(_pidx) >= _MIN_REGIME_SAMPLES:
                            _X_r = X_train[_pidx]
                            _y_r = y_train[_pidx]
                        elif len(_idx) >= _MIN_REGIME_SAMPLES:
                            _X_r = X_train[_idx]   # persistence filter too aggressive
                            _y_r = y_train[_idx]
                        else:
                            _X_r, _y_r = X_train, y_train  # regime too rare
                    else:
                        _X_r, _y_r = X_train, y_train

                    _m = _model_cls(horizon=req.horizon, n_features=n_features)
                    _m.fit(_X_r, _y_r)
                    fitted[_rname] = _m

                # Re-seed broker randomness after training
                _r.seed(seed)
                if preset == "zero_cost":       b = SimulatedBroker.zero_cost()
                elif preset == "institutional": b = SimulatedBroker.institutional()
                else:                           b = SimulatedBroker.retail()
                be = BacktestEngine(broker=b, initial_capital=req.initial_capital)
                fallback = next(iter(fitted.values()))

                # Determine LOW_VOL_BULL treatment:
                #   buy-and-hold mode → regime_always_long (unconditional, no model)
                #   active model mode → regime_default_long from caller (or None)
                if req.low_vol_bull_buy_hold:
                    _bt_ral: Optional[set] = {"LOW_VOL_BULL"}
                    _bt_rdl: Optional[set] = None
                else:
                    _bt_ral = None
                    _bt_rdl = set(req.regime_default_long) if req.regime_default_long else None

                return be.run_model(
                    req.ticker, test_df, fallback,
                    req.lookback, req.horizon,
                    position_size=req.position_size,
                    regime_models=fitted,
                    signal_hold=req.signal_hold,
                    long_only=req.long_only,
                    signal_threshold=req.signal_threshold,
                    min_confidence=req.min_confidence,
                    stop_loss_pct=req.stop_loss_pct,
                    regime_signal_thresholds=req.regime_signal_thresholds,
                    regime_long_only=req.regime_long_only,
                    regime_default_long=_bt_rdl,
                    regime_always_long=_bt_ral,
                    retrain_every=req.retrain_every,
                    retrain_epochs=req.retrain_epochs,
                    full_X=X,
                    full_y=y,
                    full_regimes=window_regimes,
                    train_split=split,
                ), be.cost_summary()
            except Exception as exc:
                raise RuntimeError(
                    f"seed={seed} {type(exc).__name__}: {exc}\n{_tb.format_exc()}"
                ) from exc

        n = req.n_simulations
        seeds = list(range(n))
        max_workers = min(n, 4)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                runs = list(pool.map(_one_sim, seeds))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")

        all_results = [r for r, _ in runs]
        cs_raw      = runs[0][1]   # cost summary from first run (representative)

        # ── Aggregate across simulations ───────────────────────────────────
        def _arr(attr):
            return np.array([getattr(r, attr) for r in all_results])

        if n > 1:
            min_len = min(len(r.equity) for r in all_results)
            eq_mat  = np.array([r.equity[:min_len] for r in all_results])

            sim_stats = SimulationStats(
                n_simulations    = n,
                sharpe_mean      = float(np.mean(_arr("sharpe"))),
                sharpe_std       = float(np.std(_arr("sharpe"))),
                total_return_mean= float(np.mean(_arr("total_return"))),
                total_return_std = float(np.std(_arr("total_return"))),
                max_drawdown_mean= float(np.mean(_arr("max_drawdown"))),
                max_drawdown_std = float(np.std(_arr("max_drawdown"))),
                win_rate_mean    = float(np.mean(_arr("win_rate"))),
                win_rate_std     = float(np.std(_arr("win_rate"))),
                alpha_mean       = float(np.mean(_arr("alpha"))),
                alpha_std        = float(np.std(_arr("alpha"))),
                beta_mean        = float(np.mean(_arr("beta"))),
                beta_std         = float(np.std(_arr("beta"))),
                equity_mean      = eq_mat.mean(axis=0).tolist(),
                equity_p10       = np.percentile(eq_mat, 10, axis=0).tolist(),
                equity_p90       = np.percentile(eq_mat, 90, axis=0).tolist(),
            )
            # Use median-Sharpe run as the representative result
            sharpes = _arr("sharpe")
            result  = all_results[int(np.argsort(sharpes)[len(sharpes) // 2])]
        else:
            sim_stats = None
            result    = all_results[0]

    cs = CostSummary(
        n_fills          = cs_raw.get("n_fills", 0),
        total_commission = cs_raw.get("total_commission", 0.0),
        total_slippage   = cs_raw.get("total_slippage", 0.0),
        total_spread     = cs_raw.get("total_spread", 0.0),
        total_friction   = cs_raw.get("total_friction", 0.0),
        avg_latency_ms   = cs_raw.get("avg_latency_ms", 0.0),
    )

    # Persist mean metrics (or single-run metrics) to DB in background
    db = get_db_optional()
    if db is not None:
        _sharpe = sim_stats.sharpe_mean if sim_stats else result.sharpe
        _ret    = sim_stats.total_return_mean if sim_stats else result.total_return
        _dd     = sim_stats.max_drawdown_mean if sim_stats else result.max_drawdown
        def _save_metric():
            try:
                from src.storage.db import Metric
                db.insert_metric(Metric(
                    run_id       = f"backtest_{req.ticker}_{uuid.uuid4().hex[:8]}",
                    ticker       = req.ticker,
                    sharpe       = _sharpe,
                    total_return = _ret,
                    max_drawdown = _dd,
                    win_rate     = result.win_rate,
                    n_trades     = result.n_trades,
                    regime       = "",
                    model_used   = REGIME_MODELS[detect_regime(feat_df)].__name__,
                    extra        = {"broker": preset, "n_sims": req.n_simulations},
                ))
            except Exception:
                pass
        background_tasks.add_task(_save_metric)

    # When Monte Carlo: report mean metrics; equity curve is median-run's curve
    # (simulation_stats carries the full distribution for the chart band)
    sharpe       = sim_stats.sharpe_mean       if sim_stats else result.sharpe
    total_return = sim_stats.total_return_mean if sim_stats else result.total_return
    max_drawdown = sim_stats.max_drawdown_mean if sim_stats else result.max_drawdown
    win_rate     = sim_stats.win_rate_mean     if sim_stats else result.win_rate
    alpha        = sim_stats.alpha_mean        if sim_stats else result.alpha
    beta         = sim_stats.beta_mean         if sim_stats else result.beta
    equity       = sim_stats.equity_mean       if sim_stats else result.equity

    return BacktestResponse(
        ticker           = req.ticker,
        broker_preset    = preset,
        sharpe           = sharpe,
        sortino          = result.sortino,
        total_return     = total_return,
        max_drawdown     = max_drawdown,
        win_rate         = win_rate,
        n_trades         = result.n_trades,
        calmar           = result.calmar,
        avg_trade_return = result.avg_trade_return,
        alpha            = alpha,
        beta             = beta,
        cost_summary     = cs,
        equity           = equity,
        regime_curve     = result.regime_curve,
        simulation_stats = sim_stats,
        bah_equity       = result.bah_equity,
        bah_return       = result.bah_return,
        buy_bars         = result.buy_bars,
        sell_bars        = result.sell_bars,
        notes            = result.notes + (f" | {req.n_simulations} MC runs" if sim_stats else ""),
        timestamp        = datetime.now(timezone.utc),
    )


# ──────────────────────────── Portfolio Backtest ───────────────────────────

@app.post("/portfolio/backtest", response_model=PortfolioBacktestResponse, tags=["execution"])
def portfolio_backtest(req: PortfolioBacktestRequest, background_tasks: BackgroundTasks):
    """Shared-capital multi-ticker portfolio backtest.

    Phase 1 (parallel): load data and train regime-conditioned models per ticker.
    Phase 2 (single loop): SharedCapitalPortfolioEngine runs all tickers through
    one bar-by-bar simulation with a shared cash pool, inverse-vol weighting,
    regime-conditional allocation shifts, and optional periodic rebalancing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.ingestion.historical import load_data
    from src.features.pipeline import build_features
    from src.models.regime import REGIME_MODELS
    from pipelines.training_pipeline import prepare_sequences
    from src.execution.broker import SimulatedBroker
    from src.execution.portfolio_engine import SharedCapitalPortfolioEngine

    end = req.end or datetime.now().strftime("%Y-%m-%d")
    n   = len(req.tickers)

    # Normalise weights (used as base allocation seed)
    if req.weights and len(req.weights) == n:
        total   = sum(req.weights)
        weights = [w / total for w in req.weights]
    else:
        weights = [1.0 / n] * n
    base_weights = dict(zip(req.tickers, weights))

    # ── Phase 1: per-ticker training (parallel) ────────────────────────────
    # Returns trained regime models + full feature df (DatetimeIndex intact)
    # + the integer split index.  No execution happens here.

    def _train_ticker(ticker: str, weight: float):
        try:
            import random as _r; import numpy as _np
            _r.seed(42); _np.random.seed(42)
            try:
                import torch as _t; _t.manual_seed(42)
            except Exception:
                pass

            raw_df   = load_data(ticker, req.start, end)
            macro_df = None
            if getattr(req, "use_macro", False):
                from src.ingestion.historical import load_macro_data as _lmd
                macro_df = _lmd(req.start, end)

            feat_df = build_features(
                raw_df, use_tda=req.use_tda, macro_df=macro_df
            ).dropna()

            if len(feat_df) < req.lookback + 50:
                raise ValueError(f"Not enough data: {len(feat_df)} rows")

            X, y   = prepare_sequences(feat_df, req.lookback, req.horizon)
            split  = int(len(X) * 0.8)
            X_tr, y_tr = X[:split], y[:split]

            # Per-window regime labels for regime-conditioned training
            from src.execution.backtest_engine import _regime_series as _rs
            if {"tda_l1", "vol_20", "sma_ratio_200"}.issubset(feat_df.columns):
                bar_reg = _rs(feat_df)
                win_reg = [bar_reg[j + req.lookback - 1] for j in range(len(X))]
            else:
                win_reg = []

            _MIN = 20
            n_feat = X_tr.shape[1]
            fitted: dict = {}
            for _re, _mc in REGIME_MODELS.items():
                _rn = _re.name
                if win_reg:
                    _idx = [i for i, r in enumerate(win_reg[:split]) if r == _rn]
                    _Xr, _yr = (X_tr[_idx], y_tr[_idx]) if len(_idx) >= _MIN else (X_tr, y_tr)
                else:
                    _Xr, _yr = X_tr, y_tr
                _m = _mc(horizon=req.horizon, n_features=n_feat)
                _m.fit(_Xr, _yr)
                fitted[_rn] = _m

            return ticker, weight, fitted, feat_df, split, None
        except Exception as exc:
            return ticker, weight, None, None, None, str(exc)

    train_results = {}
    train_errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
        futures = {
            pool.submit(_train_ticker, t, w): t
            for t, w in zip(req.tickers, weights)
        }
        for fut in as_completed(futures):
            t, w, fitted, feat_df, split, err = fut.result()
            if err:
                train_errors[t] = err
            else:
                train_results[t] = (w, fitted, feat_df, split)

    if not train_results:
        raise HTTPException(
            status_code=500,
            detail=f"All tickers failed training: {train_errors}",
        )

    # Rebuild lists in original request order, skipping failed tickers
    valid_tickers = [t for t in req.tickers if t in train_results]
    fitted_models = {t: train_results[t][1] for t in valid_tickers}
    feat_dfs      = {t: train_results[t][2] for t in valid_tickers}
    train_splits  = {t: train_results[t][3] for t in valid_tickers}
    # Re-normalise base weights across the tickers that succeeded
    _bw_raw = {t: base_weights[t] for t in valid_tickers}
    _bw_sum = sum(_bw_raw.values()) or 1.0
    base_weights_valid = {t: v / _bw_sum for t, v in _bw_raw.items()}

    # ── Phase 2: shared-capital simulation ────────────────────────────────
    preset = req.broker_preset
    if preset == "zero_cost":
        broker = SimulatedBroker.zero_cost()
    elif preset == "institutional":
        broker = SimulatedBroker.institutional()
    else:
        broker = SimulatedBroker.retail()

    engine = SharedCapitalPortfolioEngine(
        tickers        = valid_tickers,
        base_weights   = base_weights_valid,
        fitted_models  = fitted_models,
        feat_dfs       = feat_dfs,
        train_splits   = train_splits,
        broker         = broker,
        initial_capital= req.initial_capital,
        lookback       = req.lookback,
        horizon        = req.horizon,
        vol_weight     = getattr(req, "vol_weight", True),
        regime_shift   = getattr(req, "regime_shift", True),
        rebalance_freq = getattr(req, "rebalance_freq", "never"),
        long_only      = getattr(req, "long_only", True),
    )
    eng_result = engine.run()

    # ── Phase 3: build response ────────────────────────────────────────────
    portfolio_equity = eng_result.portfolio_equity

    # Per-ticker results (from engine stats + any training failures)
    ticker_results: list[TickerBacktestResult] = []
    stats_by_ticker = {s.ticker: s for s in eng_result.ticker_stats}

    for t in req.tickers:
        if t in stats_by_ticker:
            s = stats_by_ticker[t]
            ticker_results.append(TickerBacktestResult(
                ticker       = t,
                weight       = base_weights_valid.get(t, 0.0),
                sharpe       = s.sharpe,
                sortino      = s.sortino,
                total_return = s.total_return,
                max_drawdown = s.max_drawdown,
                win_rate     = s.win_rate,
                n_trades     = s.n_trades,
                calmar       = s.calmar,
                alpha        = s.alpha,
                beta         = s.beta,
                equity       = s.equity,
                regime_curve = s.regime_curve,
            ))
        else:
            ticker_results.append(TickerBacktestResult(
                ticker=t, weight=base_weights.get(t, 0.0),
                sharpe=0.0, sortino=0.0, total_return=0.0, max_drawdown=0.0,
                win_rate=0.0, n_trades=0, calmar=0.0, equity=[],
                error=train_errors.get(t, "excluded from simulation"),
            ))

    # Portfolio-level metrics
    port_eq = np.array(portfolio_equity, dtype=float)
    if len(port_eq) > 1:
        returns     = np.diff(port_eq) / np.where(port_eq[:-1] > 0, port_eq[:-1], 1.0)
        port_return = float(port_eq[-1] / port_eq[0]) - 1.0
        std_r       = float(returns.std())
        port_sharpe = (
            float(returns.mean()) / std_r * np.sqrt(252) if std_r > 1e-12 else 0.0
        )
        peak    = np.maximum.accumulate(port_eq)
        port_mdd = float(((port_eq - peak) / np.where(peak > 0, peak, 1.0)).min())
    else:
        port_return = port_sharpe = port_mdd = 0.0

    # Correlation matrix from per-ticker equity curves
    min_len = min((len(r.equity) for r in ticker_results if r.equity), default=1)
    return_series: list[np.ndarray] = []
    for r in ticker_results:
        if len(r.equity) >= 2:
            eq  = np.array(r.equity[:min_len], dtype=float)
            ret = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)
            return_series.append(ret)
        else:
            return_series.append(np.zeros(max(min_len - 1, 1)))

    try:
        mat      = np.array(return_series)
        corr     = np.corrcoef(mat)
        corr_matrix: list[list[float]] = [
            [round(float(corr[i, j]), 4) for j in range(len(ticker_results))]
            for i in range(len(ticker_results))
        ]
    except Exception:
        nt = len(ticker_results)
        corr_matrix = [[1.0 if i == j else 0.0 for j in range(nt)] for i in range(nt)]

    return PortfolioBacktestResponse(
        tickers                = [r.ticker for r in ticker_results],
        weights                = [r.weight for r in ticker_results],
        ticker_results         = ticker_results,
        portfolio_equity       = portfolio_equity,
        portfolio_sharpe       = round(float(port_sharpe), 4),
        portfolio_return       = round(float(port_return), 6),
        portfolio_max_drawdown = round(float(abs(port_mdd)), 6),
        correlation_matrix     = corr_matrix,
        initial_capital        = req.initial_capital,
        timestamp              = datetime.now(timezone.utc),
    )


# ──────────────────────────── Paper Trading ────────────────────────────────

@app.post("/paper/start", response_model=PaperStatusResponse,
          status_code=202, tags=["execution"])
def paper_start(req: PaperStartRequest):
    """Start a paper trading engine for a ticker in the background."""
    from src.execution.paper_engine import PaperEngine, PaperConfig

    if req.ticker in _paper_engines:
        existing = _paper_engines[req.ticker]
        if existing.status()["running"]:
            raise HTTPException(status_code=409,
                                detail=f"Paper engine for {req.ticker} already running")

    cfg = PaperConfig(
        interval             = req.interval,
        broker_preset        = req.broker_preset,
        initial_capital      = req.initial_capital,
        prediction_interval  = req.prediction_interval,
        max_drawdown_limit   = req.max_drawdown_limit,
        stop_loss_pct        = req.stop_loss_pct,
        latency_ms           = req.latency_ms,
        max_position_pct     = req.max_position_pct,
        max_leverage         = req.max_leverage,
        rebalance_threshold  = req.rebalance_threshold,
    )

    # Load pre-fitted model if provided
    model = None
    artifacts = get_artifacts_optional()
    if req.model_path and artifacts:
        try:
            model = artifacts.load(req.model_path)
        except Exception:
            pass

    pe = PaperEngine(ticker=req.ticker, model=model, config=cfg)
    pe.start()
    _paper_engines[req.ticker] = pe

    s = pe.status()
    return PaperStatusResponse(
        ticker      = req.ticker,
        running     = s["running"],
        bar_count   = s["bar_count"],
        last_bar_ts = s["last_bar_ts"],
        n_orders    = s["n_orders"],
        portfolio   = s["portfolio"],
        risk_status = s["risk_status"],
        cost_summary= s["cost_summary"],
        timestamp   = datetime.now(timezone.utc),
    )


@app.get("/paper/{ticker}/equity", tags=["execution"])
def paper_equity(ticker: str):
    """Full equity-curve history for a paper engine (for the dashboard PnL chart).
    Also snapshots current equity so the chart grows on every dashboard poll."""
    pe = _paper_engines.get(ticker.upper())
    if pe is None:
        raise HTTPException(status_code=404, detail=f"No paper engine for {ticker.upper()}")
    pe.snapshot_equity()   # add a point for this moment in time
    return pe.status().get("equity_history", [])


@app.get("/paper/{ticker}/status", response_model=PaperStatusResponse, tags=["execution"])
def paper_status(ticker: str):
    """Get current status of a running paper engine."""
    pe = _paper_engines.get(ticker.upper())
    if pe is None:
        raise HTTPException(status_code=404, detail=f"No paper engine for {ticker.upper()}")
    s = pe.status()
    return PaperStatusResponse(
        ticker      = ticker.upper(),
        running     = s["running"],
        bar_count   = s["bar_count"],
        last_bar_ts = s["last_bar_ts"],
        n_orders    = s["n_orders"],
        portfolio   = s["portfolio"],
        risk_status = s["risk_status"],
        cost_summary= s["cost_summary"],
        timestamp   = datetime.now(timezone.utc),
    )


@app.post("/paper/{ticker}/stop", tags=["execution"])
def paper_stop(ticker: str):
    """Stop a running paper engine."""
    pe = _paper_engines.get(ticker.upper())
    if pe is None:
        raise HTTPException(status_code=404, detail=f"No paper engine for {ticker.upper()}")
    pe.stop()
    s = pe.status()
    return {
        "ticker":    ticker.upper(),
        "running":   s["running"],
        "portfolio": s["portfolio"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/paper", tags=["execution"])
def paper_list():
    """List all paper engines and their running state."""
    return {
        "engines": [
            {"ticker": t, "running": pe.status()["running"],
             "bar_count": pe.status()["bar_count"]}
            for t, pe in _paper_engines.items()
        ],
        "count": len(_paper_engines),
    }


# ──────────────────────────── Model Benchmark ──────────────────────────────

# Maps each model class name to its designated native regime.
_NATIVE_REGIME: dict[str, str] = {
    "TFTModel":    "HIGH_VOL_BULL",
    "OnlineModel": "HIGH_VOL_BEAR",   # A2: OnlineARF replaces LightGBM
    "TCNModel":    "LOW_VOL_BULL",
    "TCNLSTMModel":"LOW_VOL_BEAR",
}


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["execution"])
def run_benchmark(req: BenchmarkRequest):
    """Train a named model and benchmark its performance broken down by market regime.

    The model is trained on the first 80% of data, then evaluated walk-forward
    on the remaining 20%.  Results show how the model performs in its native regime
    vs. all other regimes.

    When n_simulations > 1 the model is retrained with seeds 0..N-1 in parallel;
    per-regime sharpe and return are reported as mean ± std so you can judge
    whether any native-regime advantage is robust or seed-dependent.
    """
    import random
    import traceback
    from concurrent.futures import ThreadPoolExecutor
    from src.ingestion.historical import load_data
    from src.features.pipeline import build_features
    from src.models.tcn import TCNModel
    from src.models.tcn_lstm import TCNLSTMModel
    from src.models.tft import TFTModel
    from src.models.online import OnlineModel
    from src.models.regime import Regime
    from src.api.schemas import SimulationStats
    from pipelines.training_pipeline import prepare_sequences
    from src.execution.backtest_engine import BacktestEngine
    from src.execution.broker import SimulatedBroker

    MODEL_MAP = {
        "TFTModel":    TFTModel,
        "OnlineModel": OnlineModel,   # A2: OnlineARF replaces LightGBM
        "TCNModel":    TCNModel,
        "TCNLSTMModel":TCNLSTMModel,
    }

    end = req.end or datetime.now().strftime("%Y-%m-%d")

    try:
        raw_df = load_data(req.ticker, req.start, end)
        _macro_df = None
        if req.use_macro:
            from src.ingestion.historical import load_macro_data as _lmd
            _macro_df = _lmd(req.start, end)
        feat_df = build_features(raw_df, use_tda=req.use_tda, macro_df=_macro_df).dropna()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data/feature error: {exc}")

    if len(feat_df) < req.lookback + 50:
        raise HTTPException(status_code=422,
                            detail=f"Not enough data: {len(feat_df)} rows")

    X, y  = prepare_sequences(feat_df, req.lookback, req.horizon)
    split = int(len(X) * 0.8)
    n_features = X.shape[1]

    test_df = feat_df.iloc[split:].reset_index(drop=True)
    if len(test_df) < req.lookback + 10:
        raise HTTPException(status_code=422, detail="Test split too short — use a longer date range")

    native    = _NATIVE_REGIME[req.model_name]
    model_cls = MODEL_MAP[req.model_name]
    preset    = req.broker_preset

    # ── Regime-filtered training indices ──────────────────────────────────
    # Label each training window by the regime at its last bar so the model
    # only trains on dynamics matching its designated native regime.
    _bm_regime_cols_ok = {"tda_l1", "vol_20", "sma_ratio_200"}.issubset(feat_df.columns)
    if _bm_regime_cols_ok:
        from src.execution.backtest_engine import _regime_series as _bm_rs
        _bm_bar_regimes = _bm_rs(feat_df)
        _bm_window_regimes = [_bm_bar_regimes[j + req.lookback - 1] for j in range(len(X))]
    else:
        _bm_window_regimes = []

    _MIN_BM_SAMPLES = 10  # minimum native-regime windows to use filtered set
    if _bm_window_regimes:
        # First pass: windows whose last lookback bar is in the native regime.
        _native_idx = [i for i, r in enumerate(_bm_window_regimes[:split]) if r == native]

        # Second pass: require the regime to persist into the forecast horizon.
        # Near-bottom bear windows flip to bull within a few bars — those teach
        # the model to predict recoveries, which is the wrong signal.
        # Keep only windows where ≥50% of the next `horizon` bars stay native.
        def _persistent(idx: int) -> bool:
            bar_end = idx + req.lookback - 1   # last bar of the lookback window
            fwd = [
                _bm_bar_regimes[bar_end + k]
                for k in range(1, req.horizon + 1)
                if bar_end + k < len(_bm_bar_regimes)
            ]
            if not fwd:
                return False
            return sum(1 for b in fwd if b == native) / len(fwd) >= 0.5

        _persistent_idx = [i for i in _native_idx if _persistent(i)]

        # Use persistence-filtered set if enough samples; fall back progressively.
        if len(_persistent_idx) >= _MIN_BM_SAMPLES:
            _train_idx = _persistent_idx
        elif len(_native_idx) >= _MIN_BM_SAMPLES:
            _train_idx = _native_idx          # persistence filter too aggressive
        else:
            _train_idx = None                 # regime too rare — use full set

        if _train_idx is not None:
            X_bm_train = X[:split][_train_idx]
            y_bm_train = y[:split][_train_idx]
        else:
            X_bm_train, y_bm_train = X[:split], y[:split]
    else:
        X_bm_train, y_bm_train = X[:split], y[:split]

    # ── Helper: run one seed ───────────────────────────────────────────────
    def _one_sim(seed: int):
        import random as _r
        import numpy as _np
        _r.seed(seed); _np.random.seed(seed)
        try:
            import torch as _t
            _t.manual_seed(seed)
            _t.backends.cudnn.deterministic = True
        except Exception:
            pass

        _model = model_cls(horizon=req.horizon, n_features=n_features)

        if req.model_name == "OnlineModel":
            # Two-pass training for River ARF:
            #
            # Pass 1 — recent general history: Hoeffding trees need enough
            # observations to split (grace_period=50).  Regime-filtered data
            # alone gives ~50–80 windows which leaves every tree as an unsplit
            # stump predicting the global mean.  Using recent general history
            # teaches the ARF the feature→return mapping across all conditions.
            # Capped at the most-recent 400 windows to bound training time —
            # recent data is most relevant for the current test period anyway.
            #
            # Pass 2 — native-regime windows: ADWIN detects the distribution
            # shift and replaces stale trees, specialising the forest for the
            # native bear regime without discarding the structural knowledge
            # from pass 1.
            # Two-pass caps — both bounded to keep total training under ~18s:
            #   Pass 1: 200 samples × ~30ms = 6s   (general structure)
            #   Pass 2: 200 samples × ~30ms = 6s   (regime specialisation)
            # At n_models=15, learn_one costs ~30ms/sample (15 trees × 16 horizon).
            # Without these caps, a 10-year backtest produces ~300 native-regime
            # windows in pass 2 alone → 15s → combined 40s+ → socket hang up.
            _P1_MAX = 200
            _P2_MAX = 200
            _p1_X   = X[:split][-_P1_MAX:]   # most recent ≤200 training windows
            _p1_y   = y[:split][-_P1_MAX:]
            _model.fit(_p1_X, _p1_y)                  # pass 1: general (capped)
            if len(X_bm_train) >= 10:
                _p2_X = X_bm_train[-_P2_MAX:]         # most recent ≤200 native windows
                _p2_y = y_bm_train[-_P2_MAX:]
                _model.fit(_p2_X, _p2_y)              # pass 2: regime specialisation
        else:
            _model.fit(X_bm_train, y_bm_train)

        _r.seed(seed)  # re-seed broker randomness after training
        if preset == "zero_cost":       _b = SimulatedBroker.zero_cost()
        elif preset == "institutional": _b = SimulatedBroker.institutional()
        else:                           _b = SimulatedBroker.retail()

        _be = BacktestEngine(broker=_b, initial_capital=req.initial_capital)

        # Regime default long: if the caller provided an explicit list, use it.
        # Otherwise auto-enable for TCNModel (the LOW_VOL_BULL specialist) so it
        # defaults to long within its native regime instead of sitting in cash
        # during the smooth, low-signal uptrend bars it was trained on.
        if req.regime_default_long is not None:
            _rdl: Optional[set] = set(req.regime_default_long)
        elif native == "LOW_VOL_BULL":
            _rdl = {"LOW_VOL_BULL"}
        else:
            _rdl = None

        _result = _be.run_model(
            req.ticker, test_df, _model,
            req.lookback, req.horizon,
            position_size             = req.position_size,
            signal_hold               = req.signal_hold,
            long_only                 = req.long_only,
            signal_threshold          = req.signal_threshold,
            min_confidence            = req.min_confidence,
            stop_loss_pct             = req.stop_loss_pct,
            regime_signal_thresholds  = req.regime_signal_thresholds,
            native_regime             = native,   # only trade in native regime
            regime_default_long       = _rdl,     # buy-and-hold in designated regimes
        )
        return _result, _be.cost_summary()

    # ── Run N simulations ──────────────────────────────────────────────────
    # OnlineModel (River ARF) is deterministic: its seeds are fixed inside
    # _build_models() and are not affected by the numpy/torch random state
    # seeded in _one_sim.  Running N seeds gives N identical results at N×
    # the training cost.  Cap at 1 to avoid the unnecessary work.
    n = 1 if req.model_name == "OnlineModel" else req.n_simulations
    try:
        with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
            runs = list(pool.map(_one_sim, range(n)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")

    all_results = [r for r, _ in runs]
    cs_raw      = runs[0][1]

    # ── Per-regime breakdown — averaged across seeds ───────────────────────
    def _regime_metrics_for(result) -> dict[str, dict]:
        eq  = result.equity
        rc  = result.regime_curve
        out = {}
        for r in Regime:
            rname = r.name
            bars  = [i for i, c in enumerate(rc) if c == rname and i > 0]
            if not bars:
                out[rname] = {"sharpe": 0.0, "total_return": 0.0,
                              "max_drawdown": 0.0, "win_rate": 0.0, "n_bars": 0}
                continue
            rets = np.array([(eq[i] - eq[i-1]) / eq[i-1] if eq[i-1] > 0 else 0.0 for i in bars])
            vol  = float(np.std(rets))
            peak, max_dd = eq[bars[0]], 0.0
            for i in bars:
                if eq[i] > peak: peak = eq[i]
                dd = (peak - eq[i]) / peak if peak > 0 else 0.0
                if dd > max_dd: max_dd = dd
            # Win rate: fraction of ACTIVE bars (non-flat) with positive equity
            # change.  Flat bars (model in cash, eq[i] == eq[i-1]) are excluded
            # because they are not trades — counting them as "losses" massively
            # deflates the metric when the model sits out most of a regime.
            _active = rets[np.abs(rets) > 1e-10]
            _wr = float(np.mean(_active > 0)) if len(_active) > 0 else 0.0
            out[rname] = {
                "sharpe":       float(np.mean(rets) / vol * np.sqrt(252)) if vol > 1e-10 else 0.0,
                "total_return": float(np.prod(1.0 + rets) - 1.0),
                "max_drawdown": max_dd,
                "win_rate":     _wr,
                "n_bars":       len(bars),
            }
        return out

    per_seed_regime = [_regime_metrics_for(r) for r in all_results]

    regime_breakdown: list[RegimeMetrics] = []
    for r in Regime:
        rname  = r.name
        values = [s[rname] for s in per_seed_regime]
        n_bars = int(np.mean([v["n_bars"] for v in values]))
        sharpes = np.array([v["sharpe"] for v in values])
        returns = np.array([v["total_return"] for v in values])
        regime_breakdown.append(RegimeMetrics(
            regime           = rname,
            is_native        = (rname == native),
            n_bars           = n_bars,
            sharpe           = round(float(np.mean(sharpes)), 4),
            sharpe_std       = round(float(np.std(sharpes)), 4) if n > 1 else 0.0,
            total_return     = round(float(np.mean(returns)), 6),
            total_return_std = round(float(np.std(returns)), 6) if n > 1 else 0.0,
            max_drawdown     = round(float(np.mean([v["max_drawdown"] for v in values])), 6),
            win_rate         = round(float(np.mean([v["win_rate"] for v in values])), 4),
        ))

    # ── Overall MC aggregate (mirrors /backtest logic) ─────────────────────
    def _arr(attr): return np.array([getattr(r, attr) for r in all_results])

    sim_stats = None
    if n > 1:
        min_len  = min(len(r.equity) for r in all_results)
        eq_mat   = np.array([r.equity[:min_len] for r in all_results])
        sim_stats = SimulationStats(
            n_simulations     = n,
            sharpe_mean       = float(np.mean(_arr("sharpe"))),
            sharpe_std        = float(np.std(_arr("sharpe"))),
            total_return_mean = float(np.mean(_arr("total_return"))),
            total_return_std  = float(np.std(_arr("total_return"))),
            max_drawdown_mean = float(np.mean(_arr("max_drawdown"))),
            max_drawdown_std  = float(np.std(_arr("max_drawdown"))),
            win_rate_mean     = float(np.mean(_arr("win_rate"))),
            win_rate_std      = float(np.std(_arr("win_rate"))),
            alpha_mean        = float(np.mean(_arr("alpha"))),
            alpha_std         = float(np.std(_arr("alpha"))),
            beta_mean         = float(np.mean(_arr("beta"))),
            beta_std          = float(np.std(_arr("beta"))),
            equity_mean       = eq_mat.mean(axis=0).tolist(),
            equity_p10        = np.percentile(eq_mat, 10, axis=0).tolist(),
            equity_p90        = np.percentile(eq_mat, 90, axis=0).tolist(),
        )
        # Representative run: median Sharpe
        sharpes = _arr("sharpe")
        result  = all_results[int(np.argsort(sharpes)[len(sharpes) // 2])]
        equity  = sim_stats.equity_mean
    else:
        result = all_results[0]
        equity = result.equity

    cs = CostSummary(
        n_fills          = cs_raw.get("n_fills", 0),
        total_commission = cs_raw.get("total_commission", 0.0),
        total_slippage   = cs_raw.get("total_slippage", 0.0),
        total_spread     = cs_raw.get("total_spread", 0.0),
        total_friction   = cs_raw.get("total_friction", 0.0),
        avg_latency_ms   = cs_raw.get("avg_latency_ms", 0.0),
    )

    overall_sharpe  = sim_stats.sharpe_mean       if sim_stats else result.sharpe
    overall_return  = sim_stats.total_return_mean if sim_stats else result.total_return
    overall_mdd     = sim_stats.max_drawdown_mean if sim_stats else result.max_drawdown
    overall_wr      = sim_stats.win_rate_mean     if sim_stats else result.win_rate

    return BenchmarkResponse(
        ticker               = req.ticker,
        model_name           = req.model_name,
        native_regime        = native,
        overall_sharpe       = overall_sharpe,
        overall_sortino      = result.sortino,
        overall_return       = overall_return,
        overall_max_drawdown = overall_mdd,
        overall_win_rate     = overall_wr,
        overall_calmar       = result.calmar,
        n_trades             = result.n_trades,
        alpha                = sim_stats.alpha_mean if sim_stats else result.alpha,
        beta                 = sim_stats.beta_mean  if sim_stats else result.beta,
        cost_summary         = cs,
        equity               = equity,
        regime_curve         = result.regime_curve,
        regime_breakdown     = regime_breakdown,
        simulation_stats     = sim_stats,
        bah_equity           = result.bah_equity,
        bah_return           = result.bah_return,
        notes                = result.notes + (f" | {n} MC seeds" if sim_stats else ""),
        timestamp            = datetime.now(timezone.utc),
    )
