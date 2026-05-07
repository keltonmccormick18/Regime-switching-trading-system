"""FastAPI dependency injection factories.

All singletons are created lazily with @lru_cache so they are shared across
requests.  If a backend is unavailable the 503 is raised at request time, not
at startup, so the API stays alive in partial-infrastructure environments.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import HTTPException


# ──────────────────────────── PostgresDB ────────────────────────────

@lru_cache(maxsize=1)
def _get_db_singleton():
    try:
        from src.storage.db import PostgresDB
        db = PostgresDB()
        db.health_check()
        return db
    except Exception as exc:
        return exc


def get_db():
    inst = _get_db_singleton()
    if isinstance(inst, Exception):
        raise HTTPException(
            status_code=503,
            detail=f"PostgreSQL unavailable: {inst}",
        )
    return inst


# ──────────────────────────── SignalCache ────────────────────────────

@lru_cache(maxsize=1)
def _get_cache_singleton():
    try:
        from src.storage.cache import SignalCache
        return SignalCache()
    except Exception as exc:
        return exc


def get_cache():
    inst = _get_cache_singleton()
    if isinstance(inst, Exception):
        raise HTTPException(
            status_code=503,
            detail=f"Redis unavailable: {inst}",
        )
    return inst


# ──────────────────────────── ArtifactStore ────────────────────────────

@lru_cache(maxsize=1)
def _get_artifacts_singleton():
    try:
        from src.storage.artifacts import ArtifactStore
        return ArtifactStore()
    except Exception as exc:
        return exc


def get_artifacts():
    inst = _get_artifacts_singleton()
    if isinstance(inst, Exception):
        raise HTTPException(
            status_code=503,
            detail=f"ArtifactStore unavailable: {inst}",
        )
    return inst


# ──────────────────────────── Optional variants ────────────────────────────
# These return None instead of raising, for endpoints that degrade gracefully.

def get_db_optional() -> Optional[object]:
    inst = _get_db_singleton()
    return None if isinstance(inst, Exception) else inst


def get_cache_optional() -> Optional[object]:
    inst = _get_cache_singleton()
    return None if isinstance(inst, Exception) else inst


def get_artifacts_optional() -> Optional[object]:
    inst = _get_artifacts_singleton()
    return None if isinstance(inst, Exception) else inst


# ──────────────────────────── PositionTracker ────────────────────────────

@lru_cache(maxsize=1)
def _get_tracker_singleton():
    try:
        from src.execution.paper_trader import PositionTracker
        return PositionTracker()
    except Exception as exc:
        return exc


def get_tracker():
    inst = _get_tracker_singleton()
    if isinstance(inst, Exception):
        raise HTTPException(
            status_code=503,
            detail=f"PositionTracker unavailable: {inst}",
        )
    return inst


# ──────────────────────────── StrategyEngine ────────────────────────────

@lru_cache(maxsize=1)
def _get_engine_singleton():
    try:
        from src.strategy.engine import StrategyEngine
        from src.strategy.position_sizer import SizingConfig, SizingMethod
        from src.strategy.risk_manager import RiskConfig
        import os

        sizing = SizingConfig(
            method           = SizingMethod.CONFIDENCE_WEIGHTED,
            vol_target       = float(os.getenv("STRATEGY_VOL_TARGET",   "0.15")),
            max_position_pct = float(os.getenv("STRATEGY_MAX_POS_PCT",  "0.20")),
            risk_fraction    = float(os.getenv("STRATEGY_RISK_FRACTION","0.02")),
        )
        risk = RiskConfig(
            max_drawdown_limit  = float(os.getenv("RISK_MAX_DD",         "0.15")),
            stop_loss_pct       = float(os.getenv("RISK_STOP_LOSS_PCT",  "0.05")),
            use_trailing_stop   = os.getenv("RISK_TRAILING_STOP", "true").lower() == "true",
            trailing_stop_pct   = float(os.getenv("RISK_TRAILING_PCT",  "0.07")),
            vol_target          = float(os.getenv("STRATEGY_VOL_TARGET", "0.15")),
            max_daily_loss_pct  = float(os.getenv("RISK_DAILY_LOSS_PCT", "0.03")),
        )
        capital = float(os.getenv("INITIAL_CAPITAL", "100000"))
        return StrategyEngine(
            initial_capital  = capital,
            sizing_config    = sizing,
            risk_config      = risk,
        )
    except Exception as exc:
        return exc


def get_engine():
    inst = _get_engine_singleton()
    if isinstance(inst, Exception):
        raise HTTPException(
            status_code=503,
            detail=f"StrategyEngine unavailable: {inst}",
        )
    return inst


def get_engine_optional() -> Optional[object]:
    inst = _get_engine_singleton()
    return None if isinstance(inst, Exception) else inst
