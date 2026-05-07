"""PostgreSQL storage for trades and performance metrics.

Two tables:
  trades  — every filled order (backtest, paper, or live).
  metrics — per-run summary statistics logged after each training / backtest cycle.

Connection string is read from the DATABASE_URL environment variable; falls back to
the default that matches docker-compose.yml (quant:quant@localhost:5432/quant).

Usage:
    db = PostgresDB()
    db.init_schema()          # run once at startup

    trade_id = db.insert_trade(Trade(
        symbol="SPY", side="buy", price=450.0, quantity=10,
        strategy="backtest", regime="low_vol_bull", model_used="TCNModel",
    ))

    metric_id = db.insert_metric(Metric(
        run_id="run-001", ticker="SPY",
        sharpe=1.4, total_return=0.22, max_drawdown=-0.08, win_rate=0.55, n_trades=87,
    ))

    recent = db.get_trades(symbol="SPY", limit=20)
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

import psycopg2
import psycopg2.extras

_DEFAULT_DSN = "postgresql://quant:quant@localhost:5432/quant"

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id          SERIAL       PRIMARY KEY,
    created_at  TIMESTAMPTZ  DEFAULT NOW(),
    symbol      TEXT         NOT NULL,
    side        TEXT         NOT NULL CHECK (side IN ('buy', 'sell')),
    price       FLOAT8       NOT NULL,
    quantity    FLOAT8       NOT NULL,
    strategy    TEXT,
    regime      TEXT,
    model_used  TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id           SERIAL       PRIMARY KEY,
    created_at   TIMESTAMPTZ  DEFAULT NOW(),
    run_id       TEXT         NOT NULL,
    ticker       TEXT         NOT NULL,
    interval     TEXT         DEFAULT '1d',
    start_date   DATE,
    end_date     DATE,
    regime       TEXT,
    model_used   TEXT,
    sharpe       FLOAT8,
    total_return FLOAT8,
    max_drawdown FLOAT8,
    win_rate     FLOAT8,
    n_trades     INTEGER,
    extra        JSONB
);

CREATE INDEX IF NOT EXISTS trades_symbol_ts_idx  ON trades  (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS metrics_run_id_idx    ON metrics (run_id);
CREATE INDEX IF NOT EXISTS metrics_ticker_ts_idx ON metrics (ticker, created_at DESC);
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    symbol:     str
    side:       str                   # "buy" | "sell"
    price:      float
    quantity:   float
    strategy:   str = ""              # "backtest" | "paper" | "live"
    regime:     str = ""              # Regime.value from regime.py
    model_used: str = ""
    notes:      str = ""
    id:         int | None = None
    created_at: datetime | None = None


@dataclass
class Metric:
    run_id:       str
    ticker:       str
    sharpe:       float
    total_return: float
    max_drawdown: float
    win_rate:     float
    n_trades:     int
    interval:     str = "1d"
    start_date:   str | None = None
    end_date:     str | None = None
    regime:       str = ""
    model_used:   str = ""
    extra:        dict = field(default_factory=dict)
    id:           int | None = None
    created_at:   datetime | None = None


# ---------------------------------------------------------------------------
# Database client
# ---------------------------------------------------------------------------

class PostgresDB:
    """Thin PostgreSQL client for trades and metrics storage.

    Each method opens a fresh connection, executes, commits, and closes.
    This is simple and correct for low-to-medium write rates; swap in a
    connection pool (e.g. psycopg2.pool.ThreadedConnectionPool) if needed.
    """

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ.get("DATABASE_URL", _DEFAULT_DSN)

    # --- Schema ---

    def init_schema(self) -> None:
        """Create tables and indexes if they do not already exist."""
        with self._cursor() as cur:
            cur.execute(_DDL)

    def health_check(self) -> None:
        """Raise if the database is unreachable or the schema is missing."""
        with self._cursor() as cur:
            cur.execute("SELECT 1")

    # --- Trades ---

    def insert_trade(self, trade: Trade) -> int:
        """Insert a trade record. Returns the new row id."""
        sql = """
            INSERT INTO trades (symbol, side, price, quantity, strategy, regime, model_used, notes)
            VALUES (%(symbol)s, %(side)s, %(price)s, %(quantity)s,
                    %(strategy)s, %(regime)s, %(model_used)s, %(notes)s)
            RETURNING id
        """
        with self._cursor() as cur:
            cur.execute(sql, {
                "symbol":     trade.symbol,
                "side":       trade.side,
                "price":      trade.price,
                "quantity":   trade.quantity,
                "strategy":   trade.strategy or None,
                "regime":     trade.regime   or None,
                "model_used": trade.model_used or None,
                "notes":      trade.notes    or None,
            })
            return cur.fetchone()["id"]

    def get_trades(
        self,
        symbol:   str | None = None,
        strategy: str | None = None,
        limit:    int = 100,
    ) -> list[Trade]:
        """Fetch recent trades, optionally filtered by symbol and/or strategy."""
        conditions, params = [], {}
        if symbol:
            conditions.append("symbol = %(symbol)s")
            params["symbol"] = symbol
        if strategy:
            conditions.append("strategy = %(strategy)s")
            params["strategy"] = strategy
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params["limit"] = limit
        sql = f"SELECT * FROM trades {where} ORDER BY created_at DESC LIMIT %(limit)s"
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [_row_to_trade(r) for r in cur.fetchall()]

    def delete_trades_before(self, before: datetime) -> int:
        """Hard-delete trades older than `before`. Returns number of rows deleted."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM trades WHERE created_at < %s", (before,))
            return cur.rowcount

    # --- Metrics ---

    def insert_metric(self, metric: Metric) -> int:
        """Insert a metrics record. Returns the new row id."""
        sql = """
            INSERT INTO metrics
                (run_id, ticker, interval, start_date, end_date, regime, model_used,
                 sharpe, total_return, max_drawdown, win_rate, n_trades, extra)
            VALUES
                (%(run_id)s, %(ticker)s, %(interval)s, %(start_date)s, %(end_date)s,
                 %(regime)s, %(model_used)s, %(sharpe)s, %(total_return)s,
                 %(max_drawdown)s, %(win_rate)s, %(n_trades)s, %(extra)s)
            RETURNING id
        """
        with self._cursor() as cur:
            cur.execute(sql, {
                "run_id":       metric.run_id,
                "ticker":       metric.ticker,
                "interval":     metric.interval,
                "start_date":   metric.start_date or None,
                "end_date":     metric.end_date   or None,
                "regime":       metric.regime     or None,
                "model_used":   metric.model_used or None,
                "sharpe":       metric.sharpe,
                "total_return": metric.total_return,
                "max_drawdown": metric.max_drawdown,
                "win_rate":     metric.win_rate,
                "n_trades":     metric.n_trades,
                "extra":        json.dumps(metric.extra) if metric.extra else None,
            })
            return cur.fetchone()["id"]

    def get_metrics(
        self,
        run_id: str | None = None,
        ticker: str | None = None,
        limit:  int = 50,
    ) -> list[Metric]:
        """Fetch metric records, optionally filtered by run_id and/or ticker."""
        conditions, params = [], {}
        if run_id:
            conditions.append("run_id = %(run_id)s")
            params["run_id"] = run_id
        if ticker:
            conditions.append("ticker = %(ticker)s")
            params["ticker"] = ticker
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params["limit"] = limit
        sql = f"SELECT * FROM metrics {where} ORDER BY created_at DESC LIMIT %(limit)s"
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [_row_to_metric(r) for r in cur.fetchall()]

    # --- Connection context manager ---

    @contextmanager
    def _cursor(self) -> Generator:
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Row → dataclass helpers
# ---------------------------------------------------------------------------

def _row_to_trade(row: dict) -> Trade:
    return Trade(
        id         = row["id"],
        created_at = row["created_at"],
        symbol     = row["symbol"],
        side       = row["side"],
        price      = row["price"],
        quantity   = row["quantity"],
        strategy   = row.get("strategy")   or "",
        regime     = row.get("regime")     or "",
        model_used = row.get("model_used") or "",
        notes      = row.get("notes")      or "",
    )


def _row_to_metric(row: dict) -> Metric:
    extra = row.get("extra") or {}
    if isinstance(extra, str):
        extra = json.loads(extra)
    return Metric(
        id           = row["id"],
        created_at   = row["created_at"],
        run_id       = row["run_id"],
        ticker       = row["ticker"],
        interval     = row.get("interval") or "1d",
        start_date   = str(row["start_date"])  if row.get("start_date")  else None,
        end_date     = str(row["end_date"])    if row.get("end_date")    else None,
        regime       = row.get("regime")       or "",
        model_used   = row.get("model_used")   or "",
        sharpe       = row.get("sharpe")       or 0.0,
        total_return = row.get("total_return") or 0.0,
        max_drawdown = row.get("max_drawdown") or 0.0,
        win_rate     = row.get("win_rate")     or 0.0,
        n_trades     = row.get("n_trades")     or 0,
        extra        = extra,
    )
