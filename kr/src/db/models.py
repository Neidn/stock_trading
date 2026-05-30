"""Common CRUD operations for the trading database.

All functions accept an open :class:`sqlite3.Connection` (obtained via
:func:`src.db.connection.get_connection`) and operate within that connection's
transaction scope.  Callers are responsible for committing or rolling back.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Execute all migration SQL files against *db_path* in order.

    Safe to call multiple times.  ``CREATE TABLE IF NOT EXISTS`` guards DDL in
    001_init.sql.  ``ALTER TABLE ADD COLUMN`` (used in later migrations) raises
    ``OperationalError: duplicate column name`` when the column already exists —
    that error is silently ignored so restarts are idempotent.

    Args:
        db_path: Path to the SQLite database file.
    """
    import logging
    from src.db.connection import get_connection  # avoid circular at module level

    logger = logging.getLogger(__name__)
    conn = get_connection(db_path)
    for sql_file in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        try:
            conn.executescript(sql_file.read_text(encoding="utf-8"))
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc):
                logger.debug("Migration %s already applied (%s) — skipping", sql_file.name, exc)
            else:
                raise
    conn.commit()


# ---------------------------------------------------------------------------
# Klines
# ---------------------------------------------------------------------------

def insert_kline(conn: sqlite3.Connection, kline: dict) -> None:
    """Insert or replace a single OHLCV candle.

    Args:
        conn: Active database connection.
        kline: Dict with keys: symbol, interval_type, open_time, open, high,
               low, close, volume, close_time.  ``id`` is auto-generated if absent.
    """
    kline.setdefault(
        "id", f"{kline['symbol']}_{kline['interval_type']}_{kline['open_time']}"
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO klines
            (id, symbol, interval_type, open_time, open, high, low, close, volume, close_time)
        VALUES
            (:id, :symbol, :interval_type, :open_time, :open, :high, :low, :close, :volume, :close_time)
        """,
        kline,
    )


def get_klines(
    conn: sqlite3.Connection,
    symbol: str,
    interval: str,
    limit: int = 200,
) -> list[sqlite3.Row]:
    """Return the most recent *limit* candles for *symbol* / *interval*, ascending.

    Args:
        conn: Active database connection.
        symbol: Trading pair, e.g. ``'BTCUSDT'``.
        interval: Candle interval, e.g. ``'15m'``.
        limit: Maximum number of rows to return.

    Returns:
        List of :class:`sqlite3.Row` objects ordered oldest → newest.
    """
    rows = conn.execute(
        """
        SELECT * FROM klines
        WHERE symbol = ? AND interval_type = ?
        ORDER BY open_time DESC
        LIMIT ?
        """,
        (symbol, interval, limit),
    ).fetchall()
    return list(reversed(rows))


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def insert_signal(conn: sqlite3.Connection, signal: dict) -> None:
    """Persist a trading signal record.

    Args:
        conn: Active database connection.
        signal: Dict with keys matching the ``signals`` table columns.
                ``signal_id`` is auto-generated (UUID4) if absent.
                ``indicators_json`` may be a dict — will be serialised.
    """
    signal.setdefault("signal_id", str(uuid.uuid4()))
    if isinstance(signal.get("indicators_json"), dict):
        signal["indicators_json"] = json.dumps(signal["indicators_json"])
    conn.execute(
        """
        INSERT INTO signals
            (signal_id, symbol, signal_type, strategy_name, strength_score,
             entry_price, tp_price, sl_price, indicators_json, blocked, block_reason)
        VALUES
            (:signal_id, :symbol, :signal_type, :strategy_name, :strength_score,
             :entry_price, :tp_price, :sl_price, :indicators_json,
             :blocked, :block_reason)
        """,
        {
            "signal_id":      signal["signal_id"],
            "symbol":         signal["symbol"],
            "signal_type":    signal["signal_type"],
            "strategy_name":  signal["strategy_name"],
            "strength_score": signal["strength_score"],
            "entry_price":    signal.get("entry_price"),
            "tp_price":       signal.get("tp_price"),
            "sl_price":       signal.get("sl_price"),
            "indicators_json": signal.get("indicators_json"),
            "blocked":        int(signal.get("blocked", False)),
            "block_reason":   signal.get("block_reason"),
        },
    )


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def insert_order(conn: sqlite3.Connection, order: dict) -> None:
    """Persist an order record.

    Args:
        conn: Active database connection.
        order: Dict with keys matching the ``orders`` table columns.
               ``order_id`` is auto-generated if absent.
    """
    order.setdefault("order_id", str(uuid.uuid4()))
    conn.execute(
        """
        INSERT INTO orders
            (order_id, broker_order_id, symbol, side, position_side, order_type,
             price, quantity, filled_qty, avg_fill_price, status, signal_id,
             fee, fee_asset, trading_mode, updated_at)
        VALUES
            (:order_id, :broker_order_id, :symbol, :side, :position_side, :order_type,
             :price, :quantity, :filled_qty, :avg_fill_price, :status, :signal_id,
             :fee, :fee_asset, :trading_mode, :updated_at)
        """,
        {
            "order_id":       order["order_id"],
            "broker_order_id": order.get("broker_order_id"),
            "symbol":         order["symbol"],
            "side":           order["side"],
            "position_side":  order.get("position_side", "both"),  # KRX spot = always 'both'
            "order_type":     order["order_type"],
            "price":          order.get("price"),
            "quantity":       order["quantity"],
            "filled_qty":     order.get("filled_qty", "0"),
            "avg_fill_price": order.get("avg_fill_price"),
            "status":         order["status"],
            "signal_id":      order.get("signal_id"),
            "fee":            order.get("fee", "0"),
            "fee_asset":      order.get("fee_asset"),
            "trading_mode":   order.get("trading_mode", "paper"),
            "updated_at":     order.get("updated_at"),
        },
    )


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def upsert_position(conn: sqlite3.Connection, position: dict) -> None:
    """Insert or replace a position record (KRX spot, long-only).

    Args:
        conn: Active database connection.
        position: Dict with keys matching the ``positions`` table columns.
                  ``position_id`` is auto-generated if absent.
    """
    position.setdefault("position_id", str(uuid.uuid4()))
    conn.execute(
        """
        INSERT OR REPLACE INTO positions
            (position_id, symbol, side, leverage, entry_price, exit_price,
             quantity, liquidation_price, stop_loss, take_profit_1, take_profit_2,
             initial_stop_loss, trailing_activated, realized_pnl, unrealized_pnl,
             status, close_reason, trading_mode, strategy_name, opened_at, closed_at,
             fill_price, slippage_bps, market, tax_paid, t2_settle_date)
        VALUES
            (:position_id, :symbol, :side, :leverage, :entry_price, :exit_price,
             :quantity, :liquidation_price, :stop_loss, :take_profit_1, :take_profit_2,
             :initial_stop_loss, :trailing_activated, :realized_pnl, :unrealized_pnl,
             :status, :close_reason, :trading_mode, :strategy_name, :opened_at, :closed_at,
             :fill_price, :slippage_bps, :market, :tax_paid, :t2_settle_date)
        """,
        {
            "position_id":       position["position_id"],
            "symbol":            position["symbol"],
            "side":              position.get("side", "long"),   # KRX spot always long
            "leverage":          position.get("leverage", 1),    # spot = 1x
            "entry_price":       position["entry_price"],
            "exit_price":        position.get("exit_price"),
            "quantity":          position["quantity"],
            "liquidation_price": position.get("liquidation_price", "0"),  # N/A for spot
            "stop_loss":         position["stop_loss"],
            "take_profit_1":     position.get("take_profit_1"),
            "take_profit_2":     position.get("take_profit_2"),
            "initial_stop_loss": position.get("initial_stop_loss", position["stop_loss"]),
            "trailing_activated": int(position.get("trailing_activated", 0)),
            "realized_pnl":      position.get("realized_pnl", "0"),
            "unrealized_pnl":    position.get("unrealized_pnl", "0"),
            "status":            position.get("status", "open"),
            "close_reason":      position.get("close_reason"),
            "trading_mode":      position.get("trading_mode", "paper"),
            "strategy_name":     position.get("strategy_name"),
            "opened_at":         position.get("opened_at", datetime.now(timezone.utc).isoformat()),
            "closed_at":         position.get("closed_at"),
            "fill_price":        position.get("fill_price"),
            "slippage_bps":      position.get("slippage_bps"),
            "market":            position.get("market", "KOSPI"),
            "tax_paid":          position.get("tax_paid"),
            "t2_settle_date":    position.get("t2_settle_date"),
        },
    )


def insert_position(conn: sqlite3.Connection, position: dict) -> None:
    """Insert a new position record. Alias for :func:`upsert_position`."""
    upsert_position(conn, position)


def get_open_positions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all positions with status='open'.

    Args:
        conn: Active database connection.
    """
    return conn.execute(
        "SELECT * FROM positions WHERE status = 'open' ORDER BY opened_at DESC"
    ).fetchall()


def has_open_position(conn: sqlite3.Connection, symbol: str) -> bool:
    """Return True if there is at least one open position for *symbol*.

    Args:
        conn: Active database connection.
        symbol: Trading pair, e.g. ``'BTCUSDT'``.
    """
    row = conn.execute(
        "SELECT 1 FROM positions WHERE symbol = ? AND status = 'open' LIMIT 1",
        (symbol,),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# System events
# ---------------------------------------------------------------------------

def log_system_event(
    conn: sqlite3.Connection,
    module: str,
    severity: str,
    message: str,
    metadata: dict | None = None,
    event_type: str = "general",
) -> None:
    """Append a structured log entry to system_events.

    Args:
        conn: Active database connection.
        module: Source module name, e.g. ``'strategy_runner'``.
        severity: One of ``'info'``, ``'warning'``, ``'error'``, ``'critical'``.
        message: Human-readable log message.
        metadata: Optional dict of extra context (serialised to JSON).
        event_type: Free-form category tag. Default ``'general'``.
    """
    conn.execute(
        """
        INSERT INTO system_events (event_id, event_type, severity, module, message, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            event_type,
            severity,
            module,
            message,
            json.dumps(metadata) if metadata else None,
        ),
    )
