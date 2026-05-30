"""Tests for FillListener — Binance userData stream position close handler."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from src.execution.fill_listener import FillListener


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE positions (
            position_id   TEXT PRIMARY KEY,
            symbol        TEXT NOT NULL,
            side          TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'open',
            entry_price   TEXT,
            exit_price    TEXT,
            quantity      TEXT,
            close_reason  TEXT,
            closed_at     TEXT,
            realized_pnl  TEXT DEFAULT '0',
            leverage      INTEGER DEFAULT 1,
            trading_mode  TEXT DEFAULT 'testnet'
        )
    """)
    conn.execute("""
        CREATE TABLE orders (
            order_id          TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4)))),
            binance_order_id  INTEGER,
            symbol            TEXT NOT NULL,
            side              TEXT NOT NULL DEFAULT 'buy',
            position_side     TEXT NOT NULL DEFAULT 'both',
            order_type        TEXT NOT NULL DEFAULT 'limit',
            price             TEXT,
            quantity          TEXT NOT NULL DEFAULT '0',
            filled_qty        TEXT NOT NULL DEFAULT '0',
            avg_fill_price    TEXT,
            status            TEXT NOT NULL DEFAULT 'open',
            fee               TEXT NOT NULL DEFAULT '0',
            fee_asset         TEXT,
            trading_mode      TEXT NOT NULL DEFAULT 'testnet',
            created_at        TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at        TEXT
        )
    """)
    conn.commit()
    return conn


def _insert_order(conn: sqlite3.Connection, binance_id: int, symbol: str = "BTCUSDT") -> None:
    conn.execute(
        "INSERT INTO orders (order_id, binance_order_id, symbol, quantity) VALUES (?,?,?,?)",
        (f"ord-{binance_id}", binance_id, symbol, "0.01"),
    )
    conn.commit()


def _insert_open(conn: sqlite3.Connection, symbol: str) -> None:
    conn.execute(
        "INSERT INTO positions (position_id, symbol, side, status) VALUES (?,?,?,?)",
        (f"pos-{symbol}", symbol, "long", "open"),
    )
    conn.commit()


def _make_listener(conn=None, telegram=None) -> FillListener:
    if conn is None:
        conn = _make_db()
    exchange = MagicMock()
    return FillListener(exchange=exchange, conn=conn, telegram_bot=telegram)


def _fill_msg(
    symbol: str,
    order_type: str = "STOP_MARKET",
    reduce_only: bool = False,
    avg_price: str = "50000.0",
    execution_type: str = "TRADE",
    order_status: str = "FILLED",
    binance_order_id: int | None = None,
    order_side: str = "SELL",
    position_side: str = "LONG",
    last_fill_qty: str = "0.01",
) -> dict:
    return {
        "e": "ORDER_TRADE_UPDATE",
        "o": {
            "s": symbol,
            "o": order_type,
            "x": execution_type,
            "X": order_status,
            "R": reduce_only,
            "ap": avg_price,
            "i": binance_order_id,
            "S": order_side,
            "ps": position_side,
            "l": last_fill_qty,
        },
    }


# ---------------------------------------------------------------------------
# TestHandleIgnores — messages that should NOT trigger a close
# ---------------------------------------------------------------------------

class TestHandleIgnores:
    def test_non_trade_event_ignored(self):
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        fl = _make_listener(conn)
        fl._handle({"e": "ACCOUNT_UPDATE", "o": {"s": "BTCUSDT", "x": "TRADE", "X": "FILLED", "R": True}})
        row = conn.execute("SELECT status FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "open"

    def test_partial_fill_ignored(self):
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        fl = _make_listener(conn)
        fl._handle(_fill_msg("BTCUSDT", order_status="PARTIALLY_FILLED", reduce_only=True))
        row = conn.execute("SELECT status FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "open"

    def test_entry_order_ignored(self):
        """Market entry BUY+LONG (opening long) must not close position."""
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        fl = _make_listener(conn)
        # BUY+LONG = opening a long position, not closing
        fl._handle(_fill_msg("BTCUSDT", order_type="MARKET", reduce_only=False,
                              order_side="BUY", position_side="LONG"))
        row = conn.execute("SELECT status FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "open"


# ---------------------------------------------------------------------------
# TestMarkClosed — SL / TP / liquidation fills
# ---------------------------------------------------------------------------

class TestMarkClosed:
    def test_sl_fill_closes_position(self):
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        fl = _make_listener(conn)
        fl._handle(_fill_msg("BTCUSDT", order_type="STOP_MARKET", avg_price="48000.0"))
        row = conn.execute("SELECT * FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "closed"
        assert row["close_reason"] == "sl_hit"
        assert row["exit_price"] == "48000.0"
        assert row["closed_at"] is not None

    def test_tp_fill_closes_position(self):
        conn = _make_db()
        _insert_open(conn, "ETHUSDT")
        fl = _make_listener(conn)
        fl._handle(_fill_msg("ETHUSDT", order_type="TAKE_PROFIT_MARKET", avg_price="3200.0"))
        row = conn.execute("SELECT * FROM positions WHERE symbol='ETHUSDT'").fetchone()
        assert row["status"] == "closed"
        assert row["close_reason"] == "tp_hit"
        assert row["exit_price"] == "3200.0"

    def test_liquidation_fill_closes_position(self):
        conn = _make_db()
        _insert_open(conn, "SOLUSDT")
        fl = _make_listener(conn)
        fl._handle(_fill_msg("SOLUSDT", order_type="LIQUIDATION", avg_price="100.0"))
        row = conn.execute("SELECT * FROM positions WHERE symbol='SOLUSDT'").fetchone()
        assert row["status"] == "closed"
        assert row["close_reason"] == "liquidated"

    def test_reduce_only_market_close(self):
        """reduceOnly MARKET fill (strategy_exit) → market_close reason."""
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        fl = _make_listener(conn)
        fl._handle(_fill_msg("BTCUSDT", order_type="MARKET", reduce_only=True, avg_price="49000.0"))
        row = conn.execute("SELECT * FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "closed"
        assert row["close_reason"] == "market_close"

    def test_no_open_position_is_noop(self):
        """Fill for symbol with no open DB position → no crash, no rows changed."""
        conn = _make_db()
        fl = _make_listener(conn)
        fl._handle(_fill_msg("BNBUSDT", order_type="STOP_MARKET"))
        count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        assert count == 0

    def test_only_open_position_closed(self):
        """Fill must not touch already-closed positions for same symbol."""
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        conn.execute(
            "INSERT INTO positions (position_id, symbol, side, status) VALUES ('old','BTCUSDT','short','closed')"
        )
        conn.commit()
        fl = _make_listener(conn)
        fl._handle(_fill_msg("BTCUSDT", order_type="STOP_MARKET"))
        rows = conn.execute("SELECT status FROM positions WHERE symbol='BTCUSDT' ORDER BY position_id").fetchall()
        statuses = [r["status"] for r in rows]
        assert "closed" in statuses
        # old closed row still closed
        old = conn.execute("SELECT status FROM positions WHERE position_id='old'").fetchone()
        assert old["status"] == "closed"


# ---------------------------------------------------------------------------
# TestTelegramNotify
# ---------------------------------------------------------------------------

class TestTelegramNotify:
    def test_telegram_notified_on_sl(self):
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        telegram = MagicMock()
        fl = _make_listener(conn, telegram=telegram)
        fl._handle(_fill_msg("BTCUSDT", order_type="STOP_MARKET", avg_price="48000.0"))
        telegram.send_warning.assert_called_once()
        msg = telegram.send_warning.call_args[0][0]
        assert "BTCUSDT" in msg
        assert "SL" in msg

    def test_telegram_notified_on_tp(self):
        conn = _make_db()
        _insert_open(conn, "ETHUSDT")
        telegram = MagicMock()
        fl = _make_listener(conn, telegram=telegram)
        fl._handle(_fill_msg("ETHUSDT", order_type="TAKE_PROFIT_MARKET", avg_price="3200.0"))
        telegram.send_warning.assert_called_once()
        msg = telegram.send_warning.call_args[0][0]
        assert "TP" in msg

    def test_no_telegram_no_crash(self):
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        fl = _make_listener(conn, telegram=None)
        fl._handle(_fill_msg("BTCUSDT", order_type="STOP_MARKET"))
        row = conn.execute("SELECT status FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "closed"

    def test_telegram_failure_does_not_raise(self):
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        telegram = MagicMock()
        telegram.send_warning.side_effect = Exception("network error")
        fl = _make_listener(conn, telegram=telegram)
        fl._handle(_fill_msg("BTCUSDT", order_type="STOP_MARKET"))
        row = conn.execute("SELECT status FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "closed"


# ---------------------------------------------------------------------------
# TestListenKeyManagement
# ---------------------------------------------------------------------------

class TestListenKeyManagement:
    def test_create_listen_key(self):
        fl = _make_listener()
        fl._exchange.fapiPrivatePostListenKey.return_value = {"listenKey": "abc123"}
        key = fl._create_listen_key()
        assert key == "abc123"
        fl._exchange.fapiPrivatePostListenKey.assert_called_once()

    def test_put_listen_key(self):
        fl = _make_listener()
        fl._put_listen_key("abc123")
        fl._exchange.fapiPrivatePutListenKey.assert_called_once_with({"listenKey": "abc123"})


# ---------------------------------------------------------------------------
# TestOrderDbSync — orders table updated on FILLED / CANCELED events
# ---------------------------------------------------------------------------

class TestOrderDbSync:
    def test_canceled_event_updates_order_status(self):
        conn = _make_db()
        _insert_order(conn, binance_id=99)
        fl = _make_listener(conn)
        fl._handle(_fill_msg("BTCUSDT", execution_type="CANCELED", order_status="CANCELED", binance_order_id=99))
        row = conn.execute("SELECT status FROM orders WHERE binance_order_id=99").fetchone()
        assert row["status"] == "canceled"

    def test_canceled_event_does_not_close_position(self):
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        _insert_order(conn, binance_id=99)
        fl = _make_listener(conn)
        fl._handle(_fill_msg("BTCUSDT", execution_type="CANCELED", order_status="CANCELED", binance_order_id=99))
        row = conn.execute("SELECT status FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "open"

    def test_sl_fill_updates_order_status_to_filled(self):
        conn = _make_db()
        _insert_open(conn, "BTCUSDT")
        _insert_order(conn, binance_id=42)
        fl = _make_listener(conn)
        fl._handle(_fill_msg("BTCUSDT", order_type="STOP_MARKET", binance_order_id=42))
        row = conn.execute("SELECT status FROM orders WHERE binance_order_id=42").fetchone()
        assert row["status"] == "filled"

    def test_canceled_unknown_order_id_no_crash(self):
        conn = _make_db()
        fl = _make_listener(conn)
        fl._handle(_fill_msg("BTCUSDT", execution_type="CANCELED", order_status="CANCELED", binance_order_id=9999))
        # No rows, no crash


# ---------------------------------------------------------------------------
# TestTP1PartialClose — TP1/TP2 split logic
# ---------------------------------------------------------------------------

def _insert_open_with_qty(conn: sqlite3.Connection, symbol: str, qty: float = 1.0, side: str = "long") -> None:
    conn.execute(
        "INSERT INTO positions (position_id, symbol, side, status, quantity, entry_price)"
        " VALUES (?,?,?,?,?,?)",
        (f"pos-{symbol}", symbol, side, "open", str(qty), "50000.0"),
    )
    conn.commit()


def _insert_limit_order(conn: sqlite3.Connection, binance_id: int, symbol: str = "BTCUSDT") -> None:
    conn.execute(
        "INSERT INTO orders (order_id, binance_order_id, symbol, order_type, quantity)"
        " VALUES (?,?,?,?,?)",
        (f"ord-{binance_id}", binance_id, symbol, "limit", "0.5"),
    )
    conn.commit()


class TestTP1PartialClose:
    def test_tp1_reduces_position_qty(self):
        """TP1 fill with TP2 still pending → partial close, position stays open."""
        conn = _make_db()
        _insert_open_with_qty(conn, "BTCUSDT", qty=1.0)
        _insert_limit_order(conn, 101, "BTCUSDT")  # TP1
        _insert_limit_order(conn, 102, "BTCUSDT")  # TP2 still open
        fl = _make_listener(conn)
        fl._handle(_fill_msg(
            "BTCUSDT", order_type="LIMIT", avg_price="52000.0",
            last_fill_qty="0.5", binance_order_id=101,
        ))
        row = conn.execute("SELECT status, quantity FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "open"
        assert abs(float(row["quantity"]) - 0.5) < 0.0001

    def test_tp2_closes_position_fully(self):
        """TP2 fill with no more limit orders → full close."""
        conn = _make_db()
        _insert_open_with_qty(conn, "BTCUSDT", qty=0.5)
        _insert_limit_order(conn, 102, "BTCUSDT")  # TP2 (only one left)
        fl = _make_listener(conn)
        fl._handle(_fill_msg(
            "BTCUSDT", order_type="LIMIT", avg_price="55000.0",
            last_fill_qty="0.5", binance_order_id=102,
        ))
        row = conn.execute("SELECT status, close_reason FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "closed"
        assert row["close_reason"] == "tp_hit"

    def test_single_tp_closes_position_fully(self):
        """Single TP (no tp2 configured) fills → full close."""
        conn = _make_db()
        _insert_open_with_qty(conn, "BTCUSDT", qty=1.0)
        _insert_limit_order(conn, 200, "BTCUSDT")
        fl = _make_listener(conn)
        fl._handle(_fill_msg(
            "BTCUSDT", order_type="LIMIT", avg_price="53000.0",
            last_fill_qty="1.0", binance_order_id=200,
        ))
        row = conn.execute("SELECT status FROM positions WHERE symbol='BTCUSDT'").fetchone()
        assert row["status"] == "closed"

    def test_short_tp1_partial_close(self):
        """Short position TP1: BUY+SHORT → partial close."""
        conn = _make_db()
        _insert_open_with_qty(conn, "ETHUSDT", qty=2.0, side="short")
        _insert_limit_order(conn, 301, "ETHUSDT")  # TP1
        _insert_limit_order(conn, 302, "ETHUSDT")  # TP2 still open
        fl = _make_listener(conn)
        fl._handle(_fill_msg(
            "ETHUSDT", order_type="LIMIT", avg_price="2000.0",
            order_side="BUY", position_side="SHORT",
            last_fill_qty="1.0", binance_order_id=301,
        ))
        row = conn.execute("SELECT status, quantity FROM positions WHERE symbol='ETHUSDT'").fetchone()
        assert row["status"] == "open"
        assert abs(float(row["quantity"]) - 1.0) < 0.0001

    def test_tp1_telegram_notified(self):
        conn = _make_db()
        _insert_open_with_qty(conn, "BTCUSDT", qty=1.0)
        _insert_limit_order(conn, 401, "BTCUSDT")
        _insert_limit_order(conn, 402, "BTCUSDT")
        telegram = MagicMock()
        fl = _make_listener(conn, telegram=telegram)
        fl._handle(_fill_msg(
            "BTCUSDT", order_type="LIMIT", avg_price="52000.0",
            last_fill_qty="0.5", binance_order_id=401,
        ))
        telegram.send_warning.assert_called_once()
        assert "TP1" in telegram.send_warning.call_args[0][0]
