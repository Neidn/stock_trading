"""Unit tests for EmergencyHandler.

DB: in-memory SQLite with positions/orders/daily_performance schema.
order_manager and position_tracker are lightweight fakes or MagicMocks.
"""

from __future__ import annotations

import sqlite3
import uuid
import unittest
from unittest.mock import MagicMock, call

from src.safety.emergency_handler import EmergencyHandler
from src.safety.safe_mode import SafeMode
from src.execution.position_tracker import PositionTracker


# ---------------------------------------------------------------------------
# Schema / helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (symbol TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS positions (
    position_id     TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL CHECK (side IN ('long','short')),
    leverage        INTEGER NOT NULL DEFAULT 5,
    entry_price     TEXT NOT NULL,
    exit_price      TEXT,
    quantity        TEXT NOT NULL,
    liquidation_price TEXT NOT NULL DEFAULT '40000',
    stop_loss       TEXT NOT NULL,
    take_profit_1   TEXT,
    take_profit_2   TEXT,
    initial_stop_loss TEXT NOT NULL DEFAULT '48000',
    trailing_activated INTEGER DEFAULT 0,
    realized_pnl    TEXT DEFAULT '0',
    unrealized_pnl  TEXT DEFAULT '0',
    status          TEXT NOT NULL DEFAULT 'open',
    close_reason    TEXT,
    trading_mode    TEXT NOT NULL DEFAULT 'testnet',
    opened_at       TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at       TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    binance_order_id INTEGER,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    position_side   TEXT NOT NULL DEFAULT 'both',
    order_type      TEXT NOT NULL,
    price           TEXT,
    quantity        TEXT NOT NULL,
    filled_qty      TEXT NOT NULL DEFAULT '0',
    avg_fill_price  TEXT,
    status          TEXT NOT NULL DEFAULT 'open',
    fee             TEXT NOT NULL DEFAULT '0',
    fee_asset       TEXT,
    trading_mode    TEXT NOT NULL DEFAULT 'testnet',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT
);
CREATE TABLE IF NOT EXISTS daily_performance (
    perf_date       TEXT NOT NULL,
    trading_mode    TEXT NOT NULL DEFAULT 'testnet',
    total_trades    INTEGER DEFAULT 0,
    winning_trades  INTEGER DEFAULT 0,
    losing_trades   INTEGER DEFAULT 0,
    liquidated_trades INTEGER DEFAULT 0,
    gross_profit    TEXT DEFAULT '0',
    gross_loss      TEXT DEFAULT '0',
    net_pnl         TEXT DEFAULT '0',
    total_fees      TEXT DEFAULT '0',
    max_drawdown    TEXT DEFAULT '0',
    win_rate        TEXT DEFAULT '0',
    avg_liquidation_distance TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (perf_date, trading_mode)
);
CREATE TABLE IF NOT EXISTS safe_mode_events (
    event_id   TEXT PRIMARY KEY,
    action     TEXT NOT NULL CHECK (action IN ('activated','deactivated')),
    reason     TEXT NOT NULL,
    by         TEXT NOT NULL DEFAULT 'system',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        conn.execute("INSERT OR IGNORE INTO symbols VALUES (?)", (sym,))
    conn.commit()
    return conn


def _insert_position(
    conn: sqlite3.Connection,
    symbol: str = "BTCUSDT",
    side: str = "long",
    entry_price: float = 50_000.0,
    quantity: float = 0.1,
    status: str = "open",
) -> str:
    pid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO positions (
            position_id, symbol, side, entry_price, quantity,
            stop_loss, status
        ) VALUES (?,?,?,?,?,?,?)
        """,
        (pid, symbol, side, str(entry_price), str(quantity), "48000", status),
    )
    conn.commit()
    return pid


def _make_om(raises_on: set[str] | None = None) -> MagicMock:
    """Create an OrderManager mock.  Symbols in *raises_on* will raise on market_close."""
    om = MagicMock()
    raises_on = raises_on or set()

    def _market_close(symbol, close_side, qty, position_side=None):
        if symbol in raises_on:
            raise RuntimeError(f"simulated exchange error: {symbol}")
        return {"id": 99, "status": "closed", "average": 50_000.0, "filled": qty}

    om.market_close.side_effect = _market_close
    return om


def _make_handler(
    conn=None,
    om=None,
    safe_mode=None,
    telegram_bot=None,
    raises_on: set[str] | None = None,
) -> EmergencyHandler:
    c = conn or _make_conn()
    o = om or _make_om(raises_on)
    sm = safe_mode or SafeMode(conn=c)
    return EmergencyHandler(
        conn=c,
        order_manager=o,
        position_tracker=PositionTracker,
        safe_mode=sm,
        telegram_bot=telegram_bot,
    )


# ---------------------------------------------------------------------------
# close_all_positions
# ---------------------------------------------------------------------------

class TestCloseAllPositions(unittest.TestCase):

    def test_empty_positions_returns_zero(self):
        conn = _make_conn()
        h = _make_handler(conn=conn)
        result = h.close_all_positions("test")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["closed"], 0)
        self.assertEqual(result["failed"], 0)

    def test_single_long_position_closed(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", side="long", quantity=0.1)
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        result = h.close_all_positions("drawdown limit")
        self.assertEqual(result["closed"], 1)
        self.assertEqual(result["failed"], 0)

    def test_market_close_uses_opposite_side_for_long(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", side="long", quantity=0.1)
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        h.close_all_positions("test")
        # long → close with 'sell'
        om.market_close.assert_called_once_with("BTCUSDT", "sell", 0.1, position_side="long")

    def test_market_close_uses_opposite_side_for_short(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", side="short", quantity=0.2)
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        h.close_all_positions("test")
        # short → close with 'buy'
        om.market_close.assert_called_once_with("BTCUSDT", "buy", 0.2, position_side="short")

    def test_multiple_positions_all_closed(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", side="long")
        _insert_position(conn, symbol="ETHUSDT", side="short")
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        result = h.close_all_positions("multi")
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["closed"], 2)
        self.assertEqual(om.market_close.call_count, 2)

    def test_one_failure_does_not_stop_others(self):
        """ETHUSDT fails → BTCUSDT still closes."""
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", side="long")
        _insert_position(conn, symbol="ETHUSDT", side="long")
        om = _make_om(raises_on={"ETHUSDT"})
        h = _make_handler(conn=conn, om=om)
        result = h.close_all_positions("test")
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["closed"], 1)
        self.assertEqual(result["failed"], 1)

    def test_failure_recorded_in_results(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", side="long")
        om = _make_om(raises_on={"BTCUSDT"})
        h = _make_handler(conn=conn, om=om)
        result = h.close_all_positions("test")
        failed = [r for r in result["results"] if r["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["symbol"], "BTCUSDT")
        self.assertIsNotNone(failed[0]["error"])

    def test_all_failures_still_activates_safe_mode(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT")
        sm = SafeMode(conn=conn)
        om = _make_om(raises_on={"BTCUSDT"})
        h = _make_handler(conn=conn, om=om, safe_mode=sm)
        h.close_all_positions("test")
        self.assertTrue(sm.is_active())

    def test_safe_mode_activated_after_successful_close(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT")
        sm = SafeMode(conn=conn)
        h = _make_handler(conn=conn, safe_mode=sm)
        h.close_all_positions("drawdown")
        self.assertTrue(sm.is_active())

    def test_safe_mode_reason_contains_original_reason(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT")
        sm = SafeMode(conn=conn)
        h = _make_handler(conn=conn, safe_mode=sm)
        h.close_all_positions("liquidation imminent")
        self.assertIn("liquidation imminent", sm.reason)

    def test_safe_mode_not_double_activated_if_already_active(self):
        conn = _make_conn()
        sm = SafeMode(conn=conn)
        sm.activate("pre-existing reason")
        h = _make_handler(conn=conn, safe_mode=sm)
        h.close_all_positions("another reason")
        # reason should still be the original one
        self.assertEqual(sm.reason, "pre-existing reason")

    def test_telegram_notified_start_and_summary(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT")
        bot = MagicMock()
        h = _make_handler(conn=conn, telegram_bot=bot)
        h.close_all_positions("test")
        self.assertGreaterEqual(bot.send_alert.call_count, 2)

    def test_closed_positions_not_re_closed(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", status="closed")
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        result = h.close_all_positions("test")
        self.assertEqual(result["total"], 0)
        om.market_close.assert_not_called()

    def test_position_status_updated_to_closed_in_db(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT", side="long")
        h = _make_handler(conn=conn)
        h.close_all_positions("test")
        row = conn.execute(
            "SELECT status FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        self.assertEqual(row["status"], "closed")


# ---------------------------------------------------------------------------
# close_position (single symbol)
# ---------------------------------------------------------------------------

class TestCloseSinglePosition(unittest.TestCase):

    def test_not_found_returns_not_found_status(self):
        conn = _make_conn()
        h = _make_handler(conn=conn)
        result = h.close_position("BTCUSDT", "test")
        self.assertEqual(result["status"], "not_found")

    def test_closes_correct_symbol(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", side="long", quantity=0.05)
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        result = h.close_position("BTCUSDT", "manual close")
        self.assertEqual(result["status"], "closed")
        om.market_close.assert_called_once_with("BTCUSDT", "sell", 0.05, position_side="long")

    def test_exchange_failure_returns_failed_status(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT")
        om = _make_om(raises_on={"BTCUSDT"})
        h = _make_handler(conn=conn, om=om)
        result = h.close_position("BTCUSDT", "test")
        self.assertEqual(result["status"], "failed")
        self.assertIsNotNone(result["error"])

    def test_does_not_affect_other_symbols(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", side="long")
        _insert_position(conn, symbol="ETHUSDT", side="long")
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        h.close_position("BTCUSDT", "test")
        eth_row = conn.execute(
            "SELECT status FROM positions WHERE symbol='ETHUSDT'"
        ).fetchone()
        self.assertEqual(eth_row["status"], "open")


# ---------------------------------------------------------------------------
# partial_close
# ---------------------------------------------------------------------------

class TestPartialClose(unittest.TestCase):

    def test_half_close_reduces_quantity_in_db(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT", side="long",
                               entry_price=50_000.0, quantity=0.2)
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        result = h.partial_close(pid, 0.5, "WARNING 50%")
        self.assertEqual(result["status"], "closed")
        self.assertAlmostEqual(result["closed_qty"], 0.1)
        row = conn.execute(
            "SELECT quantity FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        self.assertAlmostEqual(float(row["quantity"]), 0.1)

    def test_market_close_called_with_partial_qty(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT", side="long", quantity=0.4)
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        h.partial_close(pid, 0.25, "test")
        om.market_close.assert_called_once_with("BTCUSDT", "sell", 0.1, position_side="long")  # 0.4 * 0.25

    def test_full_close_via_partial_pct_one(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT", side="long", quantity=0.1)
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        result = h.partial_close(pid, 1.0, "full via partial")
        self.assertEqual(result["status"], "closed")
        row = conn.execute(
            "SELECT status FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        self.assertEqual(row["status"], "closed")

    def test_short_position_uses_buy_to_close(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT", side="short", quantity=0.2)
        om = _make_om()
        h = _make_handler(conn=conn, om=om)
        h.partial_close(pid, 0.5, "test")
        om.market_close.assert_called_once_with("BTCUSDT", "buy", 0.1, position_side="short")

    def test_invalid_pct_raises(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT")
        h = _make_handler(conn=conn)
        with self.assertRaises(ValueError):
            h.partial_close(pid, 0.0, "zero pct")
        with self.assertRaises(ValueError):
            h.partial_close(pid, 1.5, "over 100%")

    def test_not_found_returns_not_found(self):
        conn = _make_conn()
        h = _make_handler(conn=conn)
        result = h.partial_close("nonexistent", 0.5, "test")
        self.assertEqual(result["status"], "not_found")

    def test_exchange_failure_returns_failed(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT")
        om = _make_om(raises_on={"BTCUSDT"})
        h = _make_handler(conn=conn, om=om)
        result = h.partial_close(pid, 0.5, "test")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["closed_qty"], 0.0)

    def test_position_stays_open_after_partial(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT", quantity=0.2)
        h = _make_handler(conn=conn)
        h.partial_close(pid, 0.5, "test")
        row = conn.execute(
            "SELECT status FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        self.assertEqual(row["status"], "open")


if __name__ == "__main__":
    unittest.main()
