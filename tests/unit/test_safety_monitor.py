"""Unit tests for KRX SafetyMonitor — manual SL/TP enforcement.

OrderManager calls are mocked; never hit real KIS API.
DB: in-memory SQLite.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
import unittest
from unittest.mock import AsyncMock, MagicMock

from src.safety.safety_monitor import SafetyMonitor


# ---------------------------------------------------------------------------
# Schema / helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    position_id     TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL DEFAULT 'long',
    leverage        INTEGER NOT NULL DEFAULT 1,
    entry_price     TEXT NOT NULL,
    exit_price      TEXT,
    quantity        TEXT NOT NULL,
    liquidation_price TEXT NOT NULL DEFAULT '0',
    stop_loss       TEXT NOT NULL,
    take_profit_1   TEXT,
    take_profit_2   TEXT,
    initial_stop_loss TEXT NOT NULL DEFAULT '0',
    trailing_activated INTEGER DEFAULT 0,
    realized_pnl    TEXT DEFAULT '0',
    unrealized_pnl  TEXT DEFAULT '0',
    status          TEXT NOT NULL DEFAULT 'open',
    close_reason    TEXT,
    trading_mode    TEXT NOT NULL DEFAULT 'paper',
    opened_at       TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at       TEXT
);
CREATE TABLE IF NOT EXISTS safe_mode_events (
    event_id   TEXT PRIMARY KEY,
    action     TEXT NOT NULL,
    reason     TEXT NOT NULL,
    by         TEXT NOT NULL DEFAULT 'system',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    broker_order_id TEXT DEFAULT '',
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    price           TEXT,
    quantity        TEXT NOT NULL,
    status          TEXT NOT NULL,
    filled_qty      TEXT DEFAULT '0',
    avg_fill_price  TEXT,
    updated_at      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _insert_position(
    conn, symbol="005930", side="long", quantity=10,
    entry_price=80000, stop_loss=78000,
    take_profit_1=None, take_profit_2=None, status="open"
) -> str:
    pid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO positions
           (position_id, symbol, side, entry_price, quantity, stop_loss,
            take_profit_1, take_profit_2, status)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (pid, symbol, side, str(entry_price), str(quantity), str(stop_loss),
         str(take_profit_1) if take_profit_1 else None,
         str(take_profit_2) if take_profit_2 else None,
         status),
    )
    conn.commit()
    return pid


def _make_om() -> MagicMock:
    om = MagicMock()
    om.market_close = AsyncMock()
    om.create_order = AsyncMock()
    return om


def _make_monitor(conn=None, om=None, **kwargs) -> SafetyMonitor:
    defaults = {
        "conn": conn or _make_conn(),
        "order_manager": om or _make_om(),
        "ws_manager": None,
        "kis": None,
        "telegram_bot": None,
        "pod_urls": {},
        "check_interval": 0,
    }
    defaults.update(kwargs)
    return SafetyMonitor(**defaults)


# ---------------------------------------------------------------------------
# _check_sl: stop-loss enforcement
# ---------------------------------------------------------------------------

class TestCheckSl(unittest.IsolatedAsyncioTestCase):

    async def test_sl_breach_triggers_market_close(self):
        conn = _make_conn()
        pid = _insert_position(conn, stop_loss=78000)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        triggered = await mon._check_sl(pos, price=77000)
        self.assertTrue(triggered)
        om.market_close.assert_called_once_with("005930", 10)

    async def test_sl_not_breached_no_action(self):
        conn = _make_conn()
        pid = _insert_position(conn, stop_loss=78000)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        triggered = await mon._check_sl(pos, price=80000)
        self.assertFalse(triggered)
        om.market_close.assert_not_called()

    async def test_sl_breach_marks_position_closed(self):
        conn = _make_conn()
        pid = _insert_position(conn, stop_loss=78000)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        await mon._check_sl(pos, price=77000)
        row = conn.execute("SELECT status, close_reason FROM positions WHERE position_id=?", (pid,)).fetchone()
        self.assertEqual(row["status"], "closed")
        self.assertEqual(row["close_reason"], "sl_hit")

    async def test_sl_breach_notifies_telegram(self):
        conn = _make_conn()
        pid = _insert_position(conn, stop_loss=78000)
        om = _make_om()
        telegram = MagicMock()
        mon = _make_monitor(conn=conn, om=om, telegram_bot=telegram)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        await mon._check_sl(pos, price=77000)
        telegram.send_critical.assert_called_once()
        msg = telegram.send_critical.call_args[0][0]
        self.assertIn("005930", msg)
        self.assertIn("SL", msg)

    async def test_zero_sl_no_action(self):
        conn = _make_conn()
        pid = _insert_position(conn, stop_loss=0)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        triggered = await mon._check_sl(pos, price=1000)
        self.assertFalse(triggered)
        om.market_close.assert_not_called()

    async def test_sl_market_close_failure_does_not_raise(self):
        conn = _make_conn()
        pid = _insert_position(conn, stop_loss=78000)
        om = _make_om()
        om.market_close.side_effect = RuntimeError("KIS error")
        mon = _make_monitor(conn=conn, om=om)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        # Must not raise even if market_close fails
        triggered = await mon._check_sl(pos, price=77000)
        self.assertTrue(triggered)


# ---------------------------------------------------------------------------
# _check_tp: take-profit enforcement
# ---------------------------------------------------------------------------

class TestCheckTp(unittest.IsolatedAsyncioTestCase):

    async def test_tp1_hit_sells_half_qty(self):
        conn = _make_conn()
        pid = _insert_position(conn, quantity=10, stop_loss=78000,
                               take_profit_1=86000, take_profit_2=92000)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        await mon._check_tp(pos, price=87000)
        om.create_order.assert_called_once_with("005930", "sell", 5, 86000)

    async def test_tp1_hit_moves_sl_to_breakeven(self):
        conn = _make_conn()
        pid = _insert_position(conn, entry_price=80000, quantity=10,
                               stop_loss=78000, take_profit_1=86000)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        await mon._check_tp(pos, price=87000)
        row = conn.execute("SELECT stop_loss FROM positions WHERE position_id=?", (pid,)).fetchone()
        self.assertEqual(str(row["stop_loss"]), "80000")

    async def test_tp1_not_hit_no_order(self):
        conn = _make_conn()
        pid = _insert_position(conn, quantity=10, stop_loss=78000, take_profit_1=86000)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        await mon._check_tp(pos, price=84000)
        om.create_order.assert_not_called()

    async def test_tp1_not_double_triggered(self):
        conn = _make_conn()
        pid = _insert_position(conn, quantity=10, stop_loss=78000, take_profit_1=86000)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        await mon._check_tp(pos, price=87000)  # first hit
        await mon._check_tp(pos, price=88000)  # second call — should NOT re-trigger
        self.assertEqual(om.create_order.call_count, 1)

    async def test_tp2_hit_sells_remaining(self):
        conn = _make_conn()
        pid = _insert_position(conn, quantity=10, stop_loss=78000,
                               take_profit_1=86000, take_profit_2=92000)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        mon._tp1_done.add(pid)  # simulate TP1 already done
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        await mon._check_tp(pos, price=93000)
        # TP2 with TP1 done: remaining = 10 - 5 = 5
        calls = [(c[0][1], c[0][2], c[0][3]) for c in om.create_order.call_args_list]
        tp2_calls = [c for c in calls if c[0] == "sell" and c[2] == 92000]
        self.assertEqual(len(tp2_calls), 1)
        self.assertEqual(tp2_calls[0][1], 5)

    async def test_tp2_marks_position_closed(self):
        conn = _make_conn()
        pid = _insert_position(conn, quantity=10, stop_loss=78000,
                               take_profit_1=86000, take_profit_2=92000)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        mon._tp1_done.add(pid)
        pos = dict(conn.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone())
        await mon._check_tp(pos, price=93000)
        row = conn.execute("SELECT status, close_reason FROM positions WHERE position_id=?", (pid,)).fetchone()
        self.assertEqual(row["status"], "closed")
        self.assertEqual(row["close_reason"], "tp2_hit")


# ---------------------------------------------------------------------------
# _force_close_all: EOD forced close
# ---------------------------------------------------------------------------

class TestForceCloseAll(unittest.IsolatedAsyncioTestCase):

    async def test_force_closes_all_open_positions(self):
        conn = _make_conn()
        _insert_position(conn, "005930", quantity=10)
        _insert_position(conn, "000660", quantity=5)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        await mon._force_close_all()
        self.assertEqual(om.market_close.call_count, 2)

    async def test_sets_force_closed_flag(self):
        conn = _make_conn()
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        await mon._force_close_all()
        self.assertTrue(mon._force_closed)

    async def test_force_close_not_repeated(self):
        conn = _make_conn()
        _insert_position(conn, "005930", quantity=10)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        await mon._force_close_all()
        await mon._force_close_all()  # second call should be no-op
        self.assertEqual(om.market_close.call_count, 1)

    async def test_no_open_positions_no_market_close(self):
        conn = _make_conn()
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        await mon._force_close_all()
        om.market_close.assert_not_called()

    async def test_force_close_marks_positions_closed(self):
        conn = _make_conn()
        pid = _insert_position(conn, "005930", quantity=10)
        om = _make_om()
        mon = _make_monitor(conn=conn, om=om)
        await mon._force_close_all()
        row = conn.execute("SELECT status, close_reason FROM positions WHERE position_id=?", (pid,)).fetchone()
        self.assertEqual(row["status"], "closed")
        self.assertEqual(row["close_reason"], "force_close_eod")

    async def test_force_close_failure_does_not_stop_others(self):
        conn = _make_conn()
        _insert_position(conn, "005930", quantity=10)
        _insert_position(conn, "000660", quantity=5)
        om = _make_om()
        call_count = 0
        async def fail_first(symbol, qty):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("KIS error")
        om.market_close.side_effect = fail_first
        mon = _make_monitor(conn=conn, om=om)
        await mon._force_close_all()
        self.assertEqual(call_count, 2)

    async def test_notifies_telegram_on_force_close(self):
        conn = _make_conn()
        _insert_position(conn, "005930", quantity=10)
        om = _make_om()
        telegram = MagicMock()
        mon = _make_monitor(conn=conn, om=om, telegram_bot=telegram)
        await mon._force_close_all()
        telegram.send_critical.assert_called()


# ---------------------------------------------------------------------------
# _get_price: WS first, REST fallback
# ---------------------------------------------------------------------------

class TestGetPrice(unittest.IsolatedAsyncioTestCase):

    async def test_returns_ws_price_when_available(self):
        ws = MagicMock()
        ws.get_last_price.return_value = 80500
        mon = _make_monitor(ws_manager=ws)
        price = await mon._get_price("005930")
        self.assertEqual(price, 80500)

    async def test_falls_back_to_rest_when_ws_none(self):
        kis = MagicMock()
        kis.fetch_current_price = AsyncMock(return_value={"price": "81000"})
        mon = _make_monitor(kis=kis)
        price = await mon._get_price("005930")
        self.assertEqual(price, 81000)

    async def test_returns_none_when_both_unavailable(self):
        mon = _make_monitor()
        price = await mon._get_price("005930")
        self.assertIsNone(price)


if __name__ == "__main__":
    unittest.main()
