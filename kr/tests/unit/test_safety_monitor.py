"""Unit tests for SafetyMonitor.

Exchange calls are mocked; never hit real API.
DB: in-memory SQLite.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.safety.safety_monitor import SafetyMonitor, _MARGIN_RATIO_LIMIT


# ---------------------------------------------------------------------------
# Schema / helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    position_id     TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
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
CREATE TABLE IF NOT EXISTS safe_mode_events (
    event_id   TEXT PRIMARY KEY,
    action     TEXT NOT NULL,
    reason     TEXT NOT NULL,
    by         TEXT NOT NULL DEFAULT 'system',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _insert_position(conn, symbol="BTCUSDT", side="long",
                     quantity=0.1, status="open") -> str:
    pid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO positions
           (position_id, symbol, side, entry_price, quantity, stop_loss, status)
           VALUES (?,?,?,?,?,?,?)""",
        (pid, symbol, side, "50000", str(quantity), "48000", status),
    )
    conn.commit()
    return pid


def _make_ccxt_position(symbol="BTCUSDT", side="long",
                        contracts=0.1,
                        mark_price=50000.0,
                        liquidation_price=44000.0) -> dict:
    """Minimal ccxt-style position dict."""
    return {
        "symbol": symbol,
        "side": side,
        "contracts": contracts,
        "markPrice": mark_price,
        "entryPrice": mark_price,
        "liquidationPrice": liquidation_price,
    }


def _make_balance(total: float = 10_000.0, used: float = 1_000.0) -> dict:
    """ccxt-style balance with Binance futures info fields."""
    return {
        "info": {
            "totalMarginBalance": str(total),
            "totalPositionInitialMargin": str(used),
        }
    }


def _make_monitor(**kwargs) -> SafetyMonitor:
    defaults = {"check_interval": 0, "error_sleep_sec": 0}
    defaults.update(kwargs)
    return SafetyMonitor(**defaults)


# ---------------------------------------------------------------------------
# run_forever: loop behaviour
# ---------------------------------------------------------------------------

class TestRunForever(unittest.IsolatedAsyncioTestCase):

    async def test_loop_calls_all_checks_in_order(self):
        # Use large interval so exactly one iteration runs before cancel
        mon = _make_monitor(check_interval=100)
        calls = []

        async def fake_check_positions(): calls.append("positions")
        async def fake_check_health():    calls.append("health")
        async def fake_check_pods():      calls.append("pods")

        mon._check_all_positions = fake_check_positions
        mon._check_account_health = fake_check_health
        mon._check_other_pods_health = fake_check_pods

        # Cancel after first iteration (interval=100s ensures only one run)
        async def run():
            task = asyncio.create_task(mon.run_forever())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await run()
        self.assertEqual(calls, ["positions", "health", "pods"])

    async def test_exception_does_not_stop_loop(self):
        mon = _make_monitor(check_interval=0)
        iteration_count = 0

        async def fake_check_positions():
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count == 1:
                raise RuntimeError("simulated error")

        async def noop(): pass

        mon._check_all_positions = fake_check_positions
        mon._check_account_health = noop
        mon._check_other_pods_health = noop

        async def run():
            task = asyncio.create_task(mon.run_forever())
            # Allow at least 2 iterations (first raises, second succeeds)
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await run()
        self.assertGreaterEqual(iteration_count, 2)

    async def test_exception_triggers_telegram_critical(self):
        telegram = MagicMock()
        mon = _make_monitor(telegram_bot=telegram)

        async def boom(): raise ValueError("boom")
        async def noop(): pass

        mon._check_all_positions = boom
        mon._check_account_health = noop
        mon._check_other_pods_health = noop

        async def run():
            task = asyncio.create_task(mon.run_forever())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await run()
        telegram.send_critical.assert_called()
        args = telegram.send_critical.call_args[0][0]
        self.assertIn("boom", args)


# ---------------------------------------------------------------------------
# _check_all_positions
# ---------------------------------------------------------------------------

class TestCheckAllPositions(unittest.IsolatedAsyncioTestCase):

    async def test_no_exchange_returns_silently(self):
        mon = _make_monitor()
        # Should not raise
        await mon._check_all_positions()

    async def test_safe_positions_no_action(self):
        eh = MagicMock()
        exchange = MagicMock()
        # mark=50000, liq=10000 → very safe
        exchange.fetch_positions.return_value = [
            _make_ccxt_position(mark_price=50_000.0, liquidation_price=10_000.0)
        ]
        mon = _make_monitor(exchange=exchange, emergency_handler=eh)
        await mon._check_all_positions()
        eh.close_all_positions.assert_not_called()
        eh.partial_close.assert_not_called()

    async def test_zero_contracts_skipped(self):
        eh = MagicMock()
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            _make_ccxt_position(contracts=0)
        ]
        mon = _make_monitor(exchange=exchange, emergency_handler=eh)
        await mon._check_all_positions()
        eh.close_all_positions.assert_not_called()

    async def test_critical_triggers_close_all(self):
        eh = MagicMock()
        eh.close_all_positions.return_value = {"closed": 1, "failed": 0, "total": 1, "results": []}
        exchange = MagicMock()
        # mark=50000, liq=47000 → dist=(50000-47000)/50000*100 = 6% → CRITICAL (<8%)
        exchange.fetch_positions.return_value = [
            _make_ccxt_position(mark_price=50_000.0, liquidation_price=47_000.0)
        ]
        mon = _make_monitor(exchange=exchange, emergency_handler=eh)
        await mon._check_all_positions()
        eh.close_all_positions.assert_called_once()
        reason = eh.close_all_positions.call_args[0][0]
        self.assertIn("청산 임박", reason)

    async def test_warning_triggers_partial_close(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT")
        eh = MagicMock()
        eh.partial_close.return_value = {"status": "closed", "closed_qty": 0.05}
        exchange = MagicMock()
        # mark=50000, liq=43500 → dist=(50000-43500)/50000*100=13% → WARNING (8<dist<=15)
        exchange.fetch_positions.return_value = [
            _make_ccxt_position(symbol="BTCUSDT", mark_price=50_000.0,
                                liquidation_price=43_500.0)
        ]
        mon = _make_monitor(exchange=exchange, emergency_handler=eh, conn=conn)
        await mon._check_all_positions()
        eh.partial_close.assert_called_once_with(pid, 0.5, unittest.mock.ANY)

    async def test_critical_breaks_loop_after_close_all(self):
        """Second position should NOT trigger another close after CRITICAL fires."""
        eh = MagicMock()
        eh.close_all_positions.return_value = {"closed": 2, "failed": 0, "total": 2, "results": []}
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            _make_ccxt_position(symbol="BTCUSDT", mark_price=50_000.0,
                                liquidation_price=47_000.0),  # CRITICAL
            _make_ccxt_position(symbol="ETHUSDT", mark_price=3_000.0,
                                liquidation_price=2_820.0),   # also CRITICAL
        ]
        mon = _make_monitor(exchange=exchange, emergency_handler=eh)
        await mon._check_all_positions()
        # close_all should only be called once (loop breaks after first CRITICAL)
        self.assertEqual(eh.close_all_positions.call_count, 1)

    async def test_missing_liquidation_price_skipped(self):
        eh = MagicMock()
        exchange = MagicMock()
        pos = _make_ccxt_position()
        pos["liquidationPrice"] = None
        pos["markPrice"] = None
        exchange.fetch_positions.return_value = [pos]
        mon = _make_monitor(exchange=exchange, emergency_handler=eh)
        await mon._check_all_positions()
        eh.close_all_positions.assert_not_called()

    async def test_critical_notifies_telegram(self):
        telegram = MagicMock()
        eh = MagicMock()
        eh.close_all_positions.return_value = {"closed": 0, "failed": 0, "total": 0, "results": []}
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            _make_ccxt_position(mark_price=50_000.0, liquidation_price=47_000.0)
        ]
        mon = _make_monitor(exchange=exchange, emergency_handler=eh, telegram_bot=telegram)
        await mon._check_all_positions()
        telegram.send_critical.assert_called()


# ---------------------------------------------------------------------------
# _check_stop_loss_breached
# ---------------------------------------------------------------------------

class TestCheckStopLossBreached(unittest.IsolatedAsyncioTestCase):

    async def test_long_sl_breached_triggers_close(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT", side="long")
        # stop_loss=48000, mark=47000 → breached
        conn.execute("UPDATE positions SET stop_loss='48000' WHERE position_id=?", (pid,))
        conn.commit()
        eh = MagicMock()
        eh.partial_close.return_value = {"status": "closed"}
        mon = _make_monitor(emergency_handler=eh, conn=conn)
        result = await mon._check_stop_loss_breached("BTCUSDT", "long", 47_000.0)
        self.assertTrue(result)
        eh.partial_close.assert_called_once_with(pid, 1.0, unittest.mock.ANY)

    async def test_short_sl_breached_triggers_close(self):
        conn = _make_conn()
        # short entry=50000, sl=52000 (above entry), mark=53000 → breached
        pid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO positions (position_id, symbol, side, entry_price, quantity, stop_loss, status)"
            " VALUES (?,?,?,?,?,?,?)",
            (pid, "ETHUSDT", "short", "50000", "0.5", "52000", "open"),
        )
        conn.commit()
        eh = MagicMock()
        eh.partial_close.return_value = {"status": "closed"}
        mon = _make_monitor(emergency_handler=eh, conn=conn)
        result = await mon._check_stop_loss_breached("ETHUSDT", "short", 53_000.0)
        self.assertTrue(result)
        eh.partial_close.assert_called_once_with(pid, 1.0, unittest.mock.ANY)

    async def test_long_sl_not_breached_no_action(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", side="long")
        # stop_loss=48000, mark=49000 → safe
        eh = MagicMock()
        mon = _make_monitor(emergency_handler=eh, conn=conn)
        result = await mon._check_stop_loss_breached("BTCUSDT", "long", 49_000.0)
        self.assertFalse(result)
        eh.partial_close.assert_not_called()

    async def test_no_open_position_returns_false(self):
        conn = _make_conn()
        eh = MagicMock()
        mon = _make_monitor(emergency_handler=eh, conn=conn)
        result = await mon._check_stop_loss_breached("BTCUSDT", "long", 47_000.0)
        self.assertFalse(result)
        eh.partial_close.assert_not_called()

    async def test_no_conn_returns_false(self):
        mon = _make_monitor()
        result = await mon._check_stop_loss_breached("BTCUSDT", "long", 47_000.0)
        self.assertFalse(result)

    async def test_sl_breach_notifies_telegram(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT", side="long")
        conn.execute("UPDATE positions SET stop_loss='48000' WHERE position_id=?", (pid,))
        conn.commit()
        eh = MagicMock()
        eh.partial_close.return_value = {"status": "closed"}
        telegram = MagicMock()
        mon = _make_monitor(emergency_handler=eh, conn=conn, telegram_bot=telegram)
        await mon._check_stop_loss_breached("BTCUSDT", "long", 47_000.0)
        telegram.send_critical.assert_called_once()
        msg = telegram.send_critical.call_args[0][0]
        self.assertIn("BTCUSDT", msg)
        self.assertIn("SL", msg)

    async def test_sl_breach_in_check_all_positions(self):
        """SL breach in _check_all_positions skips liquidation guard."""
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT", side="long")
        conn.execute("UPDATE positions SET stop_loss='48000' WHERE position_id=?", (pid,))
        conn.commit()
        eh = MagicMock()
        eh.partial_close.return_value = {"status": "closed"}
        exchange = MagicMock()
        # mark=47000 (below SL=48000), liq=10000 (safe)
        exchange.fetch_positions.return_value = [
            _make_ccxt_position(symbol="BTCUSDT", mark_price=47_000.0, liquidation_price=10_000.0)
        ]
        mon = _make_monitor(exchange=exchange, emergency_handler=eh, conn=conn)
        await mon._check_all_positions()
        # partial_close called for SL breach (pct=1.0)
        eh.partial_close.assert_called_once_with(pid, 1.0, unittest.mock.ANY)
        # close_all NOT called (not near liquidation)
        eh.close_all_positions.assert_not_called()


# ---------------------------------------------------------------------------
# _check_account_health
# ---------------------------------------------------------------------------

class TestCheckAccountHealth(unittest.IsolatedAsyncioTestCase):

    async def test_no_exchange_returns_silently(self):
        mon = _make_monitor()
        await mon._check_account_health()

    async def test_low_margin_no_action(self):
        exchange = MagicMock()
        # 10% usage → safe
        exchange.fetch_balance.return_value = _make_balance(total=10_000, used=1_000)
        sm = MagicMock()
        sm.is_active.return_value = False
        mon = _make_monitor(exchange=exchange, safe_mode=sm)
        await mon._check_account_health()
        sm.activate.assert_not_called()

    async def test_high_margin_activates_safe_mode(self):
        exchange = MagicMock()
        # 85% usage → breach
        exchange.fetch_balance.return_value = _make_balance(total=10_000, used=8_500)
        sm = MagicMock()
        sm.is_active.return_value = False
        mon = _make_monitor(exchange=exchange, safe_mode=sm)
        await mon._check_account_health()
        sm.activate.assert_called_once()
        reason = sm.activate.call_args[1]["reason"] or sm.activate.call_args[0][0]
        self.assertIn("마진", reason)

    async def test_high_margin_already_active_not_reactivated(self):
        exchange = MagicMock()
        exchange.fetch_balance.return_value = _make_balance(total=10_000, used=9_000)
        sm = MagicMock()
        sm.is_active.return_value = True
        mon = _make_monitor(exchange=exchange, safe_mode=sm)
        await mon._check_account_health()
        sm.activate.assert_not_called()

    async def test_high_margin_triggers_telegram(self):
        exchange = MagicMock()
        exchange.fetch_balance.return_value = _make_balance(total=10_000, used=8_500)
        telegram = MagicMock()
        sm = MagicMock()
        sm.is_active.return_value = False
        mon = _make_monitor(exchange=exchange, safe_mode=sm, telegram_bot=telegram)
        await mon._check_account_health()
        telegram.send_critical.assert_called()

    async def test_zero_balance_skipped(self):
        exchange = MagicMock()
        exchange.fetch_balance.return_value = _make_balance(total=0, used=0)
        sm = MagicMock()
        mon = _make_monitor(exchange=exchange, safe_mode=sm)
        await mon._check_account_health()
        sm.activate.assert_not_called()

    async def test_exactly_80pct_does_not_trigger(self):
        """Threshold is strictly > 80%, so 80.0% exactly should NOT trigger."""
        exchange = MagicMock()
        exchange.fetch_balance.return_value = _make_balance(total=10_000, used=8_000)
        sm = MagicMock()
        sm.is_active.return_value = False
        mon = _make_monitor(exchange=exchange, safe_mode=sm)
        await mon._check_account_health()
        sm.activate.assert_not_called()

    async def test_fallback_to_totalUsedMaintMargin(self):
        exchange = MagicMock()
        exchange.fetch_balance.return_value = {
            "info": {
                "totalMarginBalance": "10000",
                "totalUsedMaintMargin": "9000",  # fallback field, 90%
                # no totalPositionInitialMargin
            }
        }
        sm = MagicMock()
        sm.is_active.return_value = False
        mon = _make_monitor(exchange=exchange, safe_mode=sm)
        await mon._check_account_health()
        sm.activate.assert_called_once()


# ---------------------------------------------------------------------------
# _check_other_pods_health
# ---------------------------------------------------------------------------

class TestCheckOtherPodsHealth(unittest.IsolatedAsyncioTestCase):

    async def test_all_healthy_no_telegram(self):
        telegram = MagicMock()
        mon = _make_monitor(
            telegram_bot=telegram,
            pod_urls={"svc-a": "http://a/health"},
        )
        with patch.object(mon, "_http_get_status", return_value=200):
            await mon._check_other_pods_health()
        telegram.send_warning.assert_not_called()

    async def test_unhealthy_http_sends_warning(self):
        telegram = MagicMock()
        mon = _make_monitor(
            telegram_bot=telegram,
            pod_urls={"svc-a": "http://a/health"},
        )
        with patch.object(mon, "_http_get_status", return_value=503):
            await mon._check_other_pods_health()
        telegram.send_warning.assert_called_once()
        msg = telegram.send_warning.call_args[0][0]
        self.assertIn("svc-a", msg)

    async def test_connection_error_sends_warning(self):
        telegram = MagicMock()
        mon = _make_monitor(
            telegram_bot=telegram,
            pod_urls={"svc-b": "http://b/health"},
        )
        with patch.object(mon, "_http_get_status",
                          side_effect=OSError("connection refused")):
            await mon._check_other_pods_health()
        telegram.send_warning.assert_called_once()
        msg = telegram.send_warning.call_args[0][0]
        self.assertIn("svc-b", msg)

    async def test_multiple_pods_checked_independently(self):
        """One failing pod must not prevent checking others."""
        telegram = MagicMock()
        mon = _make_monitor(
            telegram_bot=telegram,
            pod_urls={
                "pod-ok":   "http://ok/health",
                "pod-fail": "http://fail/health",
            },
        )
        call_count = 0

        def side_effect(url, **_):
            nonlocal call_count
            call_count += 1
            if "fail" in url:
                raise OSError("down")
            return 200

        with patch.object(mon, "_http_get_status", side_effect=side_effect):
            await mon._check_other_pods_health()

        self.assertEqual(call_count, 2)
        self.assertEqual(telegram.send_warning.call_count, 1)

    async def test_no_pod_urls_no_calls(self):
        telegram = MagicMock()
        mon = _make_monitor(telegram_bot=telegram, pod_urls={})
        await mon._check_other_pods_health()
        telegram.send_warning.assert_not_called()


# ---------------------------------------------------------------------------
# _partial_close helpers
# ---------------------------------------------------------------------------

class TestPartialCloseHelper(unittest.IsolatedAsyncioTestCase):

    async def test_no_conn_logs_and_returns(self):
        eh = MagicMock()
        mon = _make_monitor(emergency_handler=eh, conn=None)
        await mon._partial_close("BTCUSDT", 0.5, "test")
        eh.partial_close.assert_not_called()

    async def test_no_eh_logs_and_returns(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT")
        mon = _make_monitor(emergency_handler=None, conn=conn)
        # Should not raise
        await mon._partial_close("BTCUSDT", 0.5, "test")

    async def test_no_open_position_skips(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", status="closed")
        eh = MagicMock()
        mon = _make_monitor(emergency_handler=eh, conn=conn)
        await mon._partial_close("BTCUSDT", 0.5, "test")
        eh.partial_close.assert_not_called()

    async def test_position_id_passed_correctly(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="ETHUSDT")
        eh = MagicMock()
        eh.partial_close.return_value = {"status": "closed", "closed_qty": 0.05}
        mon = _make_monitor(emergency_handler=eh, conn=conn)
        await mon._partial_close("ETHUSDT", 0.5, "reason")
        eh.partial_close.assert_called_once_with(pid, 0.5, "reason")


if __name__ == "__main__":
    unittest.main()
