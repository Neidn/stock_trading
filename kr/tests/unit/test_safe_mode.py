"""Unit tests for SafeMode.

All DB interactions use in-memory SQLite. No network calls, no file I/O.
"""

from __future__ import annotations

import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.safety.safe_mode import SafeMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
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
    conn.commit()
    return conn


def _event_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM safe_mode_events ORDER BY created_at").fetchall()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSafeModeActivateDeactivate(unittest.TestCase):

    def test_initially_inactive(self):
        sm = SafeMode()
        self.assertFalse(sm.is_active())
        self.assertEqual(sm.reason, "")

    def test_activate_sets_state(self):
        sm = SafeMode()
        sm.activate("drawdown limit hit")
        self.assertTrue(sm.is_active())
        self.assertEqual(sm.reason, "drawdown limit hit")

    def test_deactivate_clears_state(self):
        sm = SafeMode()
        sm.activate("test")
        sm.deactivate(by="manual")
        self.assertFalse(sm.is_active())
        self.assertEqual(sm.reason, "")

    def test_deactivate_idempotent(self):
        """Calling deactivate twice must not raise or double-write."""
        sm = SafeMode()
        sm.deactivate()  # already inactive → no-op
        self.assertFalse(sm.is_active())

    def test_activate_deactivate_cycle(self):
        sm = SafeMode()
        sm.activate("reason A")
        sm.deactivate()
        sm.activate("reason B")
        self.assertTrue(sm.is_active())
        self.assertEqual(sm.reason, "reason B")


class TestSafeModeDb(unittest.TestCase):

    def test_activate_writes_db_record(self):
        conn = _make_conn()
        sm = SafeMode(conn=conn)
        sm.activate("emergency stop")
        rows = _event_rows(conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "activated")
        self.assertEqual(rows[0]["reason"], "emergency stop")
        self.assertEqual(rows[0]["by"], "system")

    def test_deactivate_writes_db_record(self):
        conn = _make_conn()
        sm = SafeMode(conn=conn)
        sm.activate("x")
        sm.deactivate(by="telegram")
        rows = _event_rows(conn)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["action"], "deactivated")
        self.assertEqual(rows[1]["by"], "telegram")

    def test_deactivate_noop_when_inactive_does_not_write(self):
        """Extra deactivate call must not insert spurious DB row."""
        conn = _make_conn()
        sm = SafeMode(conn=conn)
        sm.deactivate()
        self.assertEqual(len(_event_rows(conn)), 0)

    def test_no_conn_works_without_error(self):
        sm = SafeMode(conn=None)
        sm.activate("no db")
        sm.deactivate()
        self.assertFalse(sm.is_active())

    def test_each_event_has_unique_id(self):
        conn = _make_conn()
        sm = SafeMode(conn=conn)
        sm.activate("a")
        sm.deactivate()
        sm.activate("b")
        sm.deactivate()
        rows = _event_rows(conn)
        ids = [r["event_id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_deactivate_records_original_reason(self):
        """Deactivated event must capture the reason set at activation time."""
        conn = _make_conn()
        sm = SafeMode(conn=conn)
        sm.activate("liquidation risk")
        sm.deactivate(by="auto")
        rows = _event_rows(conn)
        deact_row = next(r for r in rows if r["action"] == "deactivated")
        self.assertEqual(deact_row["reason"], "liquidation risk")


class TestSafeModeAutoRelease(unittest.TestCase):

    def test_no_release_when_timer_zero(self):
        sm = SafeMode()
        sm.activate("test", auto_release_hours=0)
        sm.check_auto_release()
        self.assertTrue(sm.is_active())

    def test_no_release_before_timer_expires(self):
        sm = SafeMode()
        sm.activate("test", auto_release_hours=2.0)
        # activated_at is now; elapsed ≈ 0 h < 2 h
        sm.check_auto_release()
        self.assertTrue(sm.is_active())

    def test_auto_release_after_timer_expires(self):
        sm = SafeMode()
        sm.activate("test", auto_release_hours=1.0)
        # Backdate activation time by 2 hours
        sm._activated_at = datetime.now(timezone.utc) - timedelta(hours=2)
        sm.check_auto_release()
        self.assertFalse(sm.is_active())

    def test_auto_release_writes_deactivated_event(self):
        conn = _make_conn()
        sm = SafeMode(conn=conn)
        sm.activate("auto test", auto_release_hours=1.0)
        sm._activated_at = datetime.now(timezone.utc) - timedelta(hours=2)
        sm.check_auto_release()
        rows = _event_rows(conn)
        deact = next((r for r in rows if r["action"] == "deactivated"), None)
        self.assertIsNotNone(deact)
        self.assertEqual(deact["by"], "auto")

    def test_auto_release_noop_when_inactive(self):
        sm = SafeMode()
        sm._auto_release_hours = 1.0
        sm._activated_at = datetime.now(timezone.utc) - timedelta(hours=2)
        sm.check_auto_release()  # _active=False → must not raise
        self.assertFalse(sm.is_active())

    def test_auto_release_exact_boundary(self):
        """Elapsed == auto_release_hours should trigger release."""
        sm = SafeMode()
        sm.activate("boundary", auto_release_hours=3.0)
        sm._activated_at = datetime.now(timezone.utc) - timedelta(hours=3)
        sm.check_auto_release()
        self.assertFalse(sm.is_active())


class TestSafeModeTelegram(unittest.TestCase):

    def test_telegram_send_alert_called_on_activate(self):
        bot = MagicMock()
        sm = SafeMode(telegram_bot=bot)
        sm.activate("alert test")
        bot.send_alert.assert_called_once()
        call_arg = bot.send_alert.call_args[0][0]
        self.assertIn("ACTIVATED", call_arg)

    def test_telegram_send_alert_called_on_deactivate(self):
        bot = MagicMock()
        sm = SafeMode(telegram_bot=bot)
        sm.activate("x")
        sm.deactivate(by="test")
        self.assertEqual(bot.send_alert.call_count, 2)
        last_call = bot.send_alert.call_args[0][0]
        self.assertIn("DEACTIVATED", last_call)

    def test_telegram_failure_does_not_raise(self):
        bot = MagicMock()
        bot.send_alert.side_effect = RuntimeError("network down")
        sm = SafeMode(telegram_bot=bot)
        sm.activate("resilience test")  # must not propagate RuntimeError
        self.assertTrue(sm.is_active())


if __name__ == "__main__":
    unittest.main()
