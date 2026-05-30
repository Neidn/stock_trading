"""Unit tests for TrailingStopManager."""

from __future__ import annotations

import sqlite3
import unittest

from src.risk.trailing_stop import TrailingStopManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS positions (
            position_id       TEXT PRIMARY KEY,
            symbol            TEXT NOT NULL,
            side              TEXT NOT NULL,
            entry_price       TEXT NOT NULL,
            stop_loss         TEXT NOT NULL,
            trailing_activated INTEGER NOT NULL DEFAULT 0,
            status            TEXT NOT NULL DEFAULT 'open'
        );
        """
    )
    conn.commit()
    return conn


def _insert_position(conn, position_id, side, entry_price, stop_loss):
    conn.execute(
        "INSERT INTO positions (position_id, symbol, side, entry_price, stop_loss) VALUES (?,?,?,?,?)",
        (position_id, "BTCUSDT", side, str(entry_price), str(stop_loss)),
    )
    conn.commit()


def _pos(position_id, side, entry_price, stop_loss):
    return {
        "position_id": position_id,
        "side": side,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
    }


# ---------------------------------------------------------------------------
# Long-side trailing stop tests
# ---------------------------------------------------------------------------

class TestTrailingStopLong(unittest.TestCase):

    def setUp(self):
        self.mgr = TrailingStopManager()
        self.entry = 10_000.0
        self.atr = 100.0
        self.initial_sl = 9_700.0  # below entry

    def _p(self, current_price):
        return _pos("pos-long", "long", self.entry, self.initial_sl)

    # Profit < 1 ATR → no move
    def test_no_move_below_1atr(self):
        p = _pos("p1", "long", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry + 50, self.atr)  # 0.5 ATR profit
        self.assertIsNone(result)

    # Profit == 1 ATR → breakeven
    def test_breakeven_at_1atr(self):
        p = _pos("p1", "long", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry + self.atr, self.atr)
        self.assertEqual(result, self.entry)

    # Profit between 1 and 2 ATR → breakeven
    def test_breakeven_between_1_and_2atr(self):
        p = _pos("p1", "long", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry + 1.5 * self.atr, self.atr)
        self.assertEqual(result, self.entry)

    # Profit == 2 ATR → entry + 0.5 ATR
    def test_move_at_2atr(self):
        p = _pos("p1", "long", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry + 2 * self.atr, self.atr)
        self.assertAlmostEqual(result, self.entry + 0.5 * self.atr)

    # Profit between 2 and 3 ATR → entry + 0.5 ATR
    def test_move_between_2_and_3atr(self):
        p = _pos("p1", "long", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry + 2.5 * self.atr, self.atr)
        self.assertAlmostEqual(result, self.entry + 0.5 * self.atr)

    # Profit >= 3 ATR → entry + 1.5 ATR
    def test_move_at_3atr(self):
        p = _pos("p1", "long", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry + 3 * self.atr, self.atr)
        self.assertAlmostEqual(result, self.entry + 1.5 * self.atr)

    def test_move_above_3atr(self):
        p = _pos("p1", "long", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry + 5 * self.atr, self.atr)
        self.assertAlmostEqual(result, self.entry + 1.5 * self.atr)


# ---------------------------------------------------------------------------
# Short-side trailing stop tests
# ---------------------------------------------------------------------------

class TestTrailingStopShort(unittest.TestCase):

    def setUp(self):
        self.mgr = TrailingStopManager()
        self.entry = 10_000.0
        self.atr = 100.0
        self.initial_sl = 10_300.0  # above entry

    # Profit < 1 ATR → no move
    def test_no_move_below_1atr(self):
        p = _pos("p-short", "short", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry - 50, self.atr)  # 0.5 ATR profit
        self.assertIsNone(result)

    # Profit == 1 ATR → breakeven
    def test_breakeven_at_1atr(self):
        p = _pos("p-short", "short", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry - self.atr, self.atr)
        self.assertEqual(result, self.entry)

    # Profit == 2 ATR → entry - 0.5 ATR
    def test_move_at_2atr(self):
        p = _pos("p-short", "short", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry - 2 * self.atr, self.atr)
        self.assertAlmostEqual(result, self.entry - 0.5 * self.atr)

    # Profit >= 3 ATR → entry - 1.5 ATR
    def test_move_at_3atr(self):
        p = _pos("p-short", "short", self.entry, self.initial_sl)
        result = self.mgr.update(p, self.entry - 3 * self.atr, self.atr)
        self.assertAlmostEqual(result, self.entry - 1.5 * self.atr)


# ---------------------------------------------------------------------------
# Favourable-only movement tests
# ---------------------------------------------------------------------------

class TestFavourableMovementOnly(unittest.TestCase):

    def setUp(self):
        self.mgr = TrailingStopManager()
        self.atr = 100.0

    def test_long_sl_never_moves_down(self):
        """If current SL is already at breakeven, another 1-ATR profit call must not lower it."""
        # SL already at entry (breakeven)
        entry = 10_000.0
        p = _pos("p1", "long", entry, entry)  # sl == entry
        result = self.mgr.update(p, entry + self.atr, self.atr)
        # candidate = entry, current_sl = entry → candidate <= current_sl → no move
        self.assertIsNone(result)

    def test_long_sl_does_not_move_backward_from_advanced_position(self):
        """SL at entry+0.5*ATR must not retreat even if profit temporarily at 1 ATR."""
        entry = 10_000.0
        advanced_sl = entry + 0.5 * self.atr  # already at 2-ATR level
        p = _pos("p2", "long", entry, advanced_sl)
        result = self.mgr.update(p, entry + self.atr, self.atr)  # 1-ATR profit
        # candidate = entry < advanced_sl → no move
        self.assertIsNone(result)

    def test_short_sl_never_moves_up(self):
        """Short SL at breakeven must not rise on 1-ATR profit call."""
        entry = 10_000.0
        p = _pos("p3", "short", entry, entry)  # sl == entry
        result = self.mgr.update(p, entry - self.atr, self.atr)
        self.assertIsNone(result)

    def test_short_sl_does_not_move_backward(self):
        """Short SL at entry-0.5*ATR must not retreat to entry on lower profit."""
        entry = 10_000.0
        advanced_sl = entry - 0.5 * self.atr
        p = _pos("p4", "short", entry, advanced_sl)
        result = self.mgr.update(p, entry - self.atr, self.atr)
        # candidate = entry > advanced_sl → no move
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Breakeven protection
# ---------------------------------------------------------------------------

class TestBreakevenProtection(unittest.TestCase):

    def setUp(self):
        self.mgr = TrailingStopManager()
        self.atr = 200.0

    def test_long_breakeven_moves_sl_to_entry(self):
        entry = 50_000.0
        initial_sl = 49_500.0
        p = _pos("be-long", "long", entry, initial_sl)
        result = self.mgr.update(p, entry + self.atr, self.atr)
        self.assertEqual(result, entry)

    def test_short_breakeven_moves_sl_to_entry(self):
        entry = 50_000.0
        initial_sl = 50_500.0
        p = _pos("be-short", "short", entry, initial_sl)
        result = self.mgr.update(p, entry - self.atr, self.atr)
        self.assertEqual(result, entry)


# ---------------------------------------------------------------------------
# Activation state
# ---------------------------------------------------------------------------

class TestActivationState(unittest.TestCase):

    def setUp(self):
        self.mgr = TrailingStopManager()

    def test_not_activated_before_move(self):
        self.assertFalse(self.mgr.is_activated("p1"))

    def test_activated_after_move(self):
        p = _pos("p1", "long", 10_000.0, 9_700.0)
        self.mgr.update(p, 10_100.0, 100.0)  # 1 ATR profit → moves to breakeven
        self.assertTrue(self.mgr.is_activated("p1"))

    def test_not_activated_when_no_move(self):
        p = _pos("p1", "long", 10_000.0, 9_700.0)
        self.mgr.update(p, 10_050.0, 100.0)  # 0.5 ATR → no move
        self.assertFalse(self.mgr.is_activated("p1"))

    def test_manual_activate(self):
        self.mgr.activate("p2")
        self.assertTrue(self.mgr.is_activated("p2"))


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

class TestDBPersistence(unittest.TestCase):

    def setUp(self):
        self.conn = _make_db()
        self.mgr = TrailingStopManager(conn=self.conn)

    def test_db_updated_on_sl_move(self):
        entry = 10_000.0
        atr = 100.0
        _insert_position(self.conn, "pos-db", "long", entry, entry - 3 * atr)

        p = _pos("pos-db", "long", entry, entry - 3 * atr)
        new_sl = self.mgr.update(p, entry + atr, atr)
        self.assertIsNotNone(new_sl)

        row = self.conn.execute(
            "SELECT stop_loss, trailing_activated FROM positions WHERE position_id = ?",
            ("pos-db",),
        ).fetchone()
        self.assertEqual(float(row["stop_loss"]), new_sl)
        self.assertEqual(row["trailing_activated"], 1)

    def test_db_not_updated_when_no_move(self):
        entry = 10_000.0
        atr = 100.0
        _insert_position(self.conn, "pos-db2", "long", entry, entry - 3 * atr)

        p = _pos("pos-db2", "long", entry, entry - 3 * atr)
        result = self.mgr.update(p, entry + 0.5 * atr, atr)  # no move
        self.assertIsNone(result)

        row = self.conn.execute(
            "SELECT trailing_activated FROM positions WHERE position_id = ?",
            ("pos-db2",),
        ).fetchone()
        self.assertEqual(row["trailing_activated"], 0)


if __name__ == "__main__":
    unittest.main()
