"""Unit tests for PositionTracker.

All DB interactions use in-memory SQLite with the minimal required schema.
"""

from __future__ import annotations

import sqlite3
import uuid
import unittest
from datetime import date

from src.execution.position_tracker import PositionTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS positions (
    position_id             TEXT PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    side                    TEXT NOT NULL CHECK (side IN ('long','short')),
    leverage                INTEGER NOT NULL,
    entry_price             TEXT NOT NULL,
    exit_price              TEXT,
    quantity                TEXT NOT NULL,
    liquidation_price       TEXT NOT NULL,
    stop_loss               TEXT NOT NULL,
    take_profit_1           TEXT,
    take_profit_2           TEXT,
    initial_stop_loss       TEXT NOT NULL,
    trailing_activated      INTEGER DEFAULT 0,
    realized_pnl            TEXT DEFAULT '0',
    unrealized_pnl          TEXT DEFAULT '0',
    status                  TEXT NOT NULL DEFAULT 'open',
    close_reason            TEXT,
    trading_mode            TEXT NOT NULL DEFAULT 'testnet',
    opened_at               TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at               TEXT
);
CREATE TABLE IF NOT EXISTS daily_performance (
    perf_date               TEXT NOT NULL,
    trading_mode            TEXT NOT NULL DEFAULT 'testnet',
    total_trades            INTEGER DEFAULT 0,
    winning_trades          INTEGER DEFAULT 0,
    losing_trades           INTEGER DEFAULT 0,
    liquidated_trades       INTEGER DEFAULT 0,
    gross_profit            TEXT DEFAULT '0',
    gross_loss              TEXT DEFAULT '0',
    net_pnl                 TEXT DEFAULT '0',
    total_fees              TEXT DEFAULT '0',
    max_drawdown            TEXT DEFAULT '0',
    win_rate                TEXT DEFAULT '0',
    avg_liquidation_distance TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (perf_date, trading_mode)
);
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # seed symbols table (FK target)
    conn.execute("INSERT OR IGNORE INTO symbols VALUES ('BTCUSDT')")
    conn.execute("INSERT OR IGNORE INTO symbols VALUES ('ETHUSDT')")
    conn.commit()
    return conn


def _insert_position(
    conn: sqlite3.Connection,
    *,
    symbol: str = "BTCUSDT",
    side: str = "long",
    leverage: int = 5,
    entry_price: float = 50_000.0,
    quantity: float = 0.1,
    stop_loss: float = 48_500.0,
    take_profit_1: float | None = 52_000.0,
    take_profit_2: float | None = 54_000.0,
    status: str = "open",
    trading_mode: str = "testnet",
) -> str:
    pid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO positions (
            position_id, symbol, side, leverage, entry_price, quantity,
            liquidation_price, stop_loss, take_profit_1, take_profit_2,
            initial_stop_loss, status, trading_mode
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            pid, symbol, side, leverage,
            str(entry_price), str(quantity),
            str(entry_price * 0.8),  # simplified liq price
            str(stop_loss),
            str(take_profit_1) if take_profit_1 else None,
            str(take_profit_2) if take_profit_2 else None,
            str(stop_loss),
            status, trading_mode,
        ),
    )
    conn.commit()
    return pid


def _get_position(conn: sqlite3.Connection, pid: str) -> sqlite3.Row:
    return conn.execute(
        "SELECT * FROM positions WHERE position_id=?", (pid,)
    ).fetchone()


def _get_perf(conn: sqlite3.Connection, perf_date: str | None = None) -> sqlite3.Row | None:
    d = perf_date or date.today().isoformat()
    return conn.execute(
        "SELECT * FROM daily_performance WHERE perf_date=?", (d,)
    ).fetchone()


# ---------------------------------------------------------------------------
# update_unrealized_pnl
# ---------------------------------------------------------------------------

class TestUpdateUnrealizedPnl(unittest.TestCase):

    def test_long_profit(self):
        conn = _make_conn()
        pid = _insert_position(conn, side="long", entry_price=50_000.0, quantity=0.1)
        PositionTracker.update_unrealized_pnl(conn, {"BTCUSDT": 51_000.0})
        row = _get_position(conn, pid)
        self.assertAlmostEqual(float(row["unrealized_pnl"]), 100.0)  # (51k-50k)*0.1

    def test_long_loss(self):
        conn = _make_conn()
        pid = _insert_position(conn, side="long", entry_price=50_000.0, quantity=0.1)
        PositionTracker.update_unrealized_pnl(conn, {"BTCUSDT": 49_000.0})
        row = _get_position(conn, pid)
        self.assertAlmostEqual(float(row["unrealized_pnl"]), -100.0)

    def test_short_profit(self):
        conn = _make_conn()
        pid = _insert_position(conn, side="short", entry_price=50_000.0, quantity=0.1,
                               stop_loss=52_000.0, take_profit_1=48_000.0, take_profit_2=46_000.0)
        PositionTracker.update_unrealized_pnl(conn, {"BTCUSDT": 49_000.0})
        row = _get_position(conn, pid)
        self.assertAlmostEqual(float(row["unrealized_pnl"]), 100.0)  # (50k-49k)*0.1

    def test_short_loss(self):
        conn = _make_conn()
        pid = _insert_position(conn, side="short", entry_price=50_000.0, quantity=0.1,
                               stop_loss=52_000.0, take_profit_1=48_000.0, take_profit_2=46_000.0)
        PositionTracker.update_unrealized_pnl(conn, {"BTCUSDT": 51_000.0})
        row = _get_position(conn, pid)
        self.assertAlmostEqual(float(row["unrealized_pnl"]), -100.0)

    def test_missing_symbol_skipped(self):
        conn = _make_conn()
        pid = _insert_position(conn, symbol="BTCUSDT")
        PositionTracker.update_unrealized_pnl(conn, {"ETHUSDT": 3_000.0})
        row = _get_position(conn, pid)
        # original default is '0'
        self.assertEqual(float(row["unrealized_pnl"]), 0.0)

    def test_multiple_positions_updated(self):
        conn = _make_conn()
        pid1 = _insert_position(conn, symbol="BTCUSDT", side="long",
                                entry_price=50_000.0, quantity=0.1)
        pid2 = _insert_position(conn, symbol="ETHUSDT", side="long",
                                entry_price=3_000.0, quantity=1.0)
        PositionTracker.update_unrealized_pnl(
            conn, {"BTCUSDT": 51_000.0, "ETHUSDT": 3_100.0}
        )
        self.assertAlmostEqual(float(_get_position(conn, pid1)["unrealized_pnl"]), 100.0)
        self.assertAlmostEqual(float(_get_position(conn, pid2)["unrealized_pnl"]), 100.0)

    def test_closed_positions_not_updated(self):
        conn = _make_conn()
        pid = _insert_position(conn, status="closed", entry_price=50_000.0, quantity=0.1)
        PositionTracker.update_unrealized_pnl(conn, {"BTCUSDT": 51_000.0})
        row = _get_position(conn, pid)
        self.assertEqual(float(row["unrealized_pnl"]), 0.0)


# ---------------------------------------------------------------------------
# check_sl_tp_hit
# ---------------------------------------------------------------------------

class TestCheckSlTpHit(unittest.TestCase):

    def test_long_sl_hit(self):
        conn = _make_conn()
        _insert_position(conn, side="long", entry_price=50_000.0,
                         stop_loss=48_500.0, take_profit_1=52_000.0)
        hits = PositionTracker.check_sl_tp_hit(conn, {"BTCUSDT": 48_000.0})
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["trigger"], "sl")

    def test_long_sl_exact_boundary(self):
        conn = _make_conn()
        _insert_position(conn, side="long", entry_price=50_000.0,
                         stop_loss=48_500.0, take_profit_1=52_000.0)
        hits = PositionTracker.check_sl_tp_hit(conn, {"BTCUSDT": 48_500.0})
        self.assertEqual(hits[0]["trigger"], "sl")

    def test_long_tp1_hit(self):
        conn = _make_conn()
        _insert_position(conn, side="long", entry_price=50_000.0,
                         stop_loss=48_500.0, take_profit_1=52_000.0, take_profit_2=54_000.0)
        hits = PositionTracker.check_sl_tp_hit(conn, {"BTCUSDT": 52_500.0})
        self.assertEqual(hits[0]["trigger"], "tp1")

    def test_long_tp2_hit_takes_priority_over_tp1(self):
        conn = _make_conn()
        _insert_position(conn, side="long", entry_price=50_000.0,
                         stop_loss=48_500.0, take_profit_1=52_000.0, take_profit_2=54_000.0)
        hits = PositionTracker.check_sl_tp_hit(conn, {"BTCUSDT": 55_000.0})
        self.assertEqual(hits[0]["trigger"], "tp2")

    def test_short_sl_hit(self):
        conn = _make_conn()
        _insert_position(conn, side="short", entry_price=50_000.0,
                         stop_loss=52_000.0, take_profit_1=48_000.0, take_profit_2=46_000.0)
        hits = PositionTracker.check_sl_tp_hit(conn, {"BTCUSDT": 53_000.0})
        self.assertEqual(hits[0]["trigger"], "sl")

    def test_short_tp1_hit(self):
        conn = _make_conn()
        _insert_position(conn, side="short", entry_price=50_000.0,
                         stop_loss=52_000.0, take_profit_1=48_000.0, take_profit_2=46_000.0)
        hits = PositionTracker.check_sl_tp_hit(conn, {"BTCUSDT": 47_500.0})
        self.assertEqual(hits[0]["trigger"], "tp1")

    def test_short_tp2_hit(self):
        conn = _make_conn()
        _insert_position(conn, side="short", entry_price=50_000.0,
                         stop_loss=52_000.0, take_profit_1=48_000.0, take_profit_2=46_000.0)
        hits = PositionTracker.check_sl_tp_hit(conn, {"BTCUSDT": 45_000.0})
        self.assertEqual(hits[0]["trigger"], "tp2")

    def test_no_hit_within_range(self):
        conn = _make_conn()
        _insert_position(conn, side="long", entry_price=50_000.0,
                         stop_loss=48_500.0, take_profit_1=52_000.0)
        hits = PositionTracker.check_sl_tp_hit(conn, {"BTCUSDT": 50_500.0})
        self.assertEqual(hits, [])

    def test_missing_symbol_not_hit(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT")
        hits = PositionTracker.check_sl_tp_hit(conn, {"ETHUSDT": 1.0})
        self.assertEqual(hits, [])

    def test_no_tp_set_only_sl_checked(self):
        conn = _make_conn()
        _insert_position(conn, side="long", entry_price=50_000.0,
                         stop_loss=48_500.0, take_profit_1=None, take_profit_2=None)
        hits = PositionTracker.check_sl_tp_hit(conn, {"BTCUSDT": 48_000.0})
        self.assertEqual(hits[0]["trigger"], "sl")

    def test_hit_dict_has_required_keys(self):
        conn = _make_conn()
        _insert_position(conn, side="long", stop_loss=48_500.0)
        hits = PositionTracker.check_sl_tp_hit(conn, {"BTCUSDT": 48_000.0})
        for key in ("position_id", "symbol", "side", "trigger", "current_price", "trigger_price"):
            self.assertIn(key, hits[0])


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------

class TestClosePosition(unittest.TestCase):

    def test_status_set_to_closed(self):
        conn = _make_conn()
        pid = _insert_position(conn, entry_price=50_000.0, quantity=0.1, side="long")
        PositionTracker.close_position(conn, pid, 51_000.0, "tp1_hit")
        row = _get_position(conn, pid)
        self.assertEqual(row["status"], "closed")

    def test_exit_price_stored(self):
        conn = _make_conn()
        pid = _insert_position(conn, entry_price=50_000.0, quantity=0.1, side="long")
        PositionTracker.close_position(conn, pid, 51_000.0, "tp1_hit")
        row = _get_position(conn, pid)
        self.assertAlmostEqual(float(row["exit_price"]), 51_000.0)

    def test_long_realized_pnl_profit(self):
        conn = _make_conn()
        pid = _insert_position(conn, side="long", entry_price=50_000.0, quantity=0.1)
        PositionTracker.close_position(conn, pid, 51_000.0, "tp1_hit")
        row = _get_position(conn, pid)
        self.assertAlmostEqual(float(row["realized_pnl"]), 100.0)  # (51k-50k)*0.1

    def test_long_realized_pnl_loss(self):
        conn = _make_conn()
        pid = _insert_position(conn, side="long", entry_price=50_000.0, quantity=0.1)
        PositionTracker.close_position(conn, pid, 48_500.0, "sl_hit")
        row = _get_position(conn, pid)
        self.assertAlmostEqual(float(row["realized_pnl"]), -150.0)  # (48.5k-50k)*0.1

    def test_short_realized_pnl_profit(self):
        conn = _make_conn()
        pid = _insert_position(conn, side="short", entry_price=50_000.0, quantity=0.1,
                               stop_loss=52_000.0, take_profit_1=48_000.0)
        PositionTracker.close_position(conn, pid, 48_000.0, "tp1_hit")
        row = _get_position(conn, pid)
        self.assertAlmostEqual(float(row["realized_pnl"]), 200.0)  # (50k-48k)*0.1

    def test_close_reason_stored(self):
        conn = _make_conn()
        pid = _insert_position(conn)
        PositionTracker.close_position(conn, pid, 50_000.0, "manual")
        row = _get_position(conn, pid)
        self.assertEqual(row["close_reason"], "manual")

    def test_unrealized_pnl_zeroed_on_close(self):
        conn = _make_conn()
        pid = _insert_position(conn, entry_price=50_000.0, quantity=0.1)
        PositionTracker.update_unrealized_pnl(conn, {"BTCUSDT": 51_000.0})
        PositionTracker.close_position(conn, pid, 51_000.0, "tp1_hit")
        row = _get_position(conn, pid)
        self.assertEqual(float(row["unrealized_pnl"]), 0.0)

    def test_daily_performance_total_trades_incremented(self):
        conn = _make_conn()
        pid = _insert_position(conn, entry_price=50_000.0, quantity=0.1, side="long")
        PositionTracker.close_position(conn, pid, 51_000.0, "tp1_hit")
        perf = _get_perf(conn)
        self.assertIsNotNone(perf)
        self.assertEqual(perf["total_trades"], 1)

    def test_daily_performance_winning_trade(self):
        conn = _make_conn()
        pid = _insert_position(conn, entry_price=50_000.0, quantity=0.1, side="long")
        PositionTracker.close_position(conn, pid, 51_000.0, "tp1_hit")
        perf = _get_perf(conn)
        self.assertEqual(perf["winning_trades"], 1)
        self.assertEqual(perf["losing_trades"], 0)

    def test_daily_performance_losing_trade(self):
        conn = _make_conn()
        pid = _insert_position(conn, entry_price=50_000.0, quantity=0.1, side="long")
        PositionTracker.close_position(conn, pid, 48_500.0, "sl_hit")
        perf = _get_perf(conn)
        self.assertEqual(perf["losing_trades"], 1)
        self.assertEqual(perf["winning_trades"], 0)

    def test_daily_performance_net_pnl_accumulated(self):
        conn = _make_conn()
        pid1 = _insert_position(conn, entry_price=50_000.0, quantity=0.1, side="long")
        pid2 = _insert_position(conn, entry_price=50_000.0, quantity=0.1, side="long")
        PositionTracker.close_position(conn, pid1, 51_000.0, "tp1_hit")  # +100
        PositionTracker.close_position(conn, pid2, 48_500.0, "sl_hit")   # -150
        perf = _get_perf(conn)
        self.assertAlmostEqual(float(perf["net_pnl"]), -50.0)
        self.assertEqual(perf["total_trades"], 2)

    def test_unknown_position_id_does_not_raise(self):
        conn = _make_conn()
        PositionTracker.close_position(conn, "nonexistent-id", 50_000.0, "test")
        # no exception raised; nothing written
        perf = _get_perf(conn)
        self.assertIsNone(perf)


# ---------------------------------------------------------------------------
# get_total_exposure
# ---------------------------------------------------------------------------

class TestGetTotalExposure(unittest.TestCase):

    def test_no_open_positions_returns_zero(self):
        conn = _make_conn()
        self.assertEqual(PositionTracker.get_total_exposure(conn), 0.0)

    def test_single_long_exposure(self):
        conn = _make_conn()
        # entry=50000, qty=0.1, lev=5 → notional = 50000*0.1*5 = 25000
        _insert_position(conn, entry_price=50_000.0, quantity=0.1, leverage=5)
        self.assertAlmostEqual(PositionTracker.get_total_exposure(conn), 25_000.0)

    def test_multiple_positions_summed(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", entry_price=50_000.0, quantity=0.1, leverage=5)
        _insert_position(conn, symbol="ETHUSDT", entry_price=3_000.0, quantity=1.0, leverage=3)
        # BTC: 50000*0.1*5 = 25000; ETH: 3000*1.0*3 = 9000
        self.assertAlmostEqual(PositionTracker.get_total_exposure(conn), 34_000.0)

    def test_closed_positions_excluded(self):
        conn = _make_conn()
        _insert_position(conn, status="closed", entry_price=50_000.0, quantity=0.1, leverage=5)
        self.assertEqual(PositionTracker.get_total_exposure(conn), 0.0)

    def test_short_position_included(self):
        conn = _make_conn()
        _insert_position(conn, side="short", entry_price=50_000.0, quantity=0.2, leverage=2,
                         stop_loss=52_000.0, take_profit_1=48_000.0)
        # 50000*0.2*2 = 20000
        self.assertAlmostEqual(PositionTracker.get_total_exposure(conn), 20_000.0)


if __name__ == "__main__":
    unittest.main()
