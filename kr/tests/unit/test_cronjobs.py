"""Unit tests for CronJob modules.

All external I/O is mocked.  DB: in-memory SQLite.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
import unittest
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.jobs.position_sync import PositionSyncJob
from src.jobs.daily_report import DailyReportJob
from src.jobs.db_archiver import DbArchiverJob


# ---------------------------------------------------------------------------
# Schema / helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol      TEXT PRIMARY KEY,
    base_asset  TEXT NOT NULL DEFAULT '',
    quote_asset TEXT NOT NULL DEFAULT 'KRW',
    is_active   INTEGER NOT NULL DEFAULT 1,
    strategy    TEXT,
    market      TEXT,
    market_cap  TEXT,
    added_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
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
    strategy_name   TEXT,
    opened_at       TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at       TEXT
);
CREATE TABLE IF NOT EXISTS klines (
    id            TEXT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    interval_type TEXT NOT NULL,
    open_time     TEXT NOT NULL,
    open          TEXT NOT NULL DEFAULT '0',
    high          TEXT NOT NULL DEFAULT '0',
    low           TEXT NOT NULL DEFAULT '0',
    close         TEXT NOT NULL DEFAULT '0',
    volume        TEXT NOT NULL DEFAULT '0',
    close_time    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(symbol, interval_type, open_time)
);
CREATE TABLE IF NOT EXISTS daily_performance (
    perf_date    TEXT NOT NULL,
    trading_mode TEXT NOT NULL DEFAULT 'paper',
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    liquidated_trades INTEGER DEFAULT 0,
    gross_profit TEXT DEFAULT '0',
    gross_loss   TEXT DEFAULT '0',
    net_pnl      TEXT DEFAULT '0',
    total_fees   TEXT DEFAULT '0',
    max_drawdown TEXT DEFAULT '0',
    win_rate     TEXT DEFAULT '0',
    avg_liquidation_distance TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (perf_date, trading_mode)
);
CREATE TABLE IF NOT EXISTS sync_events (
    event_id      TEXT PRIMARY KEY,
    success       INTEGER NOT NULL,
    discrepancies INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _insert_symbol(conn, symbol="005930", is_active=1) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO symbols (symbol, is_active) VALUES (?,?)",
        (symbol, is_active),
    )
    conn.commit()


def _insert_position(conn, symbol="005930", side="long", quantity=10,
                     entry_price=80000.0, status="open") -> str:
    pid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO positions
           (position_id, symbol, side, entry_price, quantity,
            liquidation_price, stop_loss, status)
           VALUES (?,?,?,?,?,?,?,?)""",
        (pid, symbol, side, str(entry_price), str(quantity), "0", "78000", status),
    )
    conn.commit()
    return pid


def _insert_kline(conn, symbol="005930", interval="1m", open_time="2024-01-01 00:00:00"):
    conn.execute(
        "INSERT OR IGNORE INTO klines (id, symbol, interval_type, open_time) VALUES (?,?,?,?)",
        (str(uuid.uuid4()), symbol, interval, open_time),
    )
    conn.commit()


def _insert_perf(conn, perf_date=None, total=4, wins=3, net_pnl=200.0,
                 gross_profit=300.0, gross_loss=100.0):
    today = perf_date or (date.today() - timedelta(days=1)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO daily_performance
           (perf_date, trading_mode, total_trades, winning_trades, losing_trades,
            net_pnl, gross_profit, gross_loss)
           VALUES (?,?,?,?,?,?,?,?)""",
        (today, "paper", total, wins, total - wins,
         str(net_pnl), str(gross_profit), str(gross_loss)),
    )
    conn.commit()


def _make_kis_holdings(*symbols_qtys) -> AsyncMock:
    """Return AsyncMock kis with fetch_positions returning given (symbol, qty) pairs."""
    kis = MagicMock()
    holdings = [
        {"symbol": sym, "positionAmt": str(qty), "entryPrice": "80000", "unrealizedProfit": "0"}
        for sym, qty in symbols_qtys
    ]
    kis.fetch_positions = AsyncMock(return_value=holdings)
    kis.fetch_unfilled_orders = AsyncMock(return_value=[])
    return kis


# ===========================================================================
# PositionSyncJob
# ===========================================================================

class TestPositionSyncJob(unittest.TestCase):

    def test_no_discrepancies_logged(self):
        conn = _make_conn()
        _insert_position(conn, "005930", quantity=10)
        kis = _make_kis_holdings(("005930", 10))
        result = PositionSyncJob(kis=kis, conn=conn).run()
        self.assertTrue(result["success"])
        self.assertEqual(result["discrepancies"], 0)

    def test_missing_in_db_inserted(self):
        conn = _make_conn()
        kis = _make_kis_holdings(("005930", 5))
        PositionSyncJob(kis=kis, conn=conn).run()
        row = conn.execute("SELECT * FROM positions WHERE symbol='005930'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "open")

    def test_ghost_position_closed(self):
        conn = _make_conn()
        _insert_position(conn, "000660")
        kis = _make_kis_holdings()  # nothing on KIS
        PositionSyncJob(kis=kis, conn=conn).run()
        row = conn.execute("SELECT status FROM positions WHERE symbol='000660'").fetchone()
        self.assertEqual(row["status"], "closed")

    def test_quantity_mismatch_corrected(self):
        conn = _make_conn()
        _insert_position(conn, "005930", quantity=10)
        kis = _make_kis_holdings(("005930", 20))
        PositionSyncJob(kis=kis, conn=conn).run()
        row = conn.execute("SELECT quantity FROM positions WHERE symbol='005930'").fetchone()
        self.assertEqual(int(float(row["quantity"])), 20)

    def test_sync_event_logged_on_success(self):
        conn = _make_conn()
        kis = _make_kis_holdings()
        PositionSyncJob(kis=kis, conn=conn).run()
        row = conn.execute("SELECT success FROM sync_events").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)

    def test_sync_event_logged_on_failure(self):
        conn = _make_conn()
        kis = MagicMock()
        kis.fetch_positions = AsyncMock(side_effect=RuntimeError("API down"))
        PositionSyncJob(kis=kis, conn=conn).run()
        row = conn.execute("SELECT success, error_message FROM sync_events").fetchone()
        self.assertEqual(row[0], 0)
        self.assertIn("API down", row[1])

    def test_exception_returns_failure_dict(self):
        conn = _make_conn()
        kis = MagicMock()
        kis.fetch_positions = AsyncMock(side_effect=RuntimeError("timeout"))
        result = PositionSyncJob(kis=kis, conn=conn).run()
        self.assertFalse(result["success"])
        self.assertIn("timeout", result["error"])

    def test_telegram_warning_on_discrepancy(self):
        conn = _make_conn()
        _insert_position(conn, "000660")
        kis = _make_kis_holdings()
        bot = MagicMock()
        PositionSyncJob(kis=kis, conn=conn, telegram_bot=bot).run()
        bot.send_warning.assert_called()

    def test_multiple_discrepancies_all_resolved(self):
        conn = _make_conn()
        _insert_position(conn, "000660")           # ghost
        _insert_position(conn, "005930", quantity=10)  # qty mismatch
        kis = _make_kis_holdings(("005930", 20), ("035720", 3))  # mismatch + new
        result = PositionSyncJob(kis=kis, conn=conn).run()
        self.assertEqual(result["discrepancies"], 3)  # ghost + qty_mismatch + missing


# ===========================================================================
# DailyReportJob
# ===========================================================================

class TestDailyReportJob(unittest.TestCase):

    def _yesterday(self) -> str:
        return (date.today() - timedelta(days=1)).isoformat()

    def test_no_perf_data_shows_no_trades(self):
        conn = _make_conn()
        job = DailyReportJob(conn=conn)
        report = job.run()
        self.assertIn("거래 없음", report)

    def test_perf_data_shown(self):
        conn = _make_conn()
        _insert_perf(conn, net_pnl=200.0, total=4, wins=3)
        job = DailyReportJob(conn=conn)
        report = job.run()
        self.assertIn("200.00", report)
        self.assertIn("75.0%", report)

    def test_open_positions_listed(self):
        conn = _make_conn()
        _insert_perf(conn)
        _insert_position(conn, "005930", side="long")
        job = DailyReportJob(conn=conn)
        report = job.run()
        self.assertIn("005930", report)
        self.assertIn("LONG", report)

    def test_telegram_send_called(self):
        conn = _make_conn()
        bot = MagicMock()
        DailyReportJob(conn=conn, telegram_bot=bot).run()
        bot.send_info.assert_called_once()

    def test_report_contains_date(self):
        conn = _make_conn()
        job = DailyReportJob(conn=conn)
        report = job.run()
        self.assertIn(self._yesterday(), report)

    def test_no_telegram_does_not_raise(self):
        conn = _make_conn()
        _insert_perf(conn)
        DailyReportJob(conn=conn, telegram_bot=None).run()

    def test_zero_trades_win_rate_zero(self):
        conn = _make_conn()
        _insert_perf(conn, total=0, wins=0, net_pnl=0.0)
        job = DailyReportJob(conn=conn)
        report = job.run()
        self.assertIn("0.0%", report)


# ===========================================================================
# DbArchiverJob
# ===========================================================================

class TestDbArchiverJob(unittest.TestCase):

    def test_prunes_oldest_klines_by_count(self):
        conn = _make_conn()
        for ts in ["2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-01-03 00:00:00"]:
            _insert_kline(conn, interval="1m", open_time=ts)
        result = DbArchiverJob(conn=conn, keep_candles=2).run()
        self.assertEqual(result["deleted_rows"], 1)
        row = conn.execute("SELECT count(*) FROM klines WHERE interval_type='1m'").fetchone()
        self.assertEqual(row[0], 2)

    def test_below_limit_nothing_deleted(self):
        conn = _make_conn()
        _insert_kline(conn, interval="1m", open_time="2024-01-01 00:00:00")
        result = DbArchiverJob(conn=conn, keep_candles=5).run()
        self.assertEqual(result["deleted_rows"], 0)

    def test_prunes_all_intervals_by_count(self):
        conn = _make_conn()
        for ts in ["2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-01-03 00:00:00"]:
            _insert_kline(conn, interval="1h", open_time=ts)
        result = DbArchiverJob(conn=conn, keep_candles=2).run()
        self.assertEqual(result["deleted_rows"], 1)

    def test_result_contains_deleted_rows_and_vacuumed(self):
        conn = _make_conn()
        result = DbArchiverJob(conn=conn).run()
        self.assertIn("deleted_rows", result)
        self.assertIn("vacuumed", result)

    def test_vacuumed_flag_true(self):
        conn = _make_conn()
        result = DbArchiverJob(conn=conn).run()
        self.assertTrue(result["vacuumed"])

    def test_custom_keep_candles(self):
        conn = _make_conn()
        for ts in ["2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-01-03 00:00:00"]:
            _insert_kline(conn, interval="1m", open_time=ts)
        result = DbArchiverJob(conn=conn, keep_candles=1).run()
        self.assertEqual(result["deleted_rows"], 2)

    def test_multiple_old_rows_all_pruned(self):
        conn = _make_conn()
        for i in range(7):
            ts = f"2024-01-{i+1:02d} 00:00:00"
            _insert_kline(conn, interval="1m", open_time=ts)
        result = DbArchiverJob(conn=conn, keep_candles=2).run()
        self.assertEqual(result["deleted_rows"], 5)


# ---------------------------------------------------------------------------
# _performance_factor tests
# ---------------------------------------------------------------------------

def _make_perf_conn(trades: list[tuple[str, float]]) -> sqlite3.Connection:
    """In-memory DB with closed positions. trades: [(strategy_name, realized_pnl)]"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE positions (
            position_id TEXT PRIMARY KEY,
            symbol TEXT,
            strategy_name TEXT,
            realized_pnl TEXT DEFAULT '0',
            status TEXT DEFAULT 'open',
            closed_at TEXT
        );
    """)
    for i, (strat, pnl) in enumerate(trades):
        conn.execute(
            "INSERT INTO positions VALUES (?, 'TEST', ?, ?, 'closed', datetime('now'))",
            (str(i), strat, str(pnl)),
        )
    conn.commit()
    return conn


from src.jobs.screener import _performance_factor


class TestPerformanceFactor(unittest.TestCase):
    def test_neutral_no_conn(self):
        assert _performance_factor("rsi_macd", conn=None) == 1.0

    def test_neutral_cold_start_fewer_than_5(self):
        conn = _make_perf_conn([("rsi_macd", 100.0)] * 4)
        assert _performance_factor("rsi_macd", conn) == 1.0

    def test_perfect_win_rate_clamped_at_1_5(self):
        conn = _make_perf_conn([("rsi_macd", 100.0)] * 20)
        result = _performance_factor("rsi_macd", conn)
        assert result == 1.5  # 100% win rate → 1.0/0.5=2.0 → clamped to 1.5

    def test_zero_win_rate_clamped_at_0_5(self):
        conn = _make_perf_conn([("rsi_macd", -100.0)] * 10)
        result = _performance_factor("rsi_macd", conn)
        assert result == 0.5  # 0% win rate → 0/0.5=0 → clamped to 0.5

    def test_neutral_at_50_pct_win_rate(self):
        trades = [("rsi_macd", 100.0)] * 10 + [("rsi_macd", -100.0)] * 10
        conn = _make_perf_conn(trades)
        import pytest
        result = _performance_factor("rsi_macd", conn)
        assert result == pytest.approx(1.0)

    def test_70_pct_win_rate_gives_1_4(self):
        trades = [("rsi_macd", 100.0)] * 7 + [("rsi_macd", -100.0)] * 3
        conn = _make_perf_conn(trades)
        import pytest
        result = _performance_factor("rsi_macd", conn)
        assert result == pytest.approx(1.4)

    def test_isolates_by_strategy_name(self):
        trades = (
            [("rsi_macd", 100.0)] * 10     # 100% win
            + [("zscore_reversion", -50.0)] * 10  # 0% win
        )
        conn = _make_perf_conn(trades)
        assert _performance_factor("rsi_macd", conn) == 1.5
        assert _performance_factor("zscore_reversion", conn) == 0.5

    def test_unknown_strategy_neutral(self):
        conn = _make_perf_conn([("rsi_macd", 100.0)] * 10)
        assert _performance_factor("nonexistent", conn) == 1.0


if __name__ == "__main__":
    unittest.main()
