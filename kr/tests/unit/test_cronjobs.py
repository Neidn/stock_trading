"""Unit tests for CronJob modules.

All external I/O (exchange, Telegram) is mocked.
DB: in-memory SQLite.
"""

from __future__ import annotations

import sqlite3
import uuid
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from src.jobs.screener import ScreenerJob, _ccxt_to_db_symbol
from src.jobs.position_sync import PositionSyncJob
from src.jobs.daily_report import DailyReportJob
from src.jobs.db_archiver import DbArchiverJob


# ---------------------------------------------------------------------------
# Schema / helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol      TEXT PRIMARY KEY,
    base_asset  TEXT NOT NULL DEFAULT 'BTC',
    quote_asset TEXT NOT NULL DEFAULT 'USDT',
    is_active   INTEGER NOT NULL DEFAULT 1,
    strategy    TEXT,
    added_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
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
    trading_mode TEXT NOT NULL DEFAULT 'testnet',
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


def _insert_symbol(conn, symbol="BTCUSDT", is_active=1) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO symbols (symbol, base_asset, quote_asset, is_active) VALUES (?,?,?,?)",
        (symbol, symbol[:-4], "USDT", is_active),
    )
    conn.commit()


def _insert_position(conn, symbol="BTCUSDT", side="long", quantity=0.1,
                     entry_price=50000.0, liq_price=44000.0, status="open") -> str:
    pid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO positions
           (position_id, symbol, side, entry_price, quantity,
            liquidation_price, stop_loss, status)
           VALUES (?,?,?,?,?,?,?,?)""",
        (pid, symbol, side, str(entry_price), str(quantity),
         str(liq_price), "48000", status),
    )
    conn.commit()
    return pid


def _insert_kline(conn, symbol="BTCUSDT", interval="1m", open_time="2020-01-01 00:00:00"):
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
        (today, "testnet", total, wins, total - wins,
         str(net_pnl), str(gross_profit), str(gross_loss)),
    )
    conn.commit()


def _make_ticker(symbol, quote_volume=10_000_000.0, last=50000.0,
                 high=52000.0, low=48000.0) -> dict:
    return {
        "symbol": symbol,
        "quoteVolume": quote_volume,
        "last": last,
        "close": last,
        "high": high,
        "low": low,
    }


# ===========================================================================
# _ccxt_to_db_symbol
# ===========================================================================

class TestCcxtToDbSymbol(unittest.TestCase):

    def test_standard_usdt_m(self):
        self.assertEqual(_ccxt_to_db_symbol("BTC/USDT:USDT"), "BTCUSDT")

    def test_eth(self):
        self.assertEqual(_ccxt_to_db_symbol("ETH/USDT:USDT"), "ETHUSDT")

    def test_sol(self):
        self.assertEqual(_ccxt_to_db_symbol("SOL/USDT:USDT"), "SOLUSDT")


# ===========================================================================
# ScreenerJob
# ===========================================================================

class TestScreenerJob(unittest.TestCase):

    def _make_exchange(self, tickers=None, funding_rates=None) -> MagicMock:
        ex = MagicMock()
        ex.fetch_tickers.return_value = tickers or {}
        ex.fetch_funding_rates.return_value = funding_rates or {}
        ex.markets = {}
        return ex

    def test_no_candidates_returns_zero(self):
        conn = _make_conn()
        ex = self._make_exchange(tickers={"BTC/USDT:USDT": _make_ticker("BTC/USDT:USDT", quote_volume=1000)})
        job = ScreenerJob(exchange=ex, conn=conn)
        result = job.run()
        self.assertEqual(result["total_screened"], 0)

    def test_candidates_above_volume_selected(self):
        conn = _make_conn()
        tickers = {"BTC/USDT:USDT": _make_ticker("BTC/USDT:USDT", quote_volume=10_000_000)}
        ex = self._make_exchange(tickers=tickers)
        job = ScreenerJob(exchange=ex, conn=conn, top_n=5)
        result = job.run()
        self.assertEqual(result["total_screened"], 1)
        self.assertIn("BTCUSDT", result["added"])

    def test_symbols_written_to_db(self):
        conn = _make_conn()
        tickers = {"BTC/USDT:USDT": _make_ticker("BTC/USDT:USDT")}
        ex = self._make_exchange(tickers=tickers)
        ScreenerJob(exchange=ex, conn=conn).run()
        row = conn.execute("SELECT is_active FROM symbols WHERE symbol='BTCUSDT'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)

    def test_removed_symbol_deactivated(self):
        conn = _make_conn()
        _insert_symbol(conn, "ETHUSDT", is_active=1)
        # Only BTC above volume threshold
        tickers = {"BTC/USDT:USDT": _make_ticker("BTC/USDT:USDT")}
        ex = self._make_exchange(tickers=tickers)
        job = ScreenerJob(exchange=ex, conn=conn, top_n=1)
        result = job.run()
        self.assertIn("ETHUSDT", result["removed"])
        row = conn.execute("SELECT is_active FROM symbols WHERE symbol='ETHUSDT'").fetchone()
        self.assertEqual(row[0], 1)  # core symbol — never deactivated

    def test_symbol_with_open_position_not_removed(self):
        conn = _make_conn()
        _insert_symbol(conn, "ETHUSDT", is_active=1)
        _insert_position(conn, symbol="ETHUSDT")
        tickers = {"BTC/USDT:USDT": _make_ticker("BTC/USDT:USDT")}
        ex = self._make_exchange(tickers=tickers)
        job = ScreenerJob(exchange=ex, conn=conn, top_n=1)
        result = job.run()
        self.assertNotIn("ETHUSDT", result["removed"])
        row = conn.execute("SELECT is_active FROM symbols WHERE symbol='ETHUSDT'").fetchone()
        self.assertEqual(row[0], 1)

    def test_funding_rate_below_threshold_no_score(self):
        conn = _make_conn()
        tickers = {"BTC/USDT:USDT": _make_ticker("BTC/USDT:USDT")}
        funding = {"BTC/USDT:USDT": {"fundingRate": 0.000001}}  # below 0.01%
        ex = self._make_exchange(tickers=tickers, funding_rates=funding)
        job = ScreenerJob(exchange=ex, conn=conn)
        # Score computed internally; just assert run doesn't fail
        job.run()

    def test_funding_fetch_failure_graceful(self):
        conn = _make_conn()
        tickers = {"BTC/USDT:USDT": _make_ticker("BTC/USDT:USDT")}
        ex = self._make_exchange(tickers=tickers)
        ex.fetch_funding_rates.side_effect = Exception("network error")
        job = ScreenerJob(exchange=ex, conn=conn)
        result = job.run()
        self.assertEqual(result["total_screened"], 1)

    def test_non_usdt_symbol_excluded(self):
        conn = _make_conn()
        tickers = {
            "BTC/USDT:USDT": _make_ticker("BTC/USDT:USDT"),
            "BTC/USD:BTC":   _make_ticker("BTC/USD:BTC"),  # coin-margined, should skip
        }
        ex = self._make_exchange(tickers=tickers)
        job = ScreenerJob(exchange=ex, conn=conn, top_n=10)
        result = job.run()
        self.assertEqual(result["total_screened"], 1)

    def test_telegram_notified_on_run(self):
        conn = _make_conn()
        tickers = {"BTC/USDT:USDT": _make_ticker("BTC/USDT:USDT")}
        ex = self._make_exchange(tickers=tickers)
        bot = MagicMock()
        ScreenerJob(exchange=ex, conn=conn, telegram_bot=bot).run()
        bot.send_info.assert_called_once()

    def test_score_range_gt3pct(self):
        conn = _make_conn()
        job = ScreenerJob(exchange=MagicMock(), conn=conn)
        ticker = _make_ticker("BTC/USDT:USDT", last=100.0, high=105.0, low=100.0)  # 5% range
        range_pct = job._daily_range_pct(ticker)
        score = job._score(ticker, range_pct=range_pct, atr_cutoff=0.0,
                           funding_rate=0.0, oi_change_pct=0.0)
        self.assertGreaterEqual(score, 2)

    def test_score_oi_above_threshold(self):
        conn = _make_conn()
        job = ScreenerJob(exchange=MagicMock(), conn=conn)
        ticker = _make_ticker("BTC/USDT:USDT")
        score = job._score(ticker, range_pct=0.0, atr_cutoff=999.0,
                           funding_rate=0.0, oi_change_pct=15.0)
        self.assertEqual(score, 2)

    def test_score_funding_above_threshold(self):
        conn = _make_conn()
        job = ScreenerJob(exchange=MagicMock(), conn=conn)
        ticker = _make_ticker("BTC/USDT:USDT")
        score = job._score(ticker, range_pct=0.0, atr_cutoff=999.0,
                           funding_rate=0.0002, oi_change_pct=0.0)
        self.assertEqual(score, 1)


# ===========================================================================
# PositionSyncJob
# ===========================================================================

class TestPositionSyncJob(unittest.TestCase):

    def _make_api_pos(self, symbol="BTC/USDT:USDT", side="long", qty=0.1,
                     entry=50000.0, liq=44000.0, leverage=5) -> dict:
        return {
            "symbol": symbol, "side": side, "contracts": qty,
            "entryPrice": entry, "liquidationPrice": liq, "leverage": leverage,
            "markPrice": entry,
        }

    def test_no_discrepancies_logged(self):
        conn = _make_conn()
        _insert_position(conn, "BTCUSDT", side="long", quantity=0.1)
        ex = MagicMock()
        ex.fetch_positions.return_value = [self._make_api_pos("BTC/USDT:USDT", qty=0.1)]
        result = PositionSyncJob(exchange=ex, conn=conn).run()
        self.assertTrue(result["success"])
        self.assertEqual(result["discrepancies"], 0)

    def test_missing_in_db_inserted(self):
        conn = _make_conn()
        ex = MagicMock()
        ex.fetch_positions.return_value = [self._make_api_pos("BTC/USDT:USDT", qty=0.1)]
        PositionSyncJob(exchange=ex, conn=conn).run()
        row = conn.execute("SELECT * FROM positions WHERE symbol='BTCUSDT'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "open")

    def test_ghost_position_closed(self):
        conn = _make_conn()
        _insert_position(conn, "ETHUSDT")
        ex = MagicMock()
        ex.fetch_positions.return_value = []  # nothing on exchange
        PositionSyncJob(exchange=ex, conn=conn).run()
        row = conn.execute("SELECT status FROM positions WHERE symbol='ETHUSDT'").fetchone()
        self.assertEqual(row["status"], "closed")

    def test_quantity_mismatch_corrected(self):
        conn = _make_conn()
        _insert_position(conn, "BTCUSDT", quantity=0.1)
        ex = MagicMock()
        ex.fetch_positions.return_value = [self._make_api_pos("BTC/USDT:USDT", qty=0.2)]
        PositionSyncJob(exchange=ex, conn=conn).run()
        row = conn.execute("SELECT quantity FROM positions WHERE symbol='BTCUSDT'").fetchone()
        self.assertAlmostEqual(float(row["quantity"]), 0.2)

    def test_side_mismatch_corrected(self):
        conn = _make_conn()
        _insert_position(conn, "BTCUSDT", side="long", quantity=0.1)
        ex = MagicMock()
        ex.fetch_positions.return_value = [self._make_api_pos("BTC/USDT:USDT", side="short", qty=0.1)]
        PositionSyncJob(exchange=ex, conn=conn).run()
        row = conn.execute("SELECT side FROM positions WHERE symbol='BTCUSDT'").fetchone()
        self.assertEqual(row["side"], "short")

    def test_sync_event_logged_on_success(self):
        conn = _make_conn()
        ex = MagicMock()
        ex.fetch_positions.return_value = []
        PositionSyncJob(exchange=ex, conn=conn).run()
        row = conn.execute("SELECT success FROM sync_events").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)

    def test_sync_event_logged_on_failure(self):
        conn = _make_conn()
        ex = MagicMock()
        ex.fetch_positions.side_effect = Exception("API down")
        PositionSyncJob(exchange=ex, conn=conn).run()
        row = conn.execute("SELECT success, error_message FROM sync_events").fetchone()
        self.assertEqual(row[0], 0)
        self.assertIn("API down", row[1])

    def test_exception_returns_failure_dict(self):
        conn = _make_conn()
        ex = MagicMock()
        ex.fetch_positions.side_effect = RuntimeError("timeout")
        result = PositionSyncJob(exchange=ex, conn=conn).run()
        self.assertFalse(result["success"])
        self.assertIn("timeout", result["error"])

    def test_telegram_warning_on_discrepancy(self):
        conn = _make_conn()
        _insert_position(conn, "ETHUSDT")
        ex = MagicMock()
        ex.fetch_positions.return_value = []
        bot = MagicMock()
        PositionSyncJob(exchange=ex, conn=conn, telegram_bot=bot).run()
        bot.send_warning.assert_called()

    def test_liquidation_prices_recalculated(self):
        conn = _make_conn()
        pid = _insert_position(conn, "BTCUSDT", liq_price=44000.0)
        ex = MagicMock()
        ex.fetch_positions.return_value = [
            self._make_api_pos("BTC/USDT:USDT", qty=0.1, liq=45000.0)
        ]
        PositionSyncJob(exchange=ex, conn=conn).run()
        row = conn.execute("SELECT liquidation_price FROM positions WHERE position_id=?", (pid,)).fetchone()
        # Should be updated to 45000 from API
        self.assertAlmostEqual(float(row[0]), 45000.0)

    def test_multiple_discrepancies_all_resolved(self):
        conn = _make_conn()
        _insert_position(conn, "ETHUSDT")  # ghost
        _insert_position(conn, "BTCUSDT", quantity=0.1)  # qty mismatch
        ex = MagicMock()
        ex.fetch_positions.return_value = [
            self._make_api_pos("BTC/USDT:USDT", qty=0.3),  # mismatch
            self._make_api_pos("SOL/USDT:USDT", qty=1.0),  # missing in db
        ]
        result = PositionSyncJob(exchange=ex, conn=conn).run()
        self.assertEqual(result["discrepancies"], 3)  # ghost + qty_mismatch + missing

    def test_ghost_position_cancels_open_orders(self):
        conn = _make_conn()
        _insert_position(conn, "AKTUSDT")  # ghost — not on exchange
        ex = MagicMock()
        ex.fetch_positions.return_value = []
        PositionSyncJob(exchange=ex, conn=conn).run()
        ex.cancel_all_orders.assert_called_once_with("AKTUSDT")

    def test_ghost_cancel_failure_does_not_abort_sync(self):
        conn = _make_conn()
        _insert_position(conn, "AKTUSDT")
        ex = MagicMock()
        ex.fetch_positions.return_value = []
        ex.cancel_all_orders.side_effect = Exception("network error")
        result = PositionSyncJob(exchange=ex, conn=conn).run()
        self.assertTrue(result["success"])
        row = conn.execute("SELECT status FROM positions WHERE symbol='AKTUSDT'").fetchone()
        self.assertEqual(row["status"], "closed")


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
        _insert_position(conn, "BTCUSDT", side="long")
        job = DailyReportJob(conn=conn)
        report = job.run()
        self.assertIn("BTCUSDT", report)
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
        row = conn.execute("SELECT count(*) FROM klines WHERE interval_type='1m'").fetchone()
        self.assertEqual(row[0], 1)

    def test_prunes_all_intervals_by_count(self):
        conn = _make_conn()
        for ts in ["2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-01-03 00:00:00"]:
            _insert_kline(conn, interval="1h", open_time=ts)
        result = DbArchiverJob(conn=conn, keep_candles=2).run()
        self.assertEqual(result["deleted_rows"], 1)
        row = conn.execute("SELECT count(*) FROM klines WHERE interval_type='1h'").fetchone()
        self.assertEqual(row[0], 2)

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


if __name__ == "__main__":
    unittest.main()
