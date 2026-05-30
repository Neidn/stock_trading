"""Unit tests for MarketShockDetector (and DrawdownGuard)."""

from __future__ import annotations

import sqlite3
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from src.risk.drawdown_guard import DrawdownGuard, MarketShockDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS market_shock_events (
            event_id            TEXT PRIMARY KEY,
            risk_level          TEXT NOT NULL CHECK (risk_level IN ('ELEVATED','DANGER')),
            oi_change_5m        TEXT,
            large_liquidations  TEXT,
            price_change_1m     TEXT,
            funding_rate        TEXT,
            risk_score          INTEGER NOT NULL,
            action_taken        TEXT NOT NULL,
            affected_positions  TEXT,
            created_at          TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS daily_performance (
            perf_date    TEXT NOT NULL,
            trading_mode TEXT NOT NULL DEFAULT 'testnet',
            net_pnl      TEXT DEFAULT '0',
            PRIMARY KEY (perf_date, trading_mode)
        );
        """
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# MarketShockDetector.detect — scoring tests
# ---------------------------------------------------------------------------

class TestMarketShockDetectorDetect(unittest.TestCase):

    # --- NORMAL cases (score < 3) ---

    def test_normal_all_zero(self):
        level = MarketShockDetector.detect(
            oi_change_5m=0.0,
            large_liquidations_5m=0.0,
            price_change_1m=0.0,
            funding_rate=0.0,
        )
        self.assertEqual(level, "NORMAL")

    def test_normal_small_oi_change(self):
        # oi=-1% < threshold for any points; all others 0 → score=0
        level = MarketShockDetector.detect(
            oi_change_5m=-0.01,
            large_liquidations_5m=0.0,
            price_change_1m=0.0,
            funding_rate=0.0,
        )
        self.assertEqual(level, "NORMAL")

    def test_normal_score_2(self):
        # large liq > 1M (+1) + funding > 0.1% (+1) = score 2 → NORMAL
        level = MarketShockDetector.detect(
            oi_change_5m=0.0,
            large_liquidations_5m=1_500_000,
            price_change_1m=0.0,
            funding_rate=0.0015,
        )
        self.assertEqual(level, "NORMAL")

    # --- ELEVATED cases (score == 3 or 4) ---

    def test_elevated_oi_minus_5pct(self):
        # oi < -5% → +3 → score=3 → ELEVATED
        level = MarketShockDetector.detect(
            oi_change_5m=-0.06,
            large_liquidations_5m=0.0,
            price_change_1m=0.0,
            funding_rate=0.0,
        )
        self.assertEqual(level, "ELEVATED")

    def test_elevated_liquidations_over_10m(self):
        # liq > 10M → +3 → ELEVATED
        level = MarketShockDetector.detect(
            oi_change_5m=0.0,
            large_liquidations_5m=15_000_000,
            price_change_1m=0.0,
            funding_rate=0.0,
        )
        self.assertEqual(level, "ELEVATED")

    def test_elevated_score_4(self):
        # oi < -2% (+1) + liq > 1M (+1) + price 3.5% (+2) = score 4 → ELEVATED
        level = MarketShockDetector.detect(
            oi_change_5m=-0.025,
            large_liquidations_5m=2_000_000,
            price_change_1m=0.035,
            funding_rate=0.0,
        )
        self.assertEqual(level, "ELEVATED")

    def test_elevated_price_change_plus_funding(self):
        # price > 3% (+2) + funding (+1) = 3 → ELEVATED
        level = MarketShockDetector.detect(
            oi_change_5m=0.0,
            large_liquidations_5m=0.0,
            price_change_1m=-0.04,
            funding_rate=0.002,
        )
        self.assertEqual(level, "ELEVATED")

    # --- DANGER cases (score >= 5) ---

    def test_danger_oi_and_price(self):
        # oi < -5% (+3) + price > 3% (+2) = score 5 → DANGER
        level = MarketShockDetector.detect(
            oi_change_5m=-0.06,
            large_liquidations_5m=0.0,
            price_change_1m=0.04,
            funding_rate=0.0,
        )
        self.assertEqual(level, "DANGER")

    def test_danger_liquidations_and_price(self):
        # liq > 10M (+3) + price (+2) = 5 → DANGER
        level = MarketShockDetector.detect(
            oi_change_5m=0.0,
            large_liquidations_5m=12_000_000,
            price_change_1m=-0.05,
            funding_rate=0.0,
        )
        self.assertEqual(level, "DANGER")

    def test_danger_all_signals(self):
        # oi < -5% (+3) + liq > 10M (+3) + price (+2) + funding (+1) = 9 → DANGER
        level = MarketShockDetector.detect(
            oi_change_5m=-0.07,
            large_liquidations_5m=20_000_000,
            price_change_1m=0.05,
            funding_rate=0.002,
        )
        self.assertEqual(level, "DANGER")

    def test_danger_score_exactly_5(self):
        # oi < -5% (+3) + liq > 1M (+1) + funding (+1) = 5 → DANGER
        level = MarketShockDetector.detect(
            oi_change_5m=-0.06,
            large_liquidations_5m=2_000_000,
            price_change_1m=0.0,
            funding_rate=0.0015,
        )
        self.assertEqual(level, "DANGER")

    # --- Boundary: OI exactly at threshold ---

    def test_oi_exactly_minus_5pct(self):
        # < -5% boundary — -0.05 is NOT < -0.05 → +0 (not triggered)
        level = MarketShockDetector.detect(
            oi_change_5m=-0.05,
            large_liquidations_5m=0.0,
            price_change_1m=0.0,
            funding_rate=0.0,
        )
        # -0.05 triggers oi < -0.02 (+1) only → score=1 → NORMAL
        self.assertEqual(level, "NORMAL")

    def test_oi_just_below_minus_5pct(self):
        level = MarketShockDetector.detect(
            oi_change_5m=-0.0501,
            large_liquidations_5m=0.0,
            price_change_1m=0.0,
            funding_rate=0.0,
        )
        self.assertEqual(level, "ELEVATED")  # score=3


# ---------------------------------------------------------------------------
# MarketShockDetector.current_level + record_event
# ---------------------------------------------------------------------------

class TestMarketShockDetectorDB(unittest.TestCase):

    def setUp(self):
        self.conn = _make_db()
        self.detector = MarketShockDetector(conn=self.conn)

    def test_current_level_no_events_returns_normal(self):
        self.assertEqual(self.detector.current_level("BTCUSDT"), "NORMAL")

    def test_record_and_retrieve_danger(self):
        self.detector.record_event(
            symbol="BTCUSDT",
            level="DANGER",
            scores={"total_score": 6, "oi_change_5m": -0.07},
            action="reduce_positions",
        )
        self.assertEqual(self.detector.current_level("BTCUSDT"), "DANGER")

    def test_record_and_retrieve_elevated(self):
        self.detector.record_event(
            symbol="BTCUSDT",
            level="ELEVATED",
            scores={"total_score": 3},
            action="watch",
        )
        self.assertEqual(self.detector.current_level("BTCUSDT"), "ELEVATED")

    def test_normal_not_persisted(self):
        # NORMAL must not be stored (schema CHECK would reject it)
        self.detector.record_event(
            symbol="BTCUSDT",
            level="NORMAL",
            scores={"total_score": 0},
            action="none",
        )
        row = self.conn.execute("SELECT COUNT(*) FROM market_shock_events").fetchone()
        self.assertEqual(row[0], 0)

    def test_no_conn_current_level_returns_normal(self):
        detector = MarketShockDetector(conn=None)
        self.assertEqual(detector.current_level("BTCUSDT"), "NORMAL")


# ---------------------------------------------------------------------------
# DrawdownGuard
# ---------------------------------------------------------------------------

class TestDrawdownGuard(unittest.TestCase):

    def setUp(self):
        self.conn = _make_db()

    def _insert_perf(self, perf_date: str, net_pnl: float):
        self.conn.execute(
            "INSERT OR REPLACE INTO daily_performance (perf_date, net_pnl) VALUES (?, ?)",
            (perf_date, str(net_pnl)),
        )
        self.conn.commit()

    # -- Daily limit --

    def test_daily_limit_no_data(self):
        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            self.assertFalse(DrawdownGuard.is_daily_limit_reached(self.conn))

    def test_daily_limit_positive_pnl(self):
        self._insert_perf(date.today().isoformat(), 200.0)
        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            self.assertFalse(DrawdownGuard.is_daily_limit_reached(self.conn))

    def test_daily_limit_exceeded(self):
        # loss 600 / balance 10000 = 6% > 5% default
        self._insert_perf(date.today().isoformat(), -600.0)
        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            self.assertTrue(DrawdownGuard.is_daily_limit_reached(self.conn))

    def test_daily_limit_not_exceeded(self):
        # loss 100 / balance 10000 = 1% < 5%
        self._insert_perf(date.today().isoformat(), -100.0)
        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            self.assertFalse(DrawdownGuard.is_daily_limit_reached(self.conn))

    # -- Weekly limit --

    def test_weekly_limit_no_data(self):
        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            self.assertFalse(DrawdownGuard.is_weekly_limit_reached(self.conn))

    def test_weekly_limit_exceeded(self):
        # default weekly limit = 0.05*3 = 0.15; loss 2100/10000 = 21% > 15%
        # Insert Mon/Tue/Wed of current week so all 3 days fall within week_start window
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        for i in range(3):
            d = (week_start + timedelta(days=i)).isoformat()
            self._insert_perf(d, -700.0)
        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            self.assertTrue(DrawdownGuard.is_weekly_limit_reached(self.conn))

    # -- Profit lock --

    def test_profit_lock_no_profit(self):
        lock, available = DrawdownGuard.check_and_lock_profit(self.conn, 10_000.0, 10_000.0)
        self.assertEqual(lock, 0.0)
        self.assertEqual(available, 10_000.0)

    def test_profit_lock_10pct(self):
        # 10% profit → 50% lock
        lock, available = DrawdownGuard.check_and_lock_profit(self.conn, 11_000.0, 10_000.0)
        self.assertAlmostEqual(lock, 0.50)
        self.assertAlmostEqual(available, 11_000.0 * 0.50)

    def test_profit_lock_20pct(self):
        # 20% profit → 70% lock
        lock, available = DrawdownGuard.check_and_lock_profit(self.conn, 12_000.0, 10_000.0)
        self.assertAlmostEqual(lock, 0.70)
        self.assertAlmostEqual(available, 12_000.0 * 0.30)

    def test_profit_lock_30pct(self):
        # 30% profit → 80% lock
        lock, available = DrawdownGuard.check_and_lock_profit(self.conn, 13_000.0, 10_000.0)
        self.assertAlmostEqual(lock, 0.80)
        self.assertAlmostEqual(available, 13_000.0 * 0.20)

    def test_profit_lock_zero_initial_balance(self):
        lock, available = DrawdownGuard.check_and_lock_profit(self.conn, 10_000.0, 0.0)
        self.assertEqual(lock, 0.0)
        self.assertEqual(available, 10_000.0)


if __name__ == "__main__":
    unittest.main()
