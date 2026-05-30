"""Unit tests for SignalBlocker.

All external dependencies (DB, DataGapDetector, balance cache) are replaced
with lightweight fakes — no SQLite file, no network calls.
"""

from __future__ import annotations

import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.signal.signal_blocker import (
    EconomicCalendar,
    MarketShockDetector,
    SafeMode,
    SignalBlocker,
    _dynamic_max_positions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_in_memory_db() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with the minimal schema needed."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS positions (
            position_id  TEXT PRIMARY KEY,
            symbol       TEXT NOT NULL,
            status       TEXT NOT NULL,
            side         TEXT,
            entry_price  TEXT,
            exit_price   TEXT,
            quantity     TEXT,
            closed_at    TEXT DEFAULT NULL,
            realized_pnl TEXT DEFAULT '0'
        );
        CREATE TABLE IF NOT EXISTS market_shock_events (
            event_id           TEXT PRIMARY KEY,
            risk_level         TEXT NOT NULL,
            created_at         TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    return conn


def _insert_position(
    conn: sqlite3.Connection,
    position_id: str,
    symbol: str,
    status: str = "open",
    side: str | None = None,
    entry_price: str | None = None,
    exit_price: str | None = None,
    quantity: str | None = None,
    closed_at: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO positions (position_id, symbol, status, side, entry_price, exit_price,"
        " quantity, closed_at) VALUES (?,?,?,?,?,?,?,?)",
        (position_id, symbol, status, side, entry_price, exit_price, quantity, closed_at),
    )
    conn.commit()


def _make_gap_detector(has_gap: bool = False) -> MagicMock:
    detector = MagicMock()
    detector.has_gap.return_value = has_gap
    return detector


def _make_blocker(
    conn: sqlite3.Connection | None = None,
    has_gap: bool = False,
) -> SignalBlocker:
    return SignalBlocker(
        conn=conn or _make_in_memory_db(),
        gap_detector=_make_gap_detector(has_gap),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSignalBlocker(unittest.TestCase):

    def setUp(self) -> None:
        SafeMode.deactivate()

    # 1. Data gap
    def test_blocks_on_data_gap(self) -> None:
        blocker = _make_blocker(has_gap=True)
        blocked, reason = blocker.should_block("BTCUSDT")
        self.assertTrue(blocked)
        self.assertIn("data_gap", reason)

    # 2. Safe mode
    def test_blocks_when_safe_mode_active(self) -> None:
        SafeMode.activate("emergency")
        blocker = _make_blocker()
        blocked, reason = blocker.should_block("BTCUSDT")
        self.assertTrue(blocked)
        self.assertIn("safe_mode", reason)

    # 3. Daily loss limit
    def test_blocks_on_daily_loss_limit_exceeded(self) -> None:
        conn = _make_in_memory_db()
        # Long position: bought at 100, sold at 90 → loss = (90-100)*10 = -100 USDT
        # balance 1000, limit 3% → 100/1000 = 10% > 3%
        _insert_position(conn, "pos-1", "BTCUSDT", status="closed", side="long",
                         entry_price="100.0", exit_price="90.0", quantity="10.0",
                         closed_at="2026-01-01T00:00:00+00:00")
        # Use today's date for closed_at
        from datetime import datetime, timezone
        today_ts = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE positions SET closed_at=? WHERE position_id='pos-1'", (today_ts,)
        )
        conn.commit()

        blocker = _make_blocker(conn=conn)
        config = SimpleNamespace(daily_loss_limit=0.03, max_positions=3)
        with (
            patch("src.signal.signal_blocker.load_config", return_value=config),
            patch(
                "src.utils.startup_recovery.get_cached_balance",
                return_value={"availableBalance": 1_000.0},
            ),
        ):
            blocked, reason = blocker.should_block("ETHUSDT")

        self.assertTrue(blocked)
        self.assertIn("daily_loss_limit", reason)

    # 4. Market shock
    def test_blocks_on_market_shock_elevated(self) -> None:
        blocker = _make_blocker()
        with patch.object(MarketShockDetector, "current_level", return_value="ELEVATED"):
            blocked, reason = blocker.should_block("BTCUSDT")
        self.assertTrue(blocked)
        self.assertIn("market_shock", reason)

    def test_blocks_on_market_shock_danger(self) -> None:
        blocker = _make_blocker()
        with patch.object(MarketShockDetector, "current_level", return_value="DANGER"):
            blocked, reason = blocker.should_block("BTCUSDT")
        self.assertTrue(blocked)
        self.assertIn("market_shock", reason)

    # 5. Economic event
    def test_blocks_on_economic_event(self) -> None:
        blocker = _make_blocker()
        with patch.object(EconomicCalendar, "is_high_impact_event_soon", return_value=True):
            blocked, reason = blocker.should_block("BTCUSDT")
        self.assertTrue(blocked)
        self.assertIn("economic_event", reason)

    # 6. Open position
    def test_blocks_when_open_position_exists(self) -> None:
        conn = _make_in_memory_db()
        _insert_position(conn, "pos-1", "BTCUSDT", status="open")
        blocker = _make_blocker(conn=conn)
        blocked, reason = blocker.should_block("BTCUSDT")
        self.assertTrue(blocked)
        self.assertIn("open_position", reason)

    # 7. All conditions pass → no block
    def test_passes_when_no_conditions_met(self) -> None:
        blocker = _make_blocker()
        with patch(
            "src.utils.startup_recovery.get_cached_balance",
            return_value={"availableBalance": 10_000.0},
        ):
            blocked, reason = blocker.should_block("BTCUSDT")
        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    # Edge: different symbol open position should not block target symbol
    def test_open_position_other_symbol_does_not_block(self) -> None:
        conn = _make_in_memory_db()
        _insert_position(conn, "pos-2", "ETHUSDT", status="open")
        blocker = _make_blocker(conn=conn)
        with patch(
            "src.utils.startup_recovery.get_cached_balance",
            return_value={"availableBalance": 10_000.0},
        ):
            blocked, _ = blocker.should_block("BTCUSDT")
        self.assertFalse(blocked)

    def test_blocks_when_max_positions_reached(self) -> None:
        conn = _make_in_memory_db()
        _insert_position(conn, "pos-1", "ETHUSDT", status="open")
        _insert_position(conn, "pos-2", "SOLUSDT", status="open")
        blocker = _make_blocker(conn=conn)
        config = SimpleNamespace(max_positions=2, daily_loss_limit=0.03)

        with (
            patch.object(MarketShockDetector, "current_level", return_value="NORMAL"),
            patch.object(EconomicCalendar, "is_high_impact_event_soon", return_value=False),
            patch("src.signal.signal_blocker.load_config", return_value=config),
        ):
            blocked, reason = blocker.should_block("BTCUSDT")

        self.assertTrue(blocked)
        self.assertIn("max_positions", reason)

    def test_allows_when_below_max_positions(self) -> None:
        conn = _make_in_memory_db()
        _insert_position(conn, "pos-1", "ETHUSDT", status="open")
        blocker = _make_blocker(conn=conn)
        config = SimpleNamespace(max_positions=2, daily_loss_limit=0.03)

        with (
            patch.object(MarketShockDetector, "current_level", return_value="NORMAL"),
            patch.object(EconomicCalendar, "is_high_impact_event_soon", return_value=False),
            patch("src.signal.signal_blocker.load_config", return_value=config),
        ):
            blocked, reason = blocker.should_block("BTCUSDT")

        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    # Edge: daily loss within limit → no block
    def test_daily_loss_within_limit_does_not_block(self) -> None:
        from datetime import datetime, timezone
        conn = _make_in_memory_db()
        today_ts = datetime.now(timezone.utc).isoformat()
        # Long: bought at 100, sold at 99 → loss = 1 * 10 = 10 USDT; 10/10000 = 0.1% < 3%
        _insert_position(conn, "pos-1", "BTCUSDT", status="closed", side="long",
                         entry_price="100.0", exit_price="99.0", quantity="10.0",
                         closed_at=today_ts)
        blocker = _make_blocker(conn=conn)
        config = SimpleNamespace(daily_loss_limit=0.03, max_positions=3)
        with (
            patch("src.signal.signal_blocker.load_config", return_value=config),
            patch(
                "src.utils.startup_recovery.get_cached_balance",
                return_value={"availableBalance": 10_000.0},
            ),
        ):
            blocked, _ = blocker.should_block("BTCUSDT")
        self.assertFalse(blocked)

    # Edge: profitable close today → no block (gains don't count as loss)
    def test_profitable_close_does_not_block(self) -> None:
        from datetime import datetime, timezone
        conn = _make_in_memory_db()
        today_ts = datetime.now(timezone.utc).isoformat()
        # Long: bought at 100, sold at 120 → profit = 200 USDT → should NOT block
        _insert_position(conn, "pos-1", "BTCUSDT", status="closed", side="long",
                         entry_price="100.0", exit_price="120.0", quantity="10.0",
                         closed_at=today_ts)
        blocker = _make_blocker(conn=conn)
        config = SimpleNamespace(daily_loss_limit=0.03, max_positions=3)
        with (
            patch("src.signal.signal_blocker.load_config", return_value=config),
            patch(
                "src.utils.startup_recovery.get_cached_balance",
                return_value={"availableBalance": 1_000.0},
            ),
        ):
            blocked, _ = blocker.should_block("BTCUSDT")
        self.assertFalse(blocked)

    # Edge: short position loss
    def test_blocks_short_position_loss(self) -> None:
        from datetime import datetime, timezone
        conn = _make_in_memory_db()
        today_ts = datetime.now(timezone.utc).isoformat()
        # Short: sold at 100, bought back at 115 → loss = (100-115)*10 = -150 USDT
        # balance 1000, limit 3% → 150/1000 = 15% > 3%
        _insert_position(conn, "pos-1", "BTCUSDT", status="closed", side="short",
                         entry_price="100.0", exit_price="115.0", quantity="10.0",
                         closed_at=today_ts)
        blocker = _make_blocker(conn=conn)
        config = SimpleNamespace(daily_loss_limit=0.03, max_positions=3)
        with (
            patch("src.signal.signal_blocker.load_config", return_value=config),
            patch(
                "src.utils.startup_recovery.get_cached_balance",
                return_value={"availableBalance": 1_000.0},
            ),
        ):
            blocked, reason = blocker.should_block("ETHUSDT")
        self.assertTrue(blocked)
        self.assertIn("daily_loss_limit", reason)

    # Edge: no exit_price set → not counted (strategy exit without fill price)
    def test_closed_without_exit_price_not_counted(self) -> None:
        from datetime import datetime, timezone
        conn = _make_in_memory_db()
        today_ts = datetime.now(timezone.utc).isoformat()
        _insert_position(conn, "pos-1", "BTCUSDT", status="closed", side="long",
                         entry_price="100.0", exit_price=None, quantity="100.0",
                         closed_at=today_ts)
        blocker = _make_blocker(conn=conn)
        config = SimpleNamespace(daily_loss_limit=0.03, max_positions=3)
        with (
            patch("src.signal.signal_blocker.load_config", return_value=config),
            patch(
                "src.utils.startup_recovery.get_cached_balance",
                return_value={"availableBalance": 100.0},
            ),
        ):
            blocked, _ = blocker.should_block("BTCUSDT")
        self.assertFalse(blocked)

    # Edge: no balance cached → fallback to nominal limit
    def test_blocks_no_balance_cache_fallback(self) -> None:
        from datetime import datetime, timezone
        conn = _make_in_memory_db()
        today_ts = datetime.now(timezone.utc).isoformat()
        # Loss of 400 USDT; nominal limit = 0.03 * 10_000 = 300 → 400 > 300 → block
        _insert_position(conn, "pos-1", "BTCUSDT", status="closed", side="long",
                         entry_price="100.0", exit_price="60.0", quantity="10.0",
                         closed_at=today_ts)
        blocker = _make_blocker(conn=conn)
        config = SimpleNamespace(daily_loss_limit=0.03, max_positions=3)
        with (
            patch("src.signal.signal_blocker.load_config", return_value=config),
            patch(
                "src.utils.startup_recovery.get_cached_balance",
                return_value={"availableBalance": 0},
            ),
        ):
            blocked, reason = blocker.should_block("ETHUSDT")
        self.assertTrue(blocked)
        self.assertIn("no balance cache", reason)


# ---------------------------------------------------------------------------
# Dynamic MAX_POSITIONS (regime-adaptive slot reduction)
# ---------------------------------------------------------------------------

class TestDynamicMaxPositions:
    def test_trending_returns_base(self):
        assert _dynamic_max_positions(30.0, 5) == 5

    def test_at_threshold_returns_base(self):
        assert _dynamic_max_positions(25.0, 5) == 5

    def test_weak_trend_reduces_by_one(self):
        assert _dynamic_max_positions(22.0, 5) == 4

    def test_ranging_reduces_by_two(self):
        assert _dynamic_max_positions(15.0, 5) == 3

    def test_none_adx_returns_base(self):
        assert _dynamic_max_positions(None, 5) == 5

    def test_floor_at_one(self):
        assert _dynamic_max_positions(0.0, 1) == 1
        assert _dynamic_max_positions(5.0, 2) == 1

    def test_base_three_ranging(self):
        # base=3, ranging → max(1, 3-2)=1
        assert _dynamic_max_positions(10.0, 3) == 1


class TestSignalBlockerDynamicLimit:
    def _make_conn_with_open(self, n: int) -> sqlite3.Connection:
        conn = _make_in_memory_db()
        for i in range(n):
            _insert_position(conn, f"pos-{i}", f"COIN{i}USDT", status="open")
        return conn

    def _mock_env(self, base: int = 5):
        return SimpleNamespace(max_positions=base, daily_loss_limit=0.03)

    def test_ranging_adx_tightens_limit(self):
        # 4 open positions; base=5; ADX=15 → limit=3 → should block
        conn = self._make_conn_with_open(4)
        blocker = SignalBlocker(conn=conn, gap_detector=_make_gap_detector(), btc_adx=15.0)
        config = self._mock_env(base=5)
        with (
            patch.object(MarketShockDetector, "current_level", return_value="NORMAL"),
            patch.object(EconomicCalendar, "is_high_impact_event_soon", return_value=False),
            patch("src.signal.signal_blocker.load_config", return_value=config),
        ):
            blocked, reason = blocker.should_block("NEWUSDT")
        assert blocked
        assert "max_positions" in reason
        assert "ADX=15.0" in reason

    def test_ranging_adx_allows_under_dynamic_limit(self):
        # 2 open positions; base=5; ADX=15 → limit=3 → should NOT block
        conn = self._make_conn_with_open(2)
        blocker = SignalBlocker(conn=conn, gap_detector=_make_gap_detector(), btc_adx=15.0)
        config = self._mock_env(base=5)
        with (
            patch.object(MarketShockDetector, "current_level", return_value="NORMAL"),
            patch.object(EconomicCalendar, "is_high_impact_event_soon", return_value=False),
            patch("src.signal.signal_blocker.load_config", return_value=config),
        ):
            blocked, _ = blocker.should_block("NEWUSDT")
        assert not blocked

    def test_trending_adx_uses_full_limit(self):
        # 4 open positions; base=5; ADX=30 → limit=5 → should NOT block
        conn = self._make_conn_with_open(4)
        blocker = SignalBlocker(conn=conn, gap_detector=_make_gap_detector(), btc_adx=30.0)
        config = self._mock_env(base=5)
        with (
            patch.object(MarketShockDetector, "current_level", return_value="NORMAL"),
            patch.object(EconomicCalendar, "is_high_impact_event_soon", return_value=False),
            patch("src.signal.signal_blocker.load_config", return_value=config),
        ):
            blocked, _ = blocker.should_block("NEWUSDT")
        assert not blocked

    def test_no_btc_adx_uses_base_limit(self):
        # None btc_adx → base limit unchanged
        conn = self._make_conn_with_open(5)
        blocker = SignalBlocker(conn=conn, gap_detector=_make_gap_detector(), btc_adx=None)
        config = self._mock_env(base=5)
        with (
            patch.object(MarketShockDetector, "current_level", return_value="NORMAL"),
            patch.object(EconomicCalendar, "is_high_impact_event_soon", return_value=False),
            patch("src.signal.signal_blocker.load_config", return_value=config),
        ):
            blocked, _ = blocker.should_block("NEWUSDT")
        assert blocked

    def test_weak_trend_reduces_by_one(self):
        # 4 open; base=5; ADX=22 → limit=4 → blocked
        conn = self._make_conn_with_open(4)
        blocker = SignalBlocker(conn=conn, gap_detector=_make_gap_detector(), btc_adx=22.0)
        config = self._mock_env(base=5)
        with (
            patch.object(MarketShockDetector, "current_level", return_value="NORMAL"),
            patch.object(EconomicCalendar, "is_high_impact_event_soon", return_value=False),
            patch("src.signal.signal_blocker.load_config", return_value=config),
        ):
            blocked, reason = blocker.should_block("NEWUSDT")
        assert blocked
        assert "ADX=22.0" in reason


if __name__ == "__main__":
    unittest.main()
