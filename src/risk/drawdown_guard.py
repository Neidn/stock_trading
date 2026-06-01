"""Drawdown guard and market shock detector.

MarketShockDetector:  score-based shock classification + DB persistence.
DrawdownGuard:        daily/weekly loss limits + profit-lock enforcement.

KRX context: shock metrics (OI, funding) don't apply to spot stocks;
MarketShockDetector is populated externally if needed.  DrawdownGuard
uses KRW balances from the startup_recovery balance cache.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import date, timedelta

from src.monitoring.logger import get_logger

logger = get_logger("drawdown_guard")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DANGER_THRESHOLD = 5
_ELEVATED_THRESHOLD = 3

# Profit-lock tiers: (profit_pct_floor, lock_fraction)
_PROFIT_LOCK_TIERS = [
    (0.30, 0.80),
    (0.20, 0.70),
    (0.10, 0.50),
]

_SHOCK_LOOKBACK_MINUTES = 5


# ---------------------------------------------------------------------------
# MarketShockDetector
# ---------------------------------------------------------------------------

class MarketShockDetector:
    """Score-based market shock classifier.

    Score table (generic; not all metrics apply to KRX spot):
        OI 5-min change < -5%     → +3
        OI 5-min change < -2%     → +1
        Large drops > 10B KRW     → +3
        Large drops > 1B KRW      → +1
        |price_change_1m| > 3%    → +2
        |rate| > 0.1%             → +1

    Levels:
        score >= 5  → DANGER
        score >= 3  → ELEVATED
        otherwise   → NORMAL
    """

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self._conn = conn

    @staticmethod
    def detect(
        oi_change_5m: float,
        large_liquidations_5m: float,
        price_change_1m: float,
        funding_rate: float,
    ) -> str:
        """Classify market shock level from raw metrics.

        Returns 'NORMAL' | 'ELEVATED' | 'DANGER'.
        """
        score = 0

        if oi_change_5m < -0.05:
            score += 3
        elif oi_change_5m < -0.02:
            score += 1

        if large_liquidations_5m > 10_000_000:
            score += 3
        elif large_liquidations_5m > 1_000_000:
            score += 1

        if abs(price_change_1m) > 0.03:
            score += 2

        if abs(funding_rate) > 0.001:
            score += 1

        if score >= _DANGER_THRESHOLD:
            return "DANGER"
        if score >= _ELEVATED_THRESHOLD:
            return "ELEVATED"
        return "NORMAL"

    def current_level(self, symbol: str) -> str:  # noqa: ARG002
        """Return most recent shock level within last 5 minutes from DB."""
        if self._conn is None:
            return "NORMAL"

        row = self._conn.execute(
            """
            SELECT risk_level FROM market_shock_events
            WHERE created_at >= datetime('now', ?)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (f"-{_SHOCK_LOOKBACK_MINUTES} minutes",),
        ).fetchone()

        if row is None:
            return "NORMAL"
        return row[0] if isinstance(row, (list, tuple)) else row["risk_level"]

    def record_event(self, symbol: str, level: str, scores: dict, action: str) -> None:
        """Persist a shock event to market_shock_events."""
        if level not in ("ELEVATED", "DANGER"):
            return
        if self._conn is None:
            return

        total_score = scores.get("total_score", 0)
        self._conn.execute(
            """
            INSERT INTO market_shock_events
                (event_id, risk_level, oi_change_5m, large_liquidations,
                 price_change_1m, funding_rate, risk_score, action_taken, affected_positions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()), level,
                str(scores.get("oi_change_5m", "")),
                str(scores.get("large_liquidations_5m", "")),
                str(scores.get("price_change_1m", "")),
                str(scores.get("funding_rate", "")),
                int(total_score), action, symbol,
            ),
        )
        self._conn.commit()
        logger.warning("Market shock recorded [%s]: level=%s score=%d", symbol, level, total_score)


# ---------------------------------------------------------------------------
# DrawdownGuard
# ---------------------------------------------------------------------------

class DrawdownGuard:
    """Enforce daily/weekly loss limits and profit-lock rules."""

    @staticmethod
    def is_daily_limit_reached(conn: sqlite3.Connection) -> bool:
        """Return True if today's net_pnl is at or below the daily loss limit."""
        from src.utils.config import load_config
        config = load_config()
        limit_fraction = config.daily_loss_limit

        today = date.today().isoformat()
        row = conn.execute(
            "SELECT net_pnl FROM daily_performance WHERE perf_date = ?",
            (today,),
        ).fetchone()

        if row is None:
            return False

        net_pnl = float(row[0] if isinstance(row, (list, tuple)) else row["net_pnl"] or 0)
        if net_pnl >= 0:
            return False

        from src.utils.startup_recovery import get_cached_balance
        balance = float(get_cached_balance().get("availableBalance", 0) or 0)

        if balance > 0:
            return abs(net_pnl) / balance >= limit_fraction
        else:
            abs_limit = limit_fraction * 10_000_000  # 1% of 1억 KRW fallback
            return abs(net_pnl) >= abs_limit

    @staticmethod
    def is_weekly_limit_reached(conn: sqlite3.Connection) -> bool:
        """Return True if this week's cumulative net_pnl hits the weekly loss limit."""
        from src.utils.config import load_config
        config = load_config()
        weekly_limit = float(
            os.getenv("WEEKLY_LOSS_LIMIT", str(config.daily_loss_limit * 3))
        )

        today = date.today()
        week_start = (today - timedelta(days=today.weekday())).isoformat()

        row = conn.execute(
            "SELECT COALESCE(SUM(CAST(net_pnl AS REAL)), 0) FROM daily_performance WHERE perf_date >= ?",
            (week_start,),
        ).fetchone()

        weekly_pnl = float(row[0]) if row else 0.0
        if weekly_pnl >= 0:
            return False

        from src.utils.startup_recovery import get_cached_balance
        balance = float(get_cached_balance().get("availableBalance", 0) or 0)

        if balance > 0:
            return abs(weekly_pnl) / balance >= weekly_limit
        else:
            abs_limit = weekly_limit * 10_000_000
            return abs(weekly_pnl) >= abs_limit

    @staticmethod
    def check_and_lock_profit(
        conn: sqlite3.Connection,
        current_balance: float,
        initial_balance: float,
    ) -> tuple[float, float]:
        """Evaluate profit-lock tiers and return (locked_fraction, available_krw).

        Tiers:
            >= 30% profit → lock 80%
            >= 20% profit → lock 70%
            >= 10% profit → lock 50%
        """
        if initial_balance <= 0:
            return 0.0, current_balance

        profit_pct = (current_balance - initial_balance) / initial_balance

        for floor_pct, lock_fraction in _PROFIT_LOCK_TIERS:
            if profit_pct >= floor_pct:
                available = current_balance * (1 - lock_fraction)
                logger.info(
                    "Profit lock active: profit=%.1f%% → lock=%.0f%% → available=%.0f KRW",
                    profit_pct * 100,
                    lock_fraction * 100,
                    available,
                )
                return lock_fraction, available

        return 0.0, current_balance
