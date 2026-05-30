"""Signal blocker — gate that prevents order submission under adverse conditions.

Checks are evaluated in order; first match short-circuits and returns the reason.
All blocking conditions are logged at WARNING level.
"""

from __future__ import annotations

import sqlite3
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.db.models import has_open_position
from src.monitoring.logger import get_logger
from src.risk.drawdown_guard import MarketShockDetector as _RealMarketShockDetector
from src.safety.safe_mode import SafeMode as _SafeModeClass
from src.utils.config import load_config

logger = get_logger("signal_blocker")

# Module-level SafeMode singleton
SafeMode: _SafeModeClass = _SafeModeClass()

# MarketShockDetector re-exported for backward-compat
MarketShockDetector = _RealMarketShockDetector


class EconomicCalendar:
    """High-impact economic event guard using ForexFactory calendar.

    Fetches weekly calendar JSON and caches for 1 hour.
    Fails open (returns False) on any network/parse error.
    KRX note: USD macro events (Fed, CPI) do affect KRX — kept intentionally.
    """

    _FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    _CACHE_TTL = 3600
    _cache: list = []
    _cache_time: float = 0.0

    @classmethod
    def _refresh_cache(cls) -> None:
        try:
            req = urllib.request.Request(cls._FF_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                import json
                cls._cache = json.loads(resp.read())
            cls._cache_time = time.time()
            logger.debug("EconomicCalendar: fetched %d events", len(cls._cache))
        except Exception as exc:  # noqa: BLE001
            logger.warning("EconomicCalendar fetch failed (fail-open): %s", exc)
            cls._cache_time = time.time()

    @classmethod
    def is_high_impact_event_soon(cls, minutes: int = 60) -> bool:
        """Return True if a high-impact USD event is within *minutes*."""
        if time.time() - cls._cache_time > cls._CACHE_TTL:
            cls._refresh_cache()

        if not cls._cache:
            return False

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=minutes)

        for event in cls._cache:
            if event.get("impact") != "High":
                continue
            if event.get("country") != "USD":
                continue
            date_str = event.get("date", "")
            if not date_str:
                continue
            try:
                event_dt = datetime.fromisoformat(date_str)
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
                if now <= event_dt <= cutoff:
                    logger.warning(
                        "EconomicCalendar: high-impact event soon — %s at %s",
                        event.get("title", "?"),
                        event_dt.isoformat(),
                    )
                    return True
            except (ValueError, TypeError):
                continue

        return False


# ---------------------------------------------------------------------------
# SignalBlocker
# ---------------------------------------------------------------------------

def _dynamic_max_positions(kospi_adx: float | None, base: int) -> int:
    """Reduce position slots in ranging/weak-trend markets.

    ADX >= 25 → base; 20-25 → base-1; <20 → base-2. Floor=1.
    """
    if kospi_adx is None or kospi_adx >= 25.0:
        return base
    if kospi_adx >= 20.0:
        return max(1, base - 1)
    return max(1, base - 2)


class SignalBlocker:
    """Evaluates whether a new signal should be blocked before order placement.

    Args:
        conn: Open SQLite connection.
        gap_detector: Optional data-gap detector. If None, gap check is skipped.
        market_adx: Current KOSPI ADX. Reduces position limit in ranging markets.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        gap_detector: Any | None = None,
        market_adx: float | None = None,
    ) -> None:
        self._conn = conn
        self._gap_detector = gap_detector
        self._shock_detector = _RealMarketShockDetector(conn=conn)
        self._market_adx = market_adx

    def should_block(self, symbol: str) -> tuple[bool, str]:
        """Return (True, reason) if signal must be blocked, else (False, '')."""
        checks = [
            self._check_data_gap,
            self._check_safe_mode,
            self._check_daily_loss_limit,
            self._check_market_shock,
            self._check_economic_event,
            self._check_max_positions,
            self._check_open_position,
        ]
        for check in checks:
            blocked, reason = check(symbol)
            if blocked:
                logger.warning("Signal blocked [%s]: %s", symbol, reason)
                return True, reason
        return False, ""

    def _check_data_gap(self, symbol: str) -> tuple[bool, str]:
        if self._gap_detector is None:
            return False, ""
        if self._gap_detector.has_gap(symbol):
            return True, "data_gap: missing recent candles"
        return False, ""

    def _check_safe_mode(self, _symbol: str) -> tuple[bool, str]:
        try:
            row = self._conn.execute(
                "SELECT action, reason FROM safe_mode_events ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if row and row[0] == "activated":
                return True, f"safe_mode: {row[1] or 'active'}"
        except Exception:  # noqa: BLE001
            if SafeMode.is_active():
                return True, f"safe_mode: {SafeMode.reason or 'active'}"
        return False, ""

    def _check_daily_loss_limit(self, _symbol: str) -> tuple[bool, str]:
        """Block if today's realized losses exceed the daily limit."""
        rows = self._conn.execute(
            "SELECT entry_price, exit_price, quantity, side"
            " FROM positions"
            " WHERE status='closed'"
            " AND exit_price IS NOT NULL"
            " AND date(closed_at) = date('now')",
        ).fetchall()

        gross_loss = 0.0
        for row in rows:
            try:
                entry  = float(row["entry_price"] or 0)
                exit_p = float(row["exit_price"] or 0)
                qty    = float(row["quantity"] or 0)
                side   = row["side"] if hasattr(row, "keys") else row[3]
            except (TypeError, ValueError):
                continue
            pnl = (exit_p - entry) * qty if side == "long" else (entry - exit_p) * qty
            if pnl < 0:
                gross_loss += abs(pnl)

        if gross_loss == 0.0:
            return False, ""

        config = load_config()
        limit_fraction = config.daily_loss_limit

        from src.utils.startup_recovery import get_cached_balance
        cached = get_cached_balance()
        balance = float(cached.get("availableBalance", 0) or 0)

        if balance > 0:
            if gross_loss / balance >= limit_fraction:
                return (
                    True,
                    f"daily_loss_limit: loss {gross_loss:,.0f} KRW"
                    f" >= {limit_fraction*100:.1f}% of balance {balance:,.0f}",
                )
        else:
            abs_limit = limit_fraction * 10_000_000
            if gross_loss >= abs_limit:
                return (
                    True,
                    f"daily_loss_limit: loss {gross_loss:,.0f} KRW"
                    f" >= nominal limit {abs_limit:,.0f} (no balance cache)",
                )

        return False, ""

    def _check_market_shock(self, symbol: str) -> tuple[bool, str]:
        level = self._shock_detector.current_level(symbol)
        if level in ("ELEVATED", "DANGER"):
            return True, f"market_shock: level={level}"
        return False, ""

    def _check_economic_event(self, _symbol: str) -> tuple[bool, str]:
        if EconomicCalendar.is_high_impact_event_soon(minutes=60):
            return True, "economic_event: high-impact event within 60 min"
        return False, ""

    def _check_max_positions(self, _symbol: str) -> tuple[bool, str]:
        try:
            config = load_config()
        except Exception as exc:  # noqa: BLE001
            logger.error("Risk config invalid; blocking signal: %s", exc)
            return True, f"config_error: {exc}"

        row = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE status='open'"
        ).fetchone()
        open_positions = int(row[0] if row is not None else 0)
        limit = _dynamic_max_positions(self._market_adx, config.max_positions)
        if open_positions >= limit:
            adx_str = f" (KOSPI ADX={self._market_adx:.1f})" if self._market_adx is not None else ""
            return (
                True,
                f"max_positions: {open_positions} open >= limit {limit}{adx_str}",
            )
        return False, ""

    def _check_open_position(self, symbol: str) -> tuple[bool, str]:
        if has_open_position(self._conn, symbol):
            return True, f"open_position: existing open position for {symbol}"
        return False, ""
