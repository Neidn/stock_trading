"""Market-hours guards for KRX and US markets.

KRX trades 09:00–15:30 KST, weekdays only (Korean public holidays excluded).
US  trades 09:30–16:00 ET,  weekdays only (US federal holidays excluded).

Usage::

    from src.risk.market_hours import is_market_open, is_us_market_open

    if not is_market_open():        # KRX
        return
    if not is_us_market_open():     # NYSE / NASDAQ
        return
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
ET  = ZoneInfo("America/New_York")

_OPEN_H, _OPEN_M = 9, 0
_CLOSE_H, _CLOSE_M = 15, 30

# Korean public holidays 2025–2027 (add new years as needed)
_HOLIDAYS: frozenset[datetime.date] = frozenset({
    # 2025
    datetime.date(2025, 1, 1),   # 신정
    datetime.date(2025, 1, 28),  # 설날 연휴
    datetime.date(2025, 1, 29),  # 설날
    datetime.date(2025, 1, 30),  # 설날 연휴
    datetime.date(2025, 3, 1),   # 삼일절
    datetime.date(2025, 5, 5),   # 어린이날
    datetime.date(2025, 5, 6),   # 대체공휴일
    datetime.date(2025, 6, 6),   # 현충일
    datetime.date(2025, 8, 15),  # 광복절
    datetime.date(2025, 10, 3),  # 개천절
    datetime.date(2025, 10, 5),  # 추석 연휴
    datetime.date(2025, 10, 6),  # 추석
    datetime.date(2025, 10, 7),  # 추석 연휴
    datetime.date(2025, 10, 9),  # 한글날
    datetime.date(2025, 12, 25), # 크리스마스
    # 2026
    datetime.date(2026, 1, 1),   # 신정
    datetime.date(2026, 2, 16),  # 설날 연휴
    datetime.date(2026, 2, 17),  # 설날
    datetime.date(2026, 2, 18),  # 설날 연휴
    datetime.date(2026, 3, 1),   # 삼일절
    datetime.date(2026, 5, 5),   # 어린이날
    datetime.date(2026, 6, 6),   # 현충일
    datetime.date(2026, 8, 15),  # 광복절
    datetime.date(2026, 9, 24),  # 추석 연휴
    datetime.date(2026, 9, 25),  # 추석
    datetime.date(2026, 9, 26),  # 추석 연휴
    datetime.date(2026, 10, 3),  # 개천절
    datetime.date(2026, 10, 9),  # 한글날
    datetime.date(2026, 12, 25), # 크리스마스
    # 2027
    datetime.date(2027, 1, 1),   # 신정
    datetime.date(2027, 2, 6),   # 설날 연휴
    datetime.date(2027, 2, 7),   # 설날
    datetime.date(2027, 2, 8),   # 설날 연휴
    datetime.date(2027, 3, 1),   # 삼일절
    datetime.date(2027, 5, 5),   # 어린이날
    datetime.date(2027, 6, 6),   # 현충일
    datetime.date(2027, 8, 15),  # 광복절
    datetime.date(2027, 10, 3),  # 개천절
    datetime.date(2027, 10, 9),  # 한글날
    datetime.date(2027, 12, 25), # 크리스마스
})


def _now_kst() -> datetime.datetime:
    return datetime.datetime.now(tz=KST)


def is_trading_day(date: datetime.date | None = None) -> bool:
    """Return True if *date* is a KRX trading day (weekday, not holiday)."""
    d = date or _now_kst().date()
    return d.weekday() < 5 and d not in _HOLIDAYS


def is_market_open(*, buffer_open_sec: int = 0) -> bool:
    """Return True if KRX is currently open for trading.

    Args:
        buffer_open_sec: Skip the first N seconds after open (e.g. 300 to skip
                         09:00–09:05 opening auction volatility).
    """
    now = _now_kst()
    if not is_trading_day(now.date()):
        return False

    open_dt = now.replace(hour=_OPEN_H, minute=_OPEN_M, second=0, microsecond=0)
    close_dt = now.replace(hour=_CLOSE_H, minute=_CLOSE_M, second=0, microsecond=0)

    open_dt += datetime.timedelta(seconds=buffer_open_sec)
    return open_dt <= now < close_dt


def is_closing_soon(buffer_min: int = 10) -> bool:
    """Return True if market closes within *buffer_min* minutes OR is already past close."""
    now = _now_kst()
    if not is_trading_day(now.date()):
        return False
    close_dt = now.replace(hour=_CLOSE_H, minute=_CLOSE_M, second=0, microsecond=0)
    return (close_dt - now) <= datetime.timedelta(minutes=buffer_min)


def seconds_until_open() -> float:
    """Return seconds until next KRX market open.

    Returns 0 if market is currently open.
    """
    if is_market_open():
        return 0.0

    now = _now_kst()
    candidate = now.replace(hour=_OPEN_H, minute=_OPEN_M, second=0, microsecond=0)

    # If today's open already passed, move to next day
    if candidate <= now:
        candidate += datetime.timedelta(days=1)

    # Skip non-trading days
    while not is_trading_day(candidate.date()):
        candidate += datetime.timedelta(days=1)

    return (candidate - now).total_seconds()


def minutes_until_close() -> float:
    """Return minutes remaining until KRX close. Negative if closed."""
    now = _now_kst()
    if not is_trading_day(now.date()):
        return -1.0
    close_dt = now.replace(hour=_CLOSE_H, minute=_CLOSE_M, second=0, microsecond=0)
    return (close_dt - now).total_seconds() / 60


# ---------------------------------------------------------------------------
# US market (NYSE / NASDAQ)  — 09:30–16:00 America/New_York
# ---------------------------------------------------------------------------

_US_OPEN_H,  _US_OPEN_M  = 9,  30
_US_CLOSE_H, _US_CLOSE_M = 16, 0

# US federal holidays 2025–2027 that close NYSE/NASDAQ
_US_HOLIDAYS: frozenset[datetime.date] = frozenset({
    # 2025
    datetime.date(2025, 1, 1),   # New Year's Day
    datetime.date(2025, 1, 20),  # MLK Day
    datetime.date(2025, 2, 17),  # Presidents' Day
    datetime.date(2025, 4, 18),  # Good Friday
    datetime.date(2025, 5, 26),  # Memorial Day
    datetime.date(2025, 6, 19),  # Juneteenth
    datetime.date(2025, 7, 4),   # Independence Day
    datetime.date(2025, 9, 1),   # Labor Day
    datetime.date(2025, 11, 27), # Thanksgiving
    datetime.date(2025, 12, 25), # Christmas
    # 2026
    datetime.date(2026, 1, 1),   # New Year's Day
    datetime.date(2026, 1, 19),  # MLK Day
    datetime.date(2026, 2, 16),  # Presidents' Day
    datetime.date(2026, 4, 3),   # Good Friday
    datetime.date(2026, 5, 25),  # Memorial Day
    datetime.date(2026, 6, 19),  # Juneteenth
    datetime.date(2026, 7, 3),   # Independence Day (observed Fri)
    datetime.date(2026, 9, 7),   # Labor Day
    datetime.date(2026, 11, 26), # Thanksgiving
    datetime.date(2026, 12, 25), # Christmas
    # 2027
    datetime.date(2027, 1, 1),   # New Year's Day
    datetime.date(2027, 1, 18),  # MLK Day
    datetime.date(2027, 2, 15),  # Presidents' Day
    datetime.date(2027, 3, 26),  # Good Friday
    datetime.date(2027, 5, 31),  # Memorial Day
    datetime.date(2027, 6, 18),  # Juneteenth (observed Fri)
    datetime.date(2027, 7, 5),   # Independence Day (observed Mon)
    datetime.date(2027, 9, 6),   # Labor Day
    datetime.date(2027, 11, 25), # Thanksgiving
    datetime.date(2027, 12, 24), # Christmas (observed Fri)
})


def _now_et() -> datetime.datetime:
    return datetime.datetime.now(tz=ET)


def is_us_trading_day(date: datetime.date | None = None) -> bool:
    """Return True if *date* is a NYSE/NASDAQ trading day."""
    d = date or _now_et().date()
    return d.weekday() < 5 and d not in _US_HOLIDAYS


def is_us_market_open(*, buffer_open_sec: int = 0) -> bool:
    """Return True if US market (NYSE/NASDAQ) is currently open.

    Args:
        buffer_open_sec: Skip the first N seconds after 09:30 open.
    """
    now = _now_et()
    if not is_us_trading_day(now.date()):
        return False

    open_dt  = now.replace(hour=_US_OPEN_H,  minute=_US_OPEN_M,  second=0, microsecond=0)
    close_dt = now.replace(hour=_US_CLOSE_H, minute=_US_CLOSE_M, second=0, microsecond=0)
    open_dt += datetime.timedelta(seconds=buffer_open_sec)
    return open_dt <= now < close_dt


def is_us_closing_soon(buffer_min: int = 10) -> bool:
    """Return True if US market closes within *buffer_min* minutes."""
    now = _now_et()
    if not is_us_trading_day(now.date()):
        return False
    close_dt = now.replace(hour=_US_CLOSE_H, minute=_US_CLOSE_M, second=0, microsecond=0)
    return datetime.timedelta(0) <= (close_dt - now) <= datetime.timedelta(minutes=buffer_min)


def seconds_until_us_open() -> float:
    """Return seconds until next US market open. Returns 0 if open now."""
    if is_us_market_open():
        return 0.0

    now = _now_et()
    candidate = now.replace(hour=_US_OPEN_H, minute=_US_OPEN_M, second=0, microsecond=0)
    if candidate <= now:
        candidate += datetime.timedelta(days=1)
    while not is_us_trading_day(candidate.date()):
        candidate += datetime.timedelta(days=1)
    return (candidate - now).total_seconds()


def minutes_until_us_close() -> float:
    """Return minutes remaining until US market close. Negative if closed."""
    now = _now_et()
    if not is_us_trading_day(now.date()):
        return -1.0
    close_dt = now.replace(hour=_US_CLOSE_H, minute=_US_CLOSE_M, second=0, microsecond=0)
    return (close_dt - now).total_seconds() / 60
