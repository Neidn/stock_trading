"""Time utility helpers.

All datetimes are UTC-aware unless the function name says otherwise.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return current UTC time as a timezone-aware datetime.

    Example:
        now = utc_now()
        age = (utc_now() - candle_open_time).total_seconds()
    """
    return datetime.now(timezone.utc)


def ts_to_dt(ms_timestamp: int | float) -> datetime:
    """Convert a Binance millisecond timestamp to UTC datetime.

    Args:
        ms_timestamp: Unix timestamp in milliseconds (as returned by ccxt/Binance).

    Returns:
        Timezone-aware UTC datetime.

    Example:
        dt = ts_to_dt(candle['timestamp'])
    """
    return datetime.fromtimestamp(ms_timestamp / 1000.0, tz=timezone.utc)


def dt_to_ts(dt: datetime) -> int:
    """Convert a datetime to a Binance-style millisecond Unix timestamp.

    Args:
        dt: Datetime (naive datetimes assumed UTC).

    Returns:
        Integer millisecond timestamp.

    Example:
        since = dt_to_ts(utc_now() - timedelta(hours=1))
        candles = exchange.fetch_ohlcv(symbol, since=since)
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def round_down_to_interval(dt: datetime, minutes: int) -> datetime:
    """Floor a datetime to the nearest candle open time.

    Args:
        dt: Input datetime.
        minutes: Candle interval in minutes (e.g. 1, 5, 15, 60).

    Returns:
        Timezone-aware UTC datetime truncated to the interval boundary.

    Example:
        # 14:37 UTC → 14:35 UTC for a 5-min candle
        open_time = round_down_to_interval(utc_now(), 5)
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    total_minutes = dt.hour * 60 + dt.minute
    floored_minutes = (total_minutes // minutes) * minutes
    return dt.replace(
        hour=floored_minutes // 60,
        minute=floored_minutes % 60,
        second=0,
        microsecond=0,
    )


def candle_age_seconds(candle_open_ms: int | float) -> float:
    """Seconds elapsed since a candle opened.

    Useful for DataGapDetector and freshness checks.

    Args:
        candle_open_ms: Candle open timestamp in milliseconds.

    Returns:
        Age in seconds (float).

    Example:
        if candle_age_seconds(latest_candle['timestamp']) > 120:
            flag_data_gap(symbol)
    """
    return (utc_now() - ts_to_dt(candle_open_ms)).total_seconds()
