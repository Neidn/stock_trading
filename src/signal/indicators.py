"""Vectorized technical indicators via ta-lib.

All functions:
- Accept np.ndarray inputs (use df['close'].values to convert from pandas).
- Return np.ndarray (or tuple thereof) with the same length as the input.
- Never raise exceptions — return np.nan arrays on insufficient data.
- Never use Python for-loops; rely on ta-lib's C-compiled vector operations.
"""

from __future__ import annotations

import numpy as np
import talib


def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index.

    Args:
        close: 1-D closing price array.
        period: Lookback period (default 14).

    Returns:
        RSI array in range [0, 100]; first ``period`` values are NaN.

    Example:
        rsi = calc_rsi(df['close'].values)
        current_rsi = rsi[-1]
    """
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    return talib.RSI(close, timeperiod=period)


def calc_macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence/Divergence.

    Args:
        close: 1-D closing price array.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram); early values are NaN.

    Example:
        macd, sig, hist = calc_macd(df['close'].values)
        # Bullish crossover: hist[-1] > 0 and hist[-2] <= 0
    """
    min_len = slow + signal
    if len(close) < min_len:
        nan = np.full(len(close), np.nan)
        return nan, nan, nan
    macd_line, signal_line, histogram = talib.MACD(
        close, fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    return macd_line, signal_line, histogram


def calc_bollinger(
    close: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands.

    Args:
        close: 1-D closing price array.
        period: SMA period for the middle band (default 20).
        std_dev: Number of standard deviations for upper/lower bands (default 2.0).

    Returns:
        Tuple of (upper, middle, lower); first ``period - 1`` values are NaN.

    Example:
        upper, middle, lower = calc_bollinger(df['close'].values)
        bb_width = (upper[-1] - lower[-1]) / middle[-1]
    """
    if len(close) < period:
        nan = np.full(len(close), np.nan)
        return nan, nan, nan
    upper, middle, lower = talib.BBANDS(
        close, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0
    )
    return upper, middle, lower


def calc_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range.

    Args:
        high: 1-D high price array.
        low: 1-D low price array.
        close: 1-D closing price array.
        period: Lookback period (default 14).

    Returns:
        ATR array; first ``period`` values are NaN.

    Example:
        atr = calc_atr(df['high'].values, df['low'].values, df['close'].values)
        sl = entry_price - atr[-1] * 2.0
    """
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    return talib.ATR(high, low, close, timeperiod=period)


def calc_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average.

    Args:
        close: 1-D closing price array.
        period: EMA period.

    Returns:
        EMA array; first ``period - 1`` values are NaN.

    Example:
        ema50 = calc_ema(df['close'].values, 50)
        above_ema = df['close'].values[-1] > ema50[-1]
    """
    if len(close) < period:
        return np.full(len(close), np.nan)
    return talib.EMA(close, timeperiod=period)


def calc_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average.

    Args:
        close: 1-D closing price array.
        period: SMA period.

    Returns:
        SMA array; first ``period - 1`` values are NaN.

    Example:
        sma20 = calc_sma(df['close'].values, 20)
    """
    if len(close) < period:
        return np.full(len(close), np.nan)
    return talib.SMA(close, timeperiod=period)


def calc_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Directional Index — measures trend strength (not direction).

    Args:
        high: 1-D high price array.
        low: 1-D low price array.
        close: 1-D closing price array.
        period: Lookback period (default 14).

    Returns:
        ADX array in range [0, 100]; values > 25 indicate a trending market.
        First ``2 * period`` values are NaN.

    Example:
        adx = calc_adx(df['high'].values, df['low'].values, df['close'].values)
        is_trending = adx[-1] > 25
    """
    if len(close) < 2 * period:
        return np.full(len(close), np.nan)
    return talib.ADX(high, low, close, timeperiod=period)


def calc_volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Volume ratio: current volume divided by its rolling SMA.

    A ratio > 1.3 indicates above-average activity; used as a volume filter
    before entering positions.

    Args:
        volume: 1-D volume array.
        period: SMA period for the baseline (default 20).

    Returns:
        Array of ratios; first ``period - 1`` values are NaN.

    Example:
        vol_ratio = calc_volume_ratio(df['volume'].values)
        high_volume = vol_ratio[-1] > 1.3
    """
    if len(volume) < period:
        return np.full(len(volume), np.nan)
    vol_sma = talib.SMA(volume, timeperiod=period)
    # Avoid division by zero — replace 0 SMA with NaN
    vol_sma = np.where(vol_sma == 0, np.nan, vol_sma)
    return volume / vol_sma


def calc_zscore(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling Z-score: (close - rolling_mean) / rolling_std.

    Values beyond ±2 suggest statistically extreme deviations from the mean,
    used by mean-reversion strategies to identify entry points.

    Args:
        close: 1-D closing price array.
        window: Rolling window size (default 20).

    Returns:
        Z-score array; first ``window - 1`` values are NaN.

    Example:
        zscore = calc_zscore(df['close'].values)
        oversold = zscore[-1] < -2.0   # potential long entry
        overbought = zscore[-1] > 2.0  # potential short entry
    """
    if len(close) < window:
        return np.full(len(close), np.nan)

    rolling_mean = talib.SMA(close, timeperiod=window)
    # ta-lib has no rolling std; use EMA-based variance via STDDEV
    rolling_std = talib.STDDEV(close, timeperiod=window, nbdev=1)

    std_safe = np.where(rolling_std == 0, np.nan, rolling_std)
    return (close - rolling_mean) / std_safe
