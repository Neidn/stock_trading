"""Common statistical functions for strategy analysis.

All functions:
- Accept np.ndarray inputs.
- Return scalar or np.ndarray with NaN on insufficient data.
- Never raise on bad input — return NaN / 0.5 sentinels instead.
"""

from __future__ import annotations

import numpy as np


def calc_half_life(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion.

    Estimates how many periods it takes for the spread to revert halfway
    to its mean. Used to confirm a series is mean-reverting before entry.

    Args:
        spread: 1-D price spread or residual array.

    Returns:
        Half-life in periods. Returns ``np.nan`` if series is not
        mean-reverting (positive lag-1 coefficient) or too short.

    Example:
        hl = calc_half_life(spread)
        if 5 < hl < 60:   # revert within 5–60 bars
            enter_trade()
    """
    if len(spread) < 3:
        return np.nan

    spread_lag = spread[:-1]
    spread_ret = spread[1:] - spread[:-1]

    # OLS: delta_y = beta * y_lag + alpha
    # beta = cov(lag, ret) / var(lag)
    lag_mean = np.mean(spread_lag)
    ret_mean = np.mean(spread_ret)
    cov = np.mean((spread_lag - lag_mean) * (spread_ret - ret_mean))
    var = np.var(spread_lag)

    if var == 0:
        return np.nan

    beta = cov / var

    if beta >= 0:
        # Not mean-reverting
        return np.nan

    return -np.log(2) / beta


def calc_hurst_exponent(prices: np.ndarray, max_lag: int = 20) -> float:
    """Hurst exponent via rescaled range (R/S) analysis.

    Classifies the price series behaviour:
        H < 0.5  → mean-reverting (anti-persistent)
        H ≈ 0.5  → random walk
        H > 0.5  → trending (persistent)

    Args:
        prices: 1-D price array (use close prices).
        max_lag: Maximum lag for R/S calculation (default 20).

    Returns:
        Hurst exponent in [0, 1]. Returns ``0.5`` on insufficient data.

    Example:
        h = calc_hurst_exponent(df['close'].values)
        if h < 0.45:
            # strong mean-reversion signal
    """
    min_len = max(20, max_lag + 1)
    if len(prices) < min_len:
        return 0.5

    lags = range(2, max_lag)
    rs_values = []
    for lag in lags:
        ts = prices[:lag]
        mean = np.mean(ts)
        deviation = np.cumsum(ts - mean)
        r = np.max(deviation) - np.min(deviation)
        s = np.std(ts, ddof=1)
        if s == 0:
            continue
        rs_values.append(r / s)

    if len(rs_values) < 2:
        return 0.5

    log_lags = np.log(list(range(2, 2 + len(rs_values))))
    log_rs = np.log(rs_values)

    # Linear regression slope = Hurst exponent
    poly = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(poly[0], 0.0, 1.0))


def rolling_correlation(
    a: np.ndarray,
    b: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Rolling Pearson correlation between two price series.

    Args:
        a: First 1-D array.
        b: Second 1-D array (same length as ``a``).
        window: Rolling window (default 20).

    Returns:
        Correlation array in [-1, 1]; first ``window - 1`` values are NaN.

    Example:
        corr = rolling_correlation(btc_close, eth_close)
        if corr[-1] > 0.8:
            # highly correlated pair
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    if len(a) < window:
        return np.full(len(a), np.nan)

    result = np.full(len(a), np.nan)
    for i in range(window - 1, len(a)):
        slice_a = a[i - window + 1 : i + 1]
        slice_b = b[i - window + 1 : i + 1]
        std_a = np.std(slice_a)
        std_b = np.std(slice_b)
        if std_a == 0 or std_b == 0:
            result[i] = np.nan
        else:
            result[i] = np.corrcoef(slice_a, slice_b)[0, 1]
    return result


def calc_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio from a return series.

    Assumes daily returns; annualises by sqrt(365).

    Args:
        returns: 1-D array of period returns (e.g. 0.01 = 1%).
        risk_free: Risk-free rate per period (default 0.0).

    Returns:
        Sharpe ratio. Returns ``np.nan`` if std is zero or series too short.

    Example:
        sr = calc_sharpe(daily_pnl_array)
        if sr > 1.5:
            # strategy performing well
    """
    if len(returns) < 2:
        return np.nan

    excess = returns - risk_free
    std = np.std(excess, ddof=1)
    if std == 0:
        return np.nan

    return float(np.mean(excess) / std * np.sqrt(365))
