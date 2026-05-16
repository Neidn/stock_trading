from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def average_true_range(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift()).abs()
    low_close = (frame["low"] - frame["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def rolling_high(series: pd.Series, lookback: int) -> pd.Series:
    return series.rolling(window=lookback, min_periods=lookback).max()


def relative_volume(volume: pd.Series, lookback: int = 20) -> pd.Series:
    baseline = volume.shift(1).rolling(window=lookback, min_periods=lookback).mean()
    return volume / baseline
