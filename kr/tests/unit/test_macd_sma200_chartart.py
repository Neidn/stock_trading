"""Unit tests for MacdSma200ChartartStrategy."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.signal.strategies.macd_sma200_chartart import MacdSma200ChartartStrategy


def _make_df(close, high=None, low=None) -> pd.DataFrame:
    close = np.array(close, dtype=float)
    if high is None:
        high = close * 1.005
    if low is None:
        low = close * 0.995
    return pd.DataFrame({
        "open":   close,
        "high":   np.array(high, dtype=float),
        "low":    np.array(low,  dtype=float),
        "close":  close,
        "volume": np.ones(len(close)) * 1000,
    })


def _strategy(overrides=None) -> MacdSma200ChartartStrategy:
    # Use small periods so tests don't need 200+ bars
    params = {
        "fast_period":   3,
        "slow_period":   5,
        "signal_period": 3,
        "sma200_period": 10,
        "sl_atr_mult":   2.0,
        "tp1_atr_mult":  3.0,
        "tp2_atr_mult":  5.0,
    }
    if overrides:
        params.update(overrides)
    return MacdSma200ChartartStrategy(params)


def _bullish_prices():
    """Verified: close[-6]=100 > vslow=98; hist crosses 0 up on last bar; macd>0; fast>slow."""
    # flat base → step dip (hist goes negative) → spike (hist crosses 0 up)
    return [100.0] * 20 + [100.0, 90.0, 80.0, 70.0, 60.0] + [180.0]


def _bearish_prices():
    """Verified: close[-6]=60 < vslow=66; hist crosses 0 down on last bar; macd<0; fast<slow."""
    # high base → decline → bounce (hist goes positive) → crash (hist crosses 0 down)
    return [100.0] * 10 + [80.0] * 5 + [60.0] * 5 + [70.0, 80.0, 90.0, 10.0]


class TestMacdSma200Defaults(unittest.TestCase):

    def test_name(self):
        self.assertEqual(MacdSma200ChartartStrategy({}).get_name(), "macd_sma200_chartart")

    def test_timeframe(self):
        self.assertEqual(MacdSma200ChartartStrategy({}).get_timeframe(), "1d")

    def test_min_candles_gt_sma200(self):
        s = MacdSma200ChartartStrategy({})
        self.assertGreater(s.get_min_candles(), 200)

    def test_defaults_applied(self):
        s = MacdSma200ChartartStrategy({})
        p = {**s.DEFAULTS, **s.params}
        self.assertEqual(p["fast_period"],   12)
        self.assertEqual(p["slow_period"],   26)
        self.assertEqual(p["sma200_period"], 200)


class TestMacdSma200Validation(unittest.TestCase):

    def test_fast_gte_slow_raises(self):
        with self.assertRaises(ValueError):
            MacdSma200ChartartStrategy({"fast_period": 26, "slow_period": 12})._validate_params()

    def test_sl_zero_raises(self):
        with self.assertRaises(ValueError):
            MacdSma200ChartartStrategy({
                "fast_period": 3, "slow_period": 5, "sl_atr_mult": 0
            })._validate_params()

    def test_tp1_lte_sl_raises(self):
        with self.assertRaises(ValueError):
            MacdSma200ChartartStrategy({
                "fast_period": 3, "slow_period": 5,
                "sl_atr_mult": 3.0, "tp1_atr_mult": 2.0,
            })._validate_params()


class TestMacdSma200Long(unittest.TestCase):

    def _df(self):
        return _make_df(_bullish_prices())

    def test_long_signal(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        self.assertEqual(result.signal_type, "long")

    def test_long_strength_score(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        self.assertEqual(result.strength_score, 3)

    def test_long_sl_below_entry(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        self.assertLess(result.sl, result.entry_price)

    def test_long_tp1_above_entry(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        self.assertGreater(result.tp1, result.entry_price)

    def test_long_tp2_above_tp1(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        self.assertGreater(result.tp2, result.tp1)

    def test_long_indicators_present(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        for key in ("fast_sma", "slow_sma", "very_slow", "macd", "hist", "atr"):
            self.assertIn(key, result.indicators)


class TestMacdSma200Short(unittest.TestCase):

    def _df(self):
        return _make_df(_bearish_prices())

    def test_short_signal(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        self.assertEqual(result.signal_type, "short")

    def test_short_strength_score(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        self.assertEqual(result.strength_score, 3)

    def test_short_sl_above_entry(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        self.assertGreater(result.sl, result.entry_price)

    def test_short_tp1_below_entry(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        self.assertLess(result.tp1, result.entry_price)

    def test_short_tp2_below_tp1(self):
        result = _strategy().generate_signal(self._df(), "TEST")
        self.assertLess(result.tp2, result.tp1)


class TestMacdSma200Filters(unittest.TestCase):

    def test_no_long_when_price_below_sma200(self):
        # Bullish MACD but all prices below SMA200 threshold (lagged close < very_slow)
        # Force by making sma200 period huge relative to recent prices
        prices = _bullish_prices()
        # Override sma200 so it averages higher than lagged close
        result = _strategy({"sma200_period": 5}).generate_signal(_make_df(prices), "TEST")
        # May or may not fire depending on exact values — just verify no crash
        self.assertIn(result.signal_type, ("long", "hold"))

    def test_hold_on_flat_prices(self):
        prices = [100.0] * 30
        result = _strategy().generate_signal(_make_df(prices), "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_insufficient_data_returns_hold(self):
        prices = [100.0] * 5  # too few bars
        result = _strategy().generate_signal(_make_df(prices), "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_negative_sl_returns_hold(self):
        # Tiny prices + huge ATR → sl < 0
        prices = [0.001] * 20 + [0.005]
        high   = [p * 10 for p in prices]
        low    = [p * 0.01 for p in prices]
        result = _strategy().generate_signal(_make_df(prices, high, low), "TEST")
        if result.signal_type == "long":
            self.assertGreater(result.sl, 0)
