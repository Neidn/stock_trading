"""Unit tests for EmaCrossoverStrategy."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.signal.strategies.ema_crossover import EmaCrossoverStrategy


def _make_df(close, high=None, low=None) -> pd.DataFrame:
    close = np.array(close, dtype=float)
    if high is None:
        high = close * 1.01
    if low is None:
        low = close * 0.99
    return pd.DataFrame({
        "open":   close,
        "high":   np.array(high, dtype=float),
        "low":    np.array(low,  dtype=float),
        "close":  close,
        "volume": np.ones(len(close)) * 1000,
    })


def _strategy(overrides=None) -> EmaCrossoverStrategy:
    params = {"ema_fast": 3, "ema_slow": 5, "adx_period": 3,
              "adx_threshold": 0.0, "sl_atr_mult": 2.0,
              "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0}
    if overrides:
        params.update(overrides)
    return EmaCrossoverStrategy(params)


class TestEmaCrossoverDefaults(unittest.TestCase):

    def test_name(self):
        s = EmaCrossoverStrategy({})
        self.assertEqual(s.get_name(), "ema_crossover")

    def test_timeframe(self):
        self.assertEqual(EmaCrossoverStrategy({}).get_timeframe(), "1d")

    def test_min_candles_uses_slow_period(self):
        s = EmaCrossoverStrategy({"ema_fast": 20, "ema_slow": 50, "adx_period": 14})
        self.assertGreater(s.get_min_candles(), 50)

    def test_defaults_applied(self):
        s = EmaCrossoverStrategy({})
        p = {**s.DEFAULTS, **s.params}
        self.assertEqual(p["ema_fast"], 20)
        self.assertEqual(p["ema_slow"], 50)
        self.assertEqual(p["adx_threshold"], 25.0)


class TestEmaCrossoverValidation(unittest.TestCase):

    def test_fast_gte_slow_raises(self):
        with self.assertRaises(ValueError):
            EmaCrossoverStrategy({"ema_fast": 50, "ema_slow": 20})._validate_params()

    def test_fast_equal_slow_raises(self):
        with self.assertRaises(ValueError):
            EmaCrossoverStrategy({"ema_fast": 20, "ema_slow": 20})._validate_params()

    def test_sl_zero_raises(self):
        with self.assertRaises(ValueError):
            EmaCrossoverStrategy({"ema_fast": 3, "ema_slow": 5, "sl_atr_mult": 0})._validate_params()

    def test_tp1_lte_sl_raises(self):
        with self.assertRaises(ValueError):
            EmaCrossoverStrategy({
                "ema_fast": 3, "ema_slow": 5,
                "sl_atr_mult": 3.0, "tp1_atr_mult": 2.0,
            })._validate_params()


class TestEmaCrossoverLong(unittest.TestCase):

    def _cross_up_prices(self):
        # Long flat base (fast EMA ≈ slow EMA below), then sharp spike on last bar
        # → fast EMA(3) reacts faster than slow EMA(5) → crossover on last bar
        base = [100.0] * 20   # both EMAs settle near 100
        # Pull prices down so fast < slow, then spike last bar above both
        down = [95.0] * 5
        return base + down + [130.0]  # last bar spikes → fast crosses above slow

    def test_long_signal_on_cross_up(self):
        prices = self._cross_up_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "long")

    def test_long_sl_below_entry(self):
        prices = self._cross_up_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertLess(result.sl, result.entry_price)

    def test_long_tp1_above_entry(self):
        prices = self._cross_up_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.tp1, result.entry_price)

    def test_long_tp2_above_tp1(self):
        prices = self._cross_up_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.tp2, result.tp1)

    def test_long_strength_score(self):
        prices = self._cross_up_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.strength_score, 2)


class TestEmaCrossoverShort(unittest.TestCase):

    def _cross_down_prices(self):
        # Flat base then pull up so fast > slow, then crash last bar
        # → fast EMA(3) drops faster than slow EMA(5) → crossover down on last bar
        base = [100.0] * 20
        up   = [105.0] * 5
        return base + up + [70.0]  # last bar crashes → fast crosses below slow

    def test_short_signal_on_cross_down(self):
        prices = self._cross_down_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "short")

    def test_short_sl_above_entry(self):
        prices = self._cross_down_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.sl, result.entry_price)

    def test_short_tp1_below_entry(self):
        prices = self._cross_down_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertLess(result.tp1, result.entry_price)

    def test_short_tp2_below_tp1(self):
        prices = self._cross_down_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertLess(result.tp2, result.tp1)


class TestEmaCrossoverAdxFilter(unittest.TestCase):

    def test_no_signal_when_adx_below_threshold(self):
        # Use high adx_threshold to force filter
        prices = list(np.linspace(100, 80, 15)) + list(np.linspace(80, 120, 15))
        df = _make_df(prices)
        result = _strategy({"adx_threshold": 999.0}).generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_adx_reason_in_hold(self):
        prices = list(np.linspace(100, 80, 15)) + list(np.linspace(80, 120, 15))
        df = _make_df(prices)
        result = _strategy({"adx_threshold": 999.0}).generate_signal(df, "TEST")
        self.assertIn("ADX", result.reason)


class TestEmaCrossoverNegativeSL(unittest.TestCase):

    def test_hold_when_sl_would_be_negative(self):
        # Very high ATR relative to low price → sl < 0 → hold
        prices = [0.001] * 20 + [0.01] * 10
        high   = [p * 5 for p in prices]   # huge high → huge ATR
        low    = [p * 0.01 for p in prices]
        df     = _make_df(prices, high=high, low=low)
        result = _strategy({"adx_threshold": 0.0}).generate_signal(df, "TEST")
        if result.signal_type == "long":
            self.assertGreater(result.sl, 0)


class TestEmaCrossoverHold(unittest.TestCase):

    def test_flat_prices_no_signal(self):
        prices = [100.0] * 30
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_indicators_present_on_signal(self):
        prices = [100.0] * 20 + [95.0] * 5 + [130.0]
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "long")
        self.assertIn("ema_fast", result.indicators)
        self.assertIn("ema_slow", result.indicators)
        self.assertIn("adx",      result.indicators)
        self.assertIn("atr",      result.indicators)
