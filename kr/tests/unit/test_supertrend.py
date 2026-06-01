"""Unit tests for SupertrendStrategy."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.signal.strategies.supertrend import SupertrendStrategy, _calc_supertrend


def _make_df(close, high=None, low=None) -> pd.DataFrame:
    close = np.array(close, dtype=float)
    if high is None:
        high = close * 1.02
    if low is None:
        low = close * 0.98
    return pd.DataFrame({
        "open":   close,
        "high":   np.array(high, dtype=float),
        "low":    np.array(low,  dtype=float),
        "close":  close,
        "volume": np.ones(len(close)) * 1000,
    })


def _strategy(overrides=None) -> SupertrendStrategy:
    # Use short periods so tests need fewer candles
    params = {"atr_period": 3, "multiplier": 1.5,
              "sl_atr_mult": 2.0, "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0}
    if overrides:
        params.update(overrides)
    return SupertrendStrategy(params)


# ---------------------------------------------------------------------------
# Defaults / metadata
# ---------------------------------------------------------------------------

class TestSupertrendDefaults(unittest.TestCase):

    def test_name(self):
        self.assertEqual(SupertrendStrategy({}).get_name(), "supertrend")

    def test_timeframe(self):
        self.assertEqual(SupertrendStrategy({}).get_timeframe(), "1d")

    def test_min_candles(self):
        s = SupertrendStrategy({"atr_period": 10})
        self.assertGreater(s.get_min_candles(), 10)

    def test_defaults_applied(self):
        s = SupertrendStrategy({})
        p = {**s.DEFAULTS, **s.params}
        self.assertEqual(p["atr_period"], 10)
        self.assertAlmostEqual(p["multiplier"], 3.0)

    def test_validate_multiplier_zero_raises(self):
        with self.assertRaises(ValueError):
            SupertrendStrategy({"multiplier": 0}).generate_signal(
                _make_df([100] * 50), "X"
            )

    def test_validate_tp1_less_than_sl_raises(self):
        with self.assertRaises(ValueError):
            SupertrendStrategy({"sl_atr_mult": 5.0, "tp1_atr_mult": 2.0}).generate_signal(
                _make_df([100] * 50), "X"
            )


# ---------------------------------------------------------------------------
# Internal SuperTrend calculation
# ---------------------------------------------------------------------------

class TestCalcSupertrend(unittest.TestCase):

    def _arrays(self, n=60, base=100.0):
        close = np.full(n, base)
        high  = close * 1.02
        low   = close * 0.98
        return high, low, close

    def test_output_shapes(self):
        h, l, c = self._arrays()
        up, dn, trend = _calc_supertrend(h, l, c, period=3, multiplier=1.5)
        self.assertEqual(len(up), len(c))
        self.assertEqual(len(dn), len(c))
        self.assertEqual(len(trend), len(c))

    def test_trend_values_only_1_or_minus1(self):
        h, l, c = self._arrays()
        _, _, trend = _calc_supertrend(h, l, c, period=3, multiplier=1.5)
        unique = set(trend.tolist())
        self.assertTrue(unique.issubset({1.0, -1.0}))

    def test_up_band_below_close_in_uptrend(self):
        """In bullish trend (after warmup), up band (support) should be below close."""
        close = np.array([100.0] * 10 + list(np.linspace(100, 130, 40)))
        high  = close * 1.01
        low   = close * 0.99
        period = 3
        up, _, trend = _calc_supertrend(high, low, close, period=period, multiplier=1.5)
        # Skip warmup (first 2*period bars) where band hasn't settled
        warmup = period * 2
        bullish_idx = np.where(trend[warmup:] == 1)[0] + warmup
        if len(bullish_idx):
            self.assertTrue(np.all(up[bullish_idx] < close[bullish_idx] * 1.01))

    def test_dn_band_above_close_in_downtrend(self):
        """In bearish trend, dn band (resistance) should be above most closes."""
        close = np.array([130.0] * 10 + list(np.linspace(130, 100, 40)))
        high  = close * 1.01
        low   = close * 0.99
        _, dn, trend = _calc_supertrend(high, low, close, period=3, multiplier=1.5)
        bearish_idx = np.where(trend == -1)[0]
        if len(bearish_idx):
            self.assertTrue(np.all(dn[bearish_idx] > close[bearish_idx] * 0.99))


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

class TestSupertrendSignals(unittest.TestCase):

    def _bullish_flip_prices(self):
        """Price series that forces a bearish→bullish flip on last bar."""
        # Start ranging, drop to force bearish trend, then spike up sharply
        base = [100.0] * 20
        drop = list(np.linspace(100, 80, 15))   # induces bearish trend
        spike = [79, 78, 77, 200]                # big spike flips trend up on last bar
        return base + drop + spike

    def _bearish_flip_prices(self):
        """Price series that forces a bullish→bearish flip on last bar."""
        base = [100.0] * 20
        rise = list(np.linspace(100, 130, 15))  # induces bullish trend
        crash = [131, 132, 133, 10]             # crash flips trend down on last bar
        return base + rise + crash

    def test_long_signal_on_bullish_flip(self):
        prices = self._bullish_flip_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "long")
        self.assertEqual(result.strength_score, 3)
        self.assertIsNotNone(result.sl)
        self.assertIsNotNone(result.tp1)
        self.assertIsNotNone(result.tp2)

    def test_short_signal_on_bearish_flip(self):
        prices = self._bearish_flip_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "short")
        self.assertEqual(result.strength_score, 3)
        self.assertIsNotNone(result.sl)
        self.assertIsNotNone(result.tp1)
        self.assertIsNotNone(result.tp2)

    def test_hold_when_no_flip(self):
        # Flat price — no trend flip
        prices = [100.0] * 50
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_long_sl_below_entry(self):
        prices = self._bullish_flip_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        if result.signal_type == "long":
            self.assertLess(result.sl, result.entry_price)

    def test_long_tp1_above_entry(self):
        prices = self._bullish_flip_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        if result.signal_type == "long":
            self.assertGreater(result.tp1, result.entry_price)

    def test_long_tp2_above_tp1(self):
        prices = self._bullish_flip_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        if result.signal_type == "long":
            self.assertGreater(result.tp2, result.tp1)

    def test_short_sl_above_entry(self):
        prices = self._bearish_flip_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        if result.signal_type == "short":
            self.assertGreater(result.sl, result.entry_price)

    def test_short_tp1_below_entry(self):
        prices = self._bearish_flip_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        if result.signal_type == "short":
            self.assertLess(result.tp1, result.entry_price)

    def test_short_tp2_below_tp1(self):
        prices = self._bearish_flip_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        if result.signal_type == "short":
            self.assertLess(result.tp2, result.tp1)

    def test_insufficient_candles_returns_hold(self):
        df = _make_df([100.0] * 5)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_indicators_present(self):
        prices = self._bullish_flip_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        if result.signal_type == "long":
            self.assertIn("atr", result.indicators)
            self.assertIn("supertrend_up", result.indicators)
            self.assertIn("trend", result.indicators)

    def test_negative_sl_guard(self):
        # Tiny price with huge ATR → up band would go negative
        prices = [0.0001] * 50
        df = _make_df(prices, high=np.full(50, 0.01), low=np.zeros(50))
        result = _strategy().generate_signal(df, "TEST")
        # Should not return long with negative SL
        if result.signal_type == "long":
            self.assertGreater(result.sl, 0)

    def test_custom_params_respected(self):
        s = SupertrendStrategy({"atr_period": 5, "multiplier": 2.0,
                                "sl_atr_mult": 1.5, "tp1_atr_mult": 2.0,
                                "tp2_atr_mult": 4.0})
        p = {**s.DEFAULTS, **s.params}
        self.assertEqual(p["atr_period"], 5)
        self.assertAlmostEqual(p["multiplier"], 2.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSupertrendEdgeCases(unittest.TestCase):

    def test_all_same_price_no_crash(self):
        df = _make_df([50.0] * 60)
        result = _strategy().generate_signal(df, "TEST")
        self.assertIn(result.signal_type, ("hold", "long", "short"))

    def test_ascending_prices_no_crash(self):
        prices = list(range(50, 110))
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertIn(result.signal_type, ("hold", "long", "short"))

    def test_descending_prices_no_crash(self):
        prices = list(range(110, 50, -1))
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertIn(result.signal_type, ("hold", "long", "short"))

    def test_entry_price_equals_last_close(self):
        prices = self._bullish_flip_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        if result.signal_type == "long":
            self.assertAlmostEqual(result.entry_price, float(df["close"].iloc[-1]))

    def _bullish_flip_prices(self):
        base = [100.0] * 20
        drop = list(np.linspace(100, 80, 15))
        spike = [79, 78, 77, 200]
        return base + drop + spike
