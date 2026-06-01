"""Unit tests for Week52HighStrategy."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.signal.strategies.week52_high import Week52HighStrategy


def _make_df(close, high=None, low=None, volume=None) -> pd.DataFrame:
    close = np.array(close, dtype=float)
    if high is None:
        high = close * 1.01
    if low is None:
        low = close * 0.99
    if volume is None:
        volume = np.ones(len(close)) * 1000
    return pd.DataFrame({
        "open":   close,
        "high":   np.array(high, dtype=float),
        "low":    np.array(low,  dtype=float),
        "close":  close,
        "volume": np.array(volume, dtype=float),
    })


def _strategy(overrides=None) -> Week52HighStrategy:
    params = {
        "n_days":        10,
        "vol_threshold": 0.0,   # disable volume filter so signal fires cleanly
        "adx_min":       0.0,   # disable ADX filter
        "sl_atr_mult":   2.0,
        "tp1_atr_mult":  3.0,
        "tp2_atr_mult":  5.0,
    }
    if overrides:
        params.update(overrides)
    return Week52HighStrategy(params)


def _breakout_prices(n_days=10, base=100.0, spike=150.0):
    """n_days flat base then one spike above."""
    return [base] * (n_days + 5) + [spike]


class TestWeek52HighDefaults(unittest.TestCase):

    def test_name(self):
        self.assertEqual(Week52HighStrategy({}).get_name(), "week52_high")

    def test_timeframe(self):
        self.assertEqual(Week52HighStrategy({}).get_timeframe(), "1d")

    def test_min_candles_gt_n_days(self):
        s = Week52HighStrategy({"n_days": 100})
        self.assertGreater(s.get_min_candles(), 100)

    def test_defaults_applied(self):
        s = Week52HighStrategy({})
        p = {**s.DEFAULTS, **s.params}
        self.assertEqual(p["n_days"], 100)
        self.assertEqual(p["vol_threshold"], 1.3)
        self.assertEqual(p["adx_min"], 20.0)


class TestWeek52HighSuitability(unittest.TestCase):

    def test_high_adx_high_score(self):
        score = Week52HighStrategy.suitability_score({"adx": 50, "above_sma200": True, "atr_pct": 1.0})
        self.assertGreater(score, 0.7)

    def test_low_adx_low_score(self):
        score = Week52HighStrategy.suitability_score({"adx": 5, "above_sma200": False, "atr_pct": 1.0})
        self.assertLess(score, 0.3)

    def test_above_sma200_adds_bonus(self):
        s_with    = Week52HighStrategy.suitability_score({"adx": 30, "above_sma200": True,  "atr_pct": 1.0})
        s_without = Week52HighStrategy.suitability_score({"adx": 30, "above_sma200": False, "atr_pct": 1.0})
        self.assertGreater(s_with, s_without)

    def test_high_atr_reduces_score(self):
        s_normal = Week52HighStrategy.suitability_score({"adx": 30, "above_sma200": False, "atr_pct": 1.0})
        s_volatile = Week52HighStrategy.suitability_score({"adx": 30, "above_sma200": False, "atr_pct": 6.0})
        self.assertLess(s_volatile, s_normal)

    def test_score_capped_at_0_9(self):
        score = Week52HighStrategy.suitability_score({"adx": 100, "above_sma200": True, "atr_pct": 1.0})
        self.assertLessEqual(score, 0.90)

    def test_score_non_negative(self):
        score = Week52HighStrategy.suitability_score({"adx": 0, "above_sma200": False, "atr_pct": 10.0})
        self.assertGreaterEqual(score, 0.0)


class TestWeek52HighLong(unittest.TestCase):

    def test_long_signal_on_breakout(self):
        prices = _breakout_prices()
        df = _make_df(prices, volume=np.ones(len(prices)) * 2000)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "long")

    def test_sl_below_entry(self):
        prices = _breakout_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertLess(result.sl, result.entry_price)

    def test_tp1_above_entry(self):
        prices = _breakout_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.tp1, result.entry_price)

    def test_tp2_above_tp1(self):
        prices = _breakout_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.tp2, result.tp1)

    def test_strength_score_at_least_2(self):
        prices = _breakout_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreaterEqual(result.strength_score, 2)

    def test_max_strength_3_when_all_conditions_met(self):
        prices = _breakout_prices()
        vol = np.ones(len(prices)) * 5000  # high volume → ratio > vol_threshold
        df = _make_df(prices, volume=vol)
        result = _strategy({"adx_min": 0.0, "vol_threshold": 0.0}).generate_signal(df, "TEST")
        self.assertEqual(result.strength_score, 3)

    def test_indicators_present(self):
        prices = _breakout_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        for key in ("prior_high", "cur_close", "adx", "atr", "vol_ratio"):
            self.assertIn(key, result.indicators)


class TestWeek52HighHold(unittest.TestCase):

    def test_hold_when_no_breakout(self):
        # Flat prices — last bar equals the prior high, not above
        prices = [100.0] * 20
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_hold_when_price_below_prior_high(self):
        prices = [100.0] * 15 + [90.0]  # last bar below prior high
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_hold_on_insufficient_data(self):
        prices = [100.0] * 5  # fewer than n_days+1
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")
        self.assertIn("insufficient", result.reason)

    def test_hold_when_volume_and_adx_filters_block(self):
        prices = _breakout_prices()
        df = _make_df(prices)
        # Both vol and ADX blocked → new_high alone = strength 1 < 2 → hold
        result = _strategy({"vol_threshold": 999.0, "adx_min": 999.0}).generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_hold_when_both_filters_block(self):
        prices = _breakout_prices()
        df = _make_df(prices)
        result = _strategy({"vol_threshold": 999.0, "adx_min": 999.0}).generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")


class TestWeek52HighPrimaryRegimes(unittest.TestCase):

    def test_regimes_include_trending_and_volatile(self):
        regimes = Week52HighStrategy.primary_regimes()
        self.assertIn("trending", regimes)
        self.assertIn("volatile", regimes)
