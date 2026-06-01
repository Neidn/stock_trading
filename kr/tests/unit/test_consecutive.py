"""Unit tests for ConsecutiveStrategy."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.signal.strategies.consecutive import ConsecutiveStrategy


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


def _strategy(overrides=None) -> ConsecutiveStrategy:
    params = {
        "n_days":        3,
        "sma_period":    3,
        "vol_threshold": 0.0,   # disable volume filter by default
        "sl_atr_mult":   2.0,
        "tp1_atr_mult":  3.0,
        "tp2_atr_mult":  5.0,
    }
    if overrides:
        params.update(overrides)
    return ConsecutiveStrategy(params)


def _consecutive_prices(n_base=20, n_up=3, base=100.0, step=2.0):
    """Flat base then n_up consecutive up closes."""
    base_part = [base] * n_base
    up_part   = [base + step * (i + 1) for i in range(n_up)]
    return base_part + up_part


class TestConsecutiveDefaults(unittest.TestCase):

    def test_name(self):
        self.assertEqual(ConsecutiveStrategy({}).get_name(), "consecutive")

    def test_timeframe(self):
        self.assertEqual(ConsecutiveStrategy({}).get_timeframe(), "1d")

    def test_min_candles_sufficient(self):
        s = ConsecutiveStrategy({"n_days": 3, "sma_period": 20})
        self.assertGreater(s.get_min_candles(), 20)

    def test_defaults_applied(self):
        s = ConsecutiveStrategy({})
        p = {**s.DEFAULTS, **s.params}
        self.assertEqual(p["n_days"], 3)
        self.assertEqual(p["sma_period"], 20)
        self.assertEqual(p["vol_threshold"], 1.1)


class TestConsecutiveSuitability(unittest.TestCase):

    def test_high_adx_high_score(self):
        score = ConsecutiveStrategy.suitability_score({"adx": 50, "above_sma200": True, "atr_pct": 1.0})
        self.assertGreater(score, 0.5)

    def test_low_adx_low_score(self):
        score = ConsecutiveStrategy.suitability_score({"adx": 5, "above_sma200": False, "atr_pct": 1.0})
        self.assertLess(score, 0.3)

    def test_above_sma200_adds_bonus(self):
        s_with    = ConsecutiveStrategy.suitability_score({"adx": 30, "above_sma200": True,  "atr_pct": 1.0})
        s_without = ConsecutiveStrategy.suitability_score({"adx": 30, "above_sma200": False, "atr_pct": 1.0})
        self.assertGreater(s_with, s_without)

    def test_low_atr_adds_small_bonus(self):
        s_ok  = ConsecutiveStrategy.suitability_score({"adx": 25, "above_sma200": False, "atr_pct": 2.0})
        s_bad = ConsecutiveStrategy.suitability_score({"adx": 25, "above_sma200": False, "atr_pct": 5.0})
        self.assertGreaterEqual(s_ok, s_bad)

    def test_score_capped_at_0_8(self):
        score = ConsecutiveStrategy.suitability_score({"adx": 100, "above_sma200": True, "atr_pct": 1.0})
        self.assertLessEqual(score, 0.80)

    def test_regime_is_trending(self):
        self.assertIn("trending", ConsecutiveStrategy.primary_regimes())


class TestConsecutiveLong(unittest.TestCase):

    def test_long_signal_on_consecutive_up(self):
        prices = _consecutive_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "long")

    def test_sl_below_entry(self):
        prices = _consecutive_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertLess(result.sl, result.entry_price)

    def test_tp1_above_entry(self):
        prices = _consecutive_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.tp1, result.entry_price)

    def test_tp2_above_tp1(self):
        prices = _consecutive_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.tp2, result.tp1)

    def test_strength_at_least_2(self):
        prices = _consecutive_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreaterEqual(result.strength_score, 2)

    def test_indicators_present(self):
        prices = _consecutive_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        for key in ("consecutive", "n_days", "sma", "cur_close", "atr", "vol_ratio"):
            self.assertIn(key, result.indicators)

    def test_entry_equals_last_close(self):
        prices = _consecutive_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertAlmostEqual(result.entry_price, prices[-1], places=4)


class TestConsecutiveHold(unittest.TestCase):

    def test_hold_when_last_day_breaks_streak(self):
        # 2 up days then down — streak broken
        prices = [100.0] * 20 + [102.0, 104.0, 103.0]
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_hold_on_flat_prices(self):
        prices = [100.0] * 30
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_hold_on_insufficient_data(self):
        prices = [100.0] * 2  # far below needed = max(3+1, 3)+1 = 5
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")
        self.assertIn("insufficient", result.reason)

    def test_hold_when_only_2_up_days_with_n3(self):
        # Only 2 consecutive up days, needs 3
        prices = [100.0] * 20 + [102.0, 104.0]
        df = _make_df(prices)
        result = _strategy({"n_days": 3}).generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_hold_reason_on_no_streak(self):
        prices = [100.0] * 30
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertIn("연속", result.reason)

    def test_longer_streak_required_still_fires(self):
        # 5 consecutive up days — n_days=5 should fire
        prices = _consecutive_prices(n_up=5, step=2.0)
        df = _make_df(prices)
        result = _strategy({"n_days": 5}).generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "long")
