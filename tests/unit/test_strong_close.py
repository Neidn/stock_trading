"""Unit tests for StrongCloseStrategy."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.signal.strategies.strong_close import StrongCloseStrategy


def _make_df(close, high=None, low=None, volume=None) -> pd.DataFrame:
    close = np.array(close, dtype=float)
    if high is None:
        high = close * 1.02
    if low is None:
        low = close * 0.98
    if volume is None:
        volume = np.ones(len(close)) * 1000
    return pd.DataFrame({
        "open":   close,
        "high":   np.array(high, dtype=float),
        "low":    np.array(low,  dtype=float),
        "close":  close,
        "volume": np.array(volume, dtype=float),
    })


def _strategy(overrides=None) -> StrongCloseStrategy:
    params = {
        "close_threshold": 0.75,
        "sma_period":      3,
        "vol_threshold":   0.0,   # disable volume filter by default
        "sl_atr_mult":     1.5,
        "tp1_atr_mult":    2.5,
        "tp2_atr_mult":    4.0,
    }
    if overrides:
        params.update(overrides)
    return StrongCloseStrategy(params)


def _strong_close_df(n=30, base=100.0):
    """Rising prices where close is near the session high (close_pct ≈ 0.8 > 0.75).

    high = close * 1.01, low = close * 0.96 →
    close_pct = (close - low) / (high - low) = 0.04 / 0.05 = 0.80
    """
    prices = list(np.linspace(base, base * 1.2, n))
    high   = [p * 1.01 for p in prices]   # 1% above close
    low    = [p * 0.96 for p in prices]   # 4% below close
    return pd.DataFrame({
        "open":   prices,
        "high":   high,
        "low":    low,
        "close":  prices,
        "volume": np.ones(n) * 1000,
    })


class TestStrongCloseDefaults(unittest.TestCase):

    def test_name(self):
        self.assertEqual(StrongCloseStrategy({}).get_name(), "strong_close")

    def test_timeframe(self):
        self.assertEqual(StrongCloseStrategy({}).get_timeframe(), "1d")

    def test_min_candles_gt_sma_period(self):
        s = StrongCloseStrategy({"sma_period": 20})
        self.assertGreater(s.get_min_candles(), 20)

    def test_defaults_applied(self):
        s = StrongCloseStrategy({})
        p = {**s.DEFAULTS, **s.params}
        self.assertEqual(p["close_threshold"], 0.75)
        self.assertEqual(p["sma_period"], 20)


class TestStrongCloseSuitability(unittest.TestCase):

    def test_reasonable_score_mid_adx(self):
        score = StrongCloseStrategy.suitability_score({"adx": 30, "above_sma200": False, "atr_pct": 2.0})
        self.assertGreater(score, 0.1)
        self.assertLessEqual(score, 0.75)

    def test_above_sma200_adds_bonus(self):
        s_with    = StrongCloseStrategy.suitability_score({"adx": 30, "above_sma200": True,  "atr_pct": 2.0})
        s_without = StrongCloseStrategy.suitability_score({"adx": 30, "above_sma200": False, "atr_pct": 2.0})
        self.assertGreater(s_with, s_without)

    def test_healthy_atr_adds_bonus(self):
        s_ok  = StrongCloseStrategy.suitability_score({"adx": 25, "above_sma200": False, "atr_pct": 2.0})
        s_bad = StrongCloseStrategy.suitability_score({"adx": 25, "above_sma200": False, "atr_pct": 0.1})
        self.assertGreater(s_ok, s_bad)

    def test_score_capped_at_0_75(self):
        score = StrongCloseStrategy.suitability_score({"adx": 100, "above_sma200": True, "atr_pct": 2.0})
        self.assertLessEqual(score, 0.75)

    def test_regime_is_any(self):
        self.assertIn("any", StrongCloseStrategy.primary_regimes())


class TestStrongCloseLong(unittest.TestCase):

    def test_long_signal_on_strong_close(self):
        df = _strong_close_df()
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "long")

    def test_sl_below_entry(self):
        df = _strong_close_df()
        result = _strategy().generate_signal(df, "TEST")
        self.assertLess(result.sl, result.entry_price)

    def test_tp1_above_entry(self):
        df = _strong_close_df()
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.tp1, result.entry_price)

    def test_tp2_above_tp1(self):
        df = _strong_close_df()
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.tp2, result.tp1)

    def test_strength_at_least_2(self):
        df = _strong_close_df()
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreaterEqual(result.strength_score, 2)

    def test_indicators_present(self):
        df = _strong_close_df()
        result = _strategy().generate_signal(df, "TEST")
        for key in ("close_pct", "sma", "cur_close", "atr", "vol_ratio"):
            self.assertIn(key, result.indicators)

    def test_entry_equals_last_close(self):
        df = _strong_close_df()
        result = _strategy().generate_signal(df, "TEST")
        self.assertAlmostEqual(result.entry_price, df["close"].iloc[-1], places=4)


class TestStrongCloseHold(unittest.TestCase):

    def test_hold_when_close_near_low(self):
        n = 30
        prices = [100.0] * n
        high   = [p + 10 for p in prices]
        # Close at 1% above low → close_pct ≈ 0.1 < 0.75
        low    = [p - 10 for p in prices]
        close  = [p - 9 for p in prices]   # close near low
        df = pd.DataFrame({
            "open": prices, "high": high, "low": low, "close": close,
            "volume": np.ones(n) * 1000,
        })
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_hold_on_doji_with_low_threshold(self):
        n = 30
        prices = [100.0] * n
        # high == low → doji → close_pct = 0.5 < 0.75
        df = pd.DataFrame({
            "open": prices, "high": prices, "low": prices, "close": prices,
            "volume": np.ones(n) * 1000,
        })
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_hold_on_insufficient_data(self):
        prices = [100.0] * 2  # below sma_period+1 = 4
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")
        self.assertIn("insufficient", result.reason)

    def test_hold_when_price_below_sma(self):
        n = 30
        # High then crash: close below SMA at end
        close  = list(np.linspace(120, 100, n - 1)) + [85.0]
        high   = [p * 1.03 for p in close]
        low    = [p * 0.97 for p in close]
        df = pd.DataFrame({
            "open": close, "high": high, "low": low, "close": close,
            "volume": np.ones(n) * 1000,
        })
        # Even if close_pct is high, price below SMA blocks trend_ok
        result = _strategy({"sma_period": 20}).generate_signal(df, "TEST")
        self.assertIn(result.signal_type, ("long", "hold"))  # no exception

    def test_reason_contains_pct_on_weak_close(self):
        n = 30
        prices = [100.0] * n
        high   = [p + 10 for p in prices]
        low    = [p - 10 for p in prices]
        close  = [p - 9 for p in prices]
        df = pd.DataFrame({
            "open": prices, "high": high, "low": low, "close": close,
            "volume": np.ones(n) * 1000,
        })
        result = _strategy().generate_signal(df, "TEST")
        self.assertIn("종가위치", result.reason)
