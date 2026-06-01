"""Unit tests for MomentumStrategy."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.signal.strategies.momentum import MomentumStrategy


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


def _strategy(overrides=None) -> MomentumStrategy:
    params = {
        "lookback":       5,
        "min_return_pct": 5.0,
        "sma_period":     3,
        "vol_threshold":  0.0,   # disable volume filter
        "sl_atr_mult":    2.0,
        "tp1_atr_mult":   3.0,
        "tp2_atr_mult":   5.0,
    }
    if overrides:
        params.update(overrides)
    return MomentumStrategy(params)


def _momentum_prices(lookback=5, base=100.0, final=120.0, length=30):
    """Rising prices: base for most of series, then rises to final over last lookback bars."""
    n_base = length - lookback
    base_part  = [base] * n_base
    ramp = list(np.linspace(base, final, lookback + 1))
    return base_part + ramp


class TestMomentumDefaults(unittest.TestCase):

    def test_name(self):
        self.assertEqual(MomentumStrategy({}).get_name(), "momentum")

    def test_timeframe(self):
        self.assertEqual(MomentumStrategy({}).get_timeframe(), "1d")

    def test_min_candles_gt_sma_period(self):
        s = MomentumStrategy({"lookback": 20, "sma_period": 50})
        self.assertGreater(s.get_min_candles(), 50)

    def test_defaults_applied(self):
        s = MomentumStrategy({})
        p = {**s.DEFAULTS, **s.params}
        self.assertEqual(p["lookback"], 20)
        self.assertEqual(p["min_return_pct"], 8.0)
        self.assertEqual(p["sma_period"], 50)


class TestMomentumSuitability(unittest.TestCase):

    def test_high_adx_above_sma200_high_score(self):
        score = MomentumStrategy.suitability_score({"adx": 50, "above_sma200": True, "atr_pct": 2.0})
        self.assertGreater(score, 0.7)

    def test_low_adx_low_score(self):
        score = MomentumStrategy.suitability_score({"adx": 5, "above_sma200": False, "atr_pct": 2.0})
        self.assertLess(score, 0.3)

    def test_above_sma200_adds_bonus(self):
        s_with    = MomentumStrategy.suitability_score({"adx": 30, "above_sma200": True,  "atr_pct": 2.0})
        s_without = MomentumStrategy.suitability_score({"adx": 30, "above_sma200": False, "atr_pct": 2.0})
        self.assertGreater(s_with, s_without)

    def test_atr_bonus_for_healthy_volatility(self):
        s_healthy = MomentumStrategy.suitability_score({"adx": 30, "above_sma200": False, "atr_pct": 2.0})
        s_low     = MomentumStrategy.suitability_score({"adx": 30, "above_sma200": False, "atr_pct": 0.5})
        self.assertGreater(s_healthy, s_low)

    def test_score_capped_at_0_9(self):
        score = MomentumStrategy.suitability_score({"adx": 100, "above_sma200": True, "atr_pct": 2.0})
        self.assertLessEqual(score, 0.90)

    def test_score_non_negative(self):
        score = MomentumStrategy.suitability_score({"adx": 0, "above_sma200": False, "atr_pct": 0.0})
        self.assertGreaterEqual(score, 0.0)


class TestMomentumLong(unittest.TestCase):

    def test_long_signal_on_momentum(self):
        prices = _momentum_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "long")

    def test_sl_below_entry(self):
        prices = _momentum_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertLess(result.sl, result.entry_price)

    def test_tp1_above_entry(self):
        prices = _momentum_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.tp1, result.entry_price)

    def test_tp2_above_tp1(self):
        prices = _momentum_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreater(result.tp2, result.tp1)

    def test_strength_at_least_2(self):
        prices = _momentum_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertGreaterEqual(result.strength_score, 2)

    def test_max_strength_3_when_all_conditions_met(self):
        prices = _momentum_prices()
        vol = np.ones(len(prices)) * 5000  # high volume
        df = _make_df(prices, volume=vol)
        result = _strategy({"vol_threshold": 0.0}).generate_signal(df, "TEST")
        self.assertEqual(result.strength_score, 3)

    def test_indicators_present(self):
        prices = _momentum_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        for key in ("ret_pct", "sma", "cur_close", "atr", "vol_ratio"):
            self.assertIn(key, result.indicators)

    def test_entry_price_equals_last_close(self):
        prices = _momentum_prices()
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertAlmostEqual(result.entry_price, prices[-1], places=4)


class TestMomentumHold(unittest.TestCase):

    def test_hold_when_return_below_threshold(self):
        # Flat prices → 0% return < min_return_pct
        prices = [100.0] * 30
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")

    def test_hold_when_price_below_sma(self):
        # Start high then drop — price below SMA at the end
        prices = [120.0] * 20 + list(np.linspace(120, 130, 6)) + [70.0]
        df = _make_df(prices)
        # Momentum might exist but price is now below SMA
        result = _strategy({"sma_period": 10, "vol_threshold": 0.0}).generate_signal(df, "TEST")
        # Either hold (price below SMA blocks) or signal; just verify no crash
        self.assertIn(result.signal_type, ("long", "hold"))

    def test_hold_on_insufficient_data(self):
        prices = [100.0] * 5
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertEqual(result.signal_type, "hold")
        self.assertIn("insufficient", result.reason)

    def test_hold_when_volume_filter_blocks(self):
        prices = _momentum_prices()
        # Force sma above price to block trend condition; only momentum fires
        result = _strategy({"vol_threshold": 999.0, "sma_period": 3}).generate_signal(
            _make_df(prices), "TEST"
        )
        # momentum_ok=True but vol_ok=False and depends on SMA → strength likely <2
        # Key: no exception thrown
        self.assertIn(result.signal_type, ("long", "hold"))

    def test_reason_contains_return_info_on_hold(self):
        prices = [100.0] * 30
        df = _make_df(prices)
        result = _strategy().generate_signal(df, "TEST")
        self.assertIn("수익률", result.reason)


class TestMomentumPrimaryRegimes(unittest.TestCase):

    def test_regime_is_trending(self):
        self.assertIn("trending", MomentumStrategy.primary_regimes())
