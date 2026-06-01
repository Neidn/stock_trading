"""Unit tests for per-symbol strategy assignment logic."""

from __future__ import annotations

import sqlite3
import unittest

import numpy as np

from src.jobs.screener import (
    _assign_strategy,
    _classify_regime,
    _compute_symbol_indicators,
    _discover_strategies,
)
from src.signal.base_strategy import BaseStrategy
from src.signal.strategies.bb_rsi_chartart import BbRsiChartartStrategy
from src.signal.strategies.ema_crossover import EmaCrossoverStrategy
from src.signal.strategies.macd_sma200_chartart import MacdSma200ChartartStrategy
from src.signal.strategies.supertrend import SupertrendStrategy

ALL = [BbRsiChartartStrategy, EmaCrossoverStrategy,
       SupertrendStrategy, MacdSma200ChartartStrategy]


# ---------------------------------------------------------------------------
# suitability_score contracts
# ---------------------------------------------------------------------------

class TestSuitabilityScoreContracts(unittest.TestCase):

    def _ind(self, adx=25.0, atr_pct=1.0, above_sma200=False):
        return {"adx": adx, "atr_pct": atr_pct, "above_sma200": above_sma200}

    def test_all_scores_in_range(self):
        for cls in ALL:
            score = cls.suitability_score(self._ind())
            self.assertGreaterEqual(score, 0.0, f"{cls.__name__} score < 0")
            self.assertLessEqual(score, 1.0, f"{cls.__name__} score > 1")

    def test_base_strategy_default_is_0_5(self):
        # BaseStrategy.suitability_score should return 0.5
        ind = self._ind()
        self.assertAlmostEqual(BaseStrategy.suitability_score(ind), 0.5)

    def test_missing_indicators_no_crash(self):
        for cls in ALL:
            score = cls.suitability_score({})  # empty dict
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ---------------------------------------------------------------------------
# Regime differentiation: each regime has a clear winner
# ---------------------------------------------------------------------------

class TestRegimeDifferentiation(unittest.TestCase):

    def _ind(self, adx=25.0, atr_pct=1.0, above_sma200=False):
        return {"adx": adx, "atr_pct": atr_pct, "above_sma200": above_sma200}

    def test_ranging_market_prefers_bb_rsi(self):
        """ADX=10 (ranging) → bb_rsi_chartart wins."""
        ind = self._ind(adx=10, atr_pct=0.5)
        best = _assign_strategy(ind, ALL)
        self.assertEqual(best, "bb_rsi_chartart")

    def test_trending_market_prefers_ema_or_macd(self):
        """ADX=40 (strong trend) → ema_crossover or macd_sma200 wins, not bb_rsi."""
        ind = self._ind(adx=40, atr_pct=1.5)
        best = _assign_strategy(ind, ALL)
        self.assertNotEqual(best, "bb_rsi_chartart")

    def test_trending_volatile_prefers_supertrend(self):
        """ADX=45, ATR%=4.0 → supertrend wins (trend + volatility)."""
        ind = self._ind(adx=45, atr_pct=4.0)
        best = _assign_strategy(ind, ALL)
        self.assertEqual(best, "supertrend")

    def test_trending_above_sma200_prefers_macd(self):
        """ADX=40, above_sma200=True, ATR%=1.0 → macd_sma200 wins (structure boost)."""
        ind = self._ind(adx=40, atr_pct=1.0, above_sma200=True)
        best = _assign_strategy(ind, ALL)
        self.assertEqual(best, "macd_sma200_chartart")

    def test_bb_rsi_score_falls_with_rising_adx(self):
        low_adx  = BbRsiChartartStrategy.suitability_score({"adx": 10})
        high_adx = BbRsiChartartStrategy.suitability_score({"adx": 45})
        self.assertGreater(low_adx, high_adx)

    def test_ema_crossover_score_rises_with_rising_adx(self):
        low_adx  = EmaCrossoverStrategy.suitability_score({"adx": 10})
        high_adx = EmaCrossoverStrategy.suitability_score({"adx": 45})
        self.assertGreater(high_adx, low_adx)

    def test_supertrend_score_rises_with_atr_pct(self):
        low_vol  = SupertrendStrategy.suitability_score({"adx": 30, "atr_pct": 0.5})
        high_vol = SupertrendStrategy.suitability_score({"adx": 30, "atr_pct": 5.0})
        self.assertGreater(high_vol, low_vol)

    def test_macd_score_boosted_above_sma200(self):
        above = MacdSma200ChartartStrategy.suitability_score({"adx": 30, "above_sma200": True})
        below = MacdSma200ChartartStrategy.suitability_score({"adx": 30, "above_sma200": False})
        self.assertGreater(above, below)


# ---------------------------------------------------------------------------
# _assign_strategy
# ---------------------------------------------------------------------------

class TestClassifyRegime(unittest.TestCase):

    def test_high_adx_is_trending(self):
        self.assertEqual(_classify_regime({"adx": 30, "atr_pct": 1.0}), "trending")

    def test_low_adx_is_ranging(self):
        self.assertEqual(_classify_regime({"adx": 15, "atr_pct": 1.0}), "ranging")

    def test_high_atr_low_adx_is_volatile(self):
        self.assertEqual(_classify_regime({"adx": 20, "atr_pct": 4.0}), "volatile")

    def test_default_indicators_is_ranging(self):
        self.assertEqual(_classify_regime({}), "ranging")


class TestPrimaryRegimes(unittest.TestCase):

    def test_bb_rsi_is_ranging(self):
        self.assertIn("ranging", BbRsiChartartStrategy.primary_regimes())

    def test_ema_crossover_is_trending(self):
        self.assertIn("trending", EmaCrossoverStrategy.primary_regimes())

    def test_supertrend_includes_volatile(self):
        self.assertIn("volatile", SupertrendStrategy.primary_regimes())

    def test_macd_sma200_is_trending(self):
        self.assertIn("trending", MacdSma200ChartartStrategy.primary_regimes())

    def test_base_default_is_any(self):
        self.assertIn("any", BaseStrategy.primary_regimes())

    def test_regime_filter_excludes_bb_rsi_in_trending(self):
        """In trending regime, bb_rsi is not eligible → cannot win."""
        ind = {"adx": 40, "atr_pct": 1.5, "above_sma200": False}
        # bb_rsi declares 'ranging' only → excluded in trending regime
        result = _assign_strategy(ind, ALL)
        self.assertNotEqual(result, "bb_rsi_chartart")

    def test_regime_filter_excludes_ema_in_ranging(self):
        """In ranging regime, ema_crossover not eligible → cannot win."""
        ind = {"adx": 10, "atr_pct": 0.5, "above_sma200": False}
        result = _assign_strategy(ind, ALL)
        self.assertNotEqual(result, "ema_crossover")

    def test_poorly_calibrated_strategy_blocked_by_regime(self):
        """Strategy always returning 0.9 but wrong regime cannot win."""
        class CheatStrategy(BaseStrategy):
            DEFAULTS = {}
            @classmethod
            def primary_regimes(cls): return frozenset({"trending"})
            @classmethod
            def suitability_score(cls, ind): return 0.9  # always 0.9
            def get_name(self): return "cheat"
            def get_min_candles(self): return 1
            def get_timeframe(self): return "1h"
            def generate_signal(self, df, symbol): ...

        strategies = ALL + [CheatStrategy]
        # ranging market — CheatStrategy declares 'trending' only → excluded
        ind = {"adx": 10, "atr_pct": 0.5, "above_sma200": False}
        result = _assign_strategy(ind, strategies)
        self.assertNotEqual(result, "cheat")


class TestAssignStrategy(unittest.TestCase):

    def test_returns_string(self):
        result = _assign_strategy({"adx": 25}, ALL)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_fallback_when_empty_list(self):
        result = _assign_strategy({"adx": 25}, [])
        self.assertIsInstance(result, str)

    def test_single_strategy_always_wins(self):
        result = _assign_strategy({"adx": 50}, [BbRsiChartartStrategy])
        self.assertEqual(result, "bb_rsi_chartart")


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

class TestDiscoverStrategies(unittest.TestCase):

    def test_returns_list(self):
        strategies = _discover_strategies()
        self.assertIsInstance(strategies, list)

    def test_finds_at_least_4(self):
        strategies = _discover_strategies()
        self.assertGreaterEqual(len(strategies), 4)

    def test_all_subclass_base_strategy(self):
        for cls in _discover_strategies():
            self.assertTrue(issubclass(cls, BaseStrategy),
                            f"{cls.__name__} not a BaseStrategy subclass")

    def test_all_have_suitability_score(self):
        for cls in _discover_strategies():
            self.assertTrue(callable(getattr(cls, "suitability_score", None)),
                            f"{cls.__name__} missing suitability_score")

    def test_scores_valid_range(self):
        ind = {"adx": 25.0, "atr_pct": 1.0, "above_sma200": False}
        for cls in _discover_strategies():
            score = cls.suitability_score(ind)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ---------------------------------------------------------------------------
# _compute_symbol_indicators
# ---------------------------------------------------------------------------

class TestComputeSymbolIndicators(unittest.TestCase):

    def _make_conn(self, n_rows=250) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE klines (
                symbol TEXT, interval_type TEXT, open_time TEXT,
                high TEXT, low TEXT, close TEXT,
                open TEXT, volume TEXT, close_time TEXT
            )"""
        )
        prices = np.linspace(100, 130, n_rows)
        for i, p in enumerate(prices):
            conn.execute(
                "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
                ("BTCUSDT", "1d", str(i * 86400000),
                 str(p * 1.01), str(p * 0.99), str(p),
                 str(p), "1000", str((i + 1) * 3600000)),
            )
        conn.commit()
        return conn

    def test_returns_dict_with_required_keys(self):
        conn = self._make_conn()
        ind = _compute_symbol_indicators(conn, "BTCUSDT")
        self.assertIn("adx", ind)
        self.assertIn("atr_pct", ind)
        self.assertIn("above_sma200", ind)

    def test_adx_in_valid_range(self):
        conn = self._make_conn()
        ind = _compute_symbol_indicators(conn, "BTCUSDT")
        self.assertGreaterEqual(ind["adx"], 0.0)
        self.assertLessEqual(ind["adx"], 100.0)

    def test_atr_pct_positive(self):
        conn = self._make_conn()
        ind = _compute_symbol_indicators(conn, "BTCUSDT")
        self.assertGreater(ind["atr_pct"], 0.0)

    def test_above_sma200_bool(self):
        conn = self._make_conn()
        ind = _compute_symbol_indicators(conn, "BTCUSDT")
        self.assertIsInstance(ind["above_sma200"], bool)

    def test_safe_defaults_when_insufficient_data(self):
        conn = self._make_conn(n_rows=5)
        ind = _compute_symbol_indicators(conn, "BTCUSDT")
        self.assertEqual(ind["adx"], 25.0)
        self.assertEqual(ind["atr_pct"], 1.0)

    def test_unknown_symbol_returns_defaults(self):
        conn = self._make_conn()
        ind = _compute_symbol_indicators(conn, "UNKNOWN")
        self.assertEqual(ind["adx"], 25.0)
