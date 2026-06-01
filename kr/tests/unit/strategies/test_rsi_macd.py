"""Unit tests for RsiMacdStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signal.strategies.rsi_macd import RsiMacdStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a generic OHLCV DataFrame with realistic price-like data."""
    rng = np.random.default_rng(seed)
    close = 50_000 + np.cumsum(rng.normal(0, 200, n))
    high = close + rng.uniform(50, 300, n)
    low = close - rng.uniform(50, 300, n)
    open_ = close + rng.normal(0, 100, n)
    volume = rng.uniform(100, 1_000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _make_long_df(n: int = 200) -> pd.DataFrame:
    """DataFrame engineered to produce RSI < 35, MACD hist rising, price > EMA50, high volume."""
    rng = np.random.default_rng(0)

    # Strong downtrend for first 150 bars → RSI gets oversold
    close = np.empty(n)
    close[0] = 50_000
    for i in range(1, 150):
        close[i] = close[i - 1] - rng.uniform(10, 80)   # falling → low RSI
    # Last 50 bars: slight recovery (MACD hist starts turning up)
    for i in range(150, n):
        close[i] = close[i - 1] + rng.uniform(5, 40)

    high = close + rng.uniform(20, 100, n)
    low = close - rng.uniform(20, 100, n)
    open_ = close + rng.normal(0, 30, n)

    # Last bar: spike in volume
    volume = rng.uniform(100, 300, n)
    volume[-1] = volume[:-1].mean() * 3.0   # vol_ratio >> 1.3

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _make_short_df(n: int = 200) -> pd.DataFrame:
    """Decelerating uptrend: RSI > 65, MACD hist falling, volume spike → short_score=3 > long_score."""
    rng = np.random.default_rng(1)

    close = np.empty(n)
    close[0] = 30_000
    # Strong consistent uptrend for 190 bars → RSI > 65
    for i in range(1, 190):
        close[i] = close[i - 1] + 50.0
    # Decelerating uptrend → MACD hist peaks and falls
    for i in range(190, 195):
        close[i] = close[i - 1] + 30.0
    for i in range(195, n):
        close[i] = close[i - 1] + 5.0

    high = close + rng.uniform(20, 80, n)
    low = close - rng.uniform(20, 80, n)
    open_ = close + rng.normal(0, 20, n)

    volume = rng.uniform(100, 300, n)
    volume[-1] = volume[:-1].mean() * 3.0   # vol_ratio >> 1.3

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRsiMacdStrategy:
    def test_long_signal_generated(self):
        """RSI oversold + MACD hist rising + high volume → long signal, score >= 2."""
        strategy = RsiMacdStrategy({})
        df = _make_long_df()
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "long", f"Expected long, got {result.signal_type}: {result.reason}"
        assert result.strength_score >= 2

    def test_short_signal_generated(self):
        """RSI overbought + MACD hist falling + high volume → short signal."""
        strategy = RsiMacdStrategy({})
        df = _make_short_df()
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "short", f"Expected short, got {result.signal_type}: {result.reason}"
        assert result.strength_score >= 2

    def test_no_signal_when_conditions_insufficient(self):
        """Only 1 condition met per direction → signal_type == 'none'.

        Design: extreme rsi thresholds and vol_multiplier eliminate conditions 1 and 4.
        A downtrend with a last-bar bounce ensures MACD hist rising (long_cond2 ✓) while
        close < EMA50 (short_cond3 ✓) — conditions pointing OPPOSITE directions so neither
        direction reaches a score of 2.
        """
        # Eliminate RSI and volume conditions so at most cond2 + cond3 can fire,
        # but designed to fire in opposite directions (score = 1 each → 'none')
        params = {
            "rsi_oversold": 5,
            "rsi_overbought": 95,
            "vol_multiplier": 50.0,   # impossible to achieve
            "sl_atr_mult": 1.0,
            "tp1_atr_mult": 2.0,
            "tp2_atr_mult": 3.0,
        }
        strategy = RsiMacdStrategy(params)

        n = 200
        # Steady downtrend → close < EMA50 (short_cond3 ✓, long_cond3 ✗)
        close = np.empty(n)
        close[0] = 50_000
        for i in range(1, n - 1):
            close[i] = close[i - 1] - 40.0
        # Last bar bounces up → MACD hist turns up (long_cond2 ✓, short_cond2 ✗)
        close[n - 1] = close[n - 2] + 600.0

        high = close + 30
        low = close - 30
        open_ = close.copy()
        volume = np.full(n, 200.0)   # vol_ratio = 1.0 << 50.0

        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "none", f"Expected none, got {result.signal_type}: {result.reason}"

    def test_insufficient_data_returns_none(self):
        """DataFrame shorter than get_min_candles() → is_data_sufficient() is False."""
        strategy = RsiMacdStrategy({})
        min_candles = strategy.get_min_candles()
        df = _make_df(n=min_candles - 1)
        assert strategy.is_data_sufficient(df) is False

    def test_sl_above_liquidation(self):
        """Stop-loss must be within sl_atr_mult × ATR of entry and above 5x liq price."""
        params = {"sl_atr_mult": 2.0, "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0}
        strategy = RsiMacdStrategy(params)
        df = _make_long_df()
        result = strategy.generate_signal(df, "BTCUSDT")

        if result.signal_type != "long":
            pytest.skip(f"No long signal generated for SL test: {result.reason}")

        entry = result.entry_price
        sl = result.sl
        atr = result.indicators["atr"]
        sl_atr_mult = params["sl_atr_mult"]

        # SL distance should be approximately sl_atr_mult × ATR
        sl_distance = entry - sl
        assert abs(sl_distance - sl_atr_mult * atr) < 1e-6, (
            f"SL distance {sl_distance:.4f} ≠ {sl_atr_mult} × ATR {atr:.4f}"
        )

        # 5× leverage liquidation price for long: entry × (1 - 1/leverage) = entry × 0.8
        leverage = 5
        liq_price = entry * (1 - 1 / leverage)
        assert sl > liq_price, (
            f"Stop-loss {sl:.2f} is below 5x liquidation price {liq_price:.2f}"
        )

    def test_indicators_snapshot_included(self):
        """SignalResult.indicators must contain rsi, macd_hist, and atr keys."""
        strategy = RsiMacdStrategy({})
        df = _make_df(n=200)
        result = strategy.generate_signal(df, "BTCUSDT")
        # indicators snapshot is filled regardless of signal direction
        for key in ("rsi", "macd_hist", "atr"):
            assert key in result.indicators, f"Missing key '{key}' in indicators"

    def test_get_timeframe(self):
        strategy = RsiMacdStrategy({})
        assert strategy.get_timeframe() == "1d"
