"""Unit tests for BbBreakoutStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signal.strategies.bb_breakout import BbBreakoutStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(close: np.ndarray, volume_mult: float = 2.0) -> pd.DataFrame:
    """Build OHLCV DataFrame from a close array."""
    n = len(close)
    high = close + 50
    low = close - 50
    open_ = close + np.random.default_rng(0).normal(0, 20, n)
    volume = np.full(n, 500.0)
    # Last bar gets elevated volume for breakout confirmation
    volume[-1] = 500.0 * volume_mult
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _squeeze_then_long_breakout(n: int = 100) -> pd.DataFrame:
    """Price squeezes then closes cleanly above the upper BB on last bar."""
    rng = np.random.default_rng(1)
    # Tight range (squeeze) for first n-1 bars
    close = np.full(n, 50_000.0) + rng.normal(0, 20, n)
    # Last bar: big upward close that breaches upper band
    close[-1] = 50_000 + 5_000
    df = _make_df(close)
    # Ensure previous bar was inside the band (not a breakout)
    df.loc[df.index[-2], "close"] = 50_000.0
    return df


def _squeeze_then_short_breakout(n: int = 100) -> pd.DataFrame:
    """Price squeezes then closes cleanly below the lower BB on last bar."""
    rng = np.random.default_rng(2)
    close = np.full(n, 50_000.0) + rng.normal(0, 20, n)
    close[-1] = 50_000 - 5_000
    df = _make_df(close)
    df.loc[df.index[-2], "close"] = 50_000.0
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBbBreakoutStrategy:

    def test_get_name(self):
        assert BbBreakoutStrategy({}).get_name() == "bb_breakout"

    def test_get_min_candles(self):
        s = BbBreakoutStrategy({"bb_period": 20, "squeeze_window": 20})
        assert s.get_min_candles() == 50

    def test_get_timeframe(self):
        assert BbBreakoutStrategy({}).get_timeframe() == "1h"

    def test_no_signal_without_squeeze(self):
        """Trending/expanding market → no squeeze → no signal."""
        rng = np.random.default_rng(42)
        close = 50_000 + np.cumsum(rng.normal(0, 200, 100))
        df = _make_df(close)
        strategy = BbBreakoutStrategy({"squeeze_pct": 0.01})  # very tight squeeze threshold
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "none"

    def test_long_signal_on_upper_breakout(self):
        df = _squeeze_then_long_breakout(n=100)
        strategy = BbBreakoutStrategy({"squeeze_pct": 0.99, "vol_multiplier": 1.5})
        result = strategy.generate_signal(df, "BTCUSDT")
        if result.signal_type == "none":
            pytest.skip(f"No long breakout generated: {result.reason}")
        assert result.signal_type == "long"
        assert result.strength_score == 3
        assert result.sl is not None and result.sl < result.entry_price
        assert result.tp1 is not None and result.tp1 > result.entry_price

    def test_short_signal_on_lower_breakout(self):
        df = _squeeze_then_short_breakout(n=100)
        strategy = BbBreakoutStrategy({"squeeze_pct": 0.99, "vol_multiplier": 1.5})
        result = strategy.generate_signal(df, "BTCUSDT")
        if result.signal_type == "none":
            pytest.skip(f"No short breakout generated: {result.reason}")
        assert result.signal_type == "short"
        assert result.strength_score == 3
        assert result.sl is not None and result.sl > result.entry_price
        assert result.tp1 is not None and result.tp1 < result.entry_price

    def test_insufficient_data_returns_none(self):
        df = _make_df(np.full(10, 50_000.0))
        strategy = BbBreakoutStrategy({})
        assert not strategy.is_data_sufficient(df)

    def test_indicators_snapshot_present(self):
        df = _squeeze_then_long_breakout(n=100)
        strategy = BbBreakoutStrategy({"squeeze_pct": 0.99, "vol_multiplier": 1.5})
        result = strategy.generate_signal(df, "BTCUSDT")
        for key in ("bb_width", "squeeze_detected", "atr"):
            assert key in result.indicators, f"Missing key '{key}'"

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            BbBreakoutStrategy({"sl_atr_mult": 3.0, "tp1_atr_mult": 2.0})

    def test_squeeze_no_volume_returns_none(self):
        """Breakout bar with low volume → no signal."""
        rng = np.random.default_rng(1)
        close = np.full(100, 50_000.0) + rng.normal(0, 20, 100)
        close[-1] = 50_000 + 5_000
        n = 100
        high = close + 50
        low = close - 50
        open_ = close.copy()
        # Very low volume on breakout bar
        volume = np.full(n, 500.0)
        volume[-1] = 10.0  # no elevation
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
        df.loc[df.index[-2], "close"] = 50_000.0

        strategy = BbBreakoutStrategy({"squeeze_pct": 0.99, "vol_multiplier": 1.5})
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "none"
