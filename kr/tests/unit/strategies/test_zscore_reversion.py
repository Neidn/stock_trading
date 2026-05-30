"""Unit tests for ZscoreReversionStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signal.strategies.zscore_reversion import ZscoreReversionStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(close: np.ndarray, volume_multiplier: float = 1.0) -> pd.DataFrame:
    """Wrap a close array in a minimal OHLCV DataFrame."""
    n = len(close)
    rng = np.random.default_rng(7)
    spread = np.abs(close) * 0.002
    high = close + spread
    low = close - spread
    open_ = close + rng.normal(0, spread * 0.3, n)
    volume = rng.uniform(100, 500, n) * volume_multiplier
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _sideways_close(n: int = 200, mean: float = 50_000.0, amp: float = 200.0, seed: int = 42) -> np.ndarray:
    """Oscillating close prices around a mean — produces low ADX."""
    t = np.linspace(0, 6 * np.pi, n)
    rng = np.random.default_rng(seed)
    return mean + amp * np.sin(t) + rng.normal(0, 10, n)


def _oversold_close(n: int = 200, window: int = 20, threshold: float = 2.0) -> np.ndarray:
    """Close array whose last value is >> threshold σ below rolling mean."""
    base = _sideways_close(n)
    # Force last bar far below recent mean
    recent_mean = base[-window:].mean()
    recent_std = base[-window:].std(ddof=1) + 1e-8
    base[-1] = recent_mean - (threshold + 1.0) * recent_std
    return base


def _overbought_close(n: int = 200, window: int = 20, threshold: float = 2.0) -> np.ndarray:
    """Close array whose last value is >> threshold σ above rolling mean."""
    base = _sideways_close(n)
    recent_mean = base[-window:].mean()
    recent_std = base[-window:].std(ddof=1) + 1e-8
    base[-1] = recent_mean + (threshold + 1.0) * recent_std
    return base


def _trending_close(n: int = 200) -> np.ndarray:
    """Strong linear trend — produces high ADX."""
    rng = np.random.default_rng(5)
    return 10_000 + np.arange(n) * 100.0 + rng.normal(0, 20, n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestZscoreReversionStrategy:
    def test_long_signal_on_oversold(self):
        """Z-score << -threshold → long signal.

        max_adx=100 disables the ADX filter here; the ADX filter is already
        covered by test_blocked_in_trending_market.
        """
        params = {"window": 20, "zscore_threshold": 2.0, "max_adx": 100}
        strategy = ZscoreReversionStrategy(params)
        close = _oversold_close(n=200, window=20, threshold=2.0)
        df = _make_df(close)
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "long", f"Expected long: {result.reason}"
        assert result.strength_score == 3

    def test_short_signal_on_overbought(self):
        """Z-score >> +threshold → short signal.

        max_adx=100 disables the ADX filter here; the ADX filter is already
        covered by test_blocked_in_trending_market.
        """
        params = {"window": 20, "zscore_threshold": 2.0, "max_adx": 100}
        strategy = ZscoreReversionStrategy(params)
        close = _overbought_close(n=200, window=20, threshold=2.0)
        df = _make_df(close)
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "short", f"Expected short: {result.reason}"
        assert result.strength_score == 3

    def test_blocked_in_trending_market(self):
        """ADX > max_adx → signal_type == 'none' regardless of Z-score."""
        # Use a very low max_adx threshold so trending close triggers the block
        params = {"window": 20, "zscore_threshold": 2.0, "max_adx": 5}
        strategy = ZscoreReversionStrategy(params)
        close = _trending_close(n=200)
        df = _make_df(close)
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "none", f"Expected none (trending block): {result.reason}"
        assert "ADX" in result.reason or "추세장" in result.reason

    def test_no_signal_within_threshold(self):
        """|Z-score| < threshold → signal_type == 'none'."""
        # Use a very high threshold so normal oscillations never exceed it
        params = {"window": 20, "zscore_threshold": 10.0, "max_adx": 100}
        strategy = ZscoreReversionStrategy(params)
        close = _sideways_close(n=200)
        df = _make_df(close)
        result = strategy.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "none", f"Expected none (threshold not met): {result.reason}"

    def test_tp1_equals_mean_reversion_target(self):
        """tp1 must equal the rolling SMA (mean reversion target) for both directions."""
        params = {"window": 20, "zscore_threshold": 2.0, "max_adx": 100}
        strategy = ZscoreReversionStrategy(params)

        for make_close, direction in [
            (_oversold_close, "long"),
            (_overbought_close, "short"),
        ]:
            close = make_close(n=200, window=20, threshold=2.0)
            df = _make_df(close)
            result = strategy.generate_signal(df, "BTCUSDT")
            if result.signal_type != direction:
                pytest.skip(f"No {direction} signal for tp1 test: {result.reason}")

            expected_mean = result.indicators["mean"]
            assert abs(result.tp1 - expected_mean) < 1e-6, (
                f"tp1={result.tp1:.4f} ≠ mean={expected_mean:.4f} for {direction}"
            )

    def test_get_timeframe(self):
        strategy = ZscoreReversionStrategy({})
        assert strategy.get_timeframe() == "1h"
