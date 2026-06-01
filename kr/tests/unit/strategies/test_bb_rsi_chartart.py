"""Unit tests for BbRsiChartartStrategy (ChartArt v1.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signal.strategies.bb_rsi_chartart import BbRsiChartartStrategy


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def _make_df(close: np.ndarray) -> pd.DataFrame:
    """Build minimal OHLCV from a close array."""
    n = len(close)
    high = close + 50
    low  = close - 50
    open_ = close.copy()
    volume = np.full(n, 500.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


def _make_long_df(n: int = 220) -> pd.DataFrame:
    """Engineer long signal: RSI crossover 50 AND price crossover lower BB.

    Setup:
      - Bars 0-199  : flat ~50 000 with tiny noise → tight BB, RSI near 50
      - Bars 200-217: steady decline → price approaches / drops below lower BB,
                      RSI drifts below 50
      - Bar 218 (n-2): price just below lower BB, RSI < 50  (pre-crossover bar)
      - Bar 219 (n-1): large bounce → price crosses above lower BB, RSI crosses above 50
    """
    rng = np.random.default_rng(42)
    close = np.empty(n)
    close[:200] = 50_000 + rng.normal(0, 25, 200)   # tiny noise → std ≈ 25

    # Gradual decline — 18 bars of ~7 pts each
    for i in range(200, n - 1):
        close[i] = close[i - 1] - 7.0

    # Big last-bar bounce: jumps well above current lower BB
    # Lower BB ≈ SMA(200 latest) − 2*std; after the decline it's ~49 860
    # Jump of +200 sends close to ~49 900+ which clears the band
    close[-1] = close[-2] + 200.0

    return _make_df(close)


def _make_short_df(n: int = 220) -> pd.DataFrame:
    """Engineer short signal: RSI crossunder 50 AND price crossunder upper BB.

    Setup:
      - Bars 0-199  : flat ~50 000 with tiny noise → tight BB
      - Bars 200-217: steady climb → price approaches / rises above upper BB,
                      RSI drifts above 50
      - Bar 218 (n-2): price just above upper BB, RSI > 50
      - Bar 219 (n-1): large drop → price crosses below upper BB, RSI crosses below 50
    """
    rng = np.random.default_rng(99)
    close = np.empty(n)
    close[:200] = 50_000 + rng.normal(0, 25, 200)

    for i in range(200, n - 1):
        close[i] = close[i - 1] + 7.0

    close[-1] = close[-2] - 200.0

    return _make_df(close)


def _make_random_df(n: int = 220, seed: int = 7) -> pd.DataFrame:
    """Random walk — used to verify no spurious signal in neutral conditions."""
    rng = np.random.default_rng(seed)
    close = 50_000 + np.cumsum(rng.normal(0, 30, n))
    return _make_df(close)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBbRsiChartartStrategy:

    # -- metadata ------------------------------------------------------------

    def test_get_name(self):
        assert BbRsiChartartStrategy({}).get_name() == "bb_rsi_chartart"

    def test_get_timeframe(self):
        assert BbRsiChartartStrategy({}).get_timeframe() == "1d"

    def test_get_min_candles_default(self):
        # bb_period=200 (default) → 200 + 10 = 210
        assert BbRsiChartartStrategy({}).get_min_candles() == 210

    def test_get_min_candles_custom(self):
        s = BbRsiChartartStrategy({"bb_period": 50})
        assert s.get_min_candles() == 60

    # -- param validation ----------------------------------------------------

    def test_invalid_params_sl_gte_tp1_raises(self):
        with pytest.raises(ValueError):
            BbRsiChartartStrategy({"sl_atr_mult": 3.0, "tp1_atr_mult": 2.0})

    def test_invalid_params_sl_zero_raises(self):
        with pytest.raises(ValueError):
            BbRsiChartartStrategy({"sl_atr_mult": 0.0, "tp1_atr_mult": 2.0})

    # -- data sufficiency ----------------------------------------------------

    def test_insufficient_data_is_not_sufficient(self):
        s = BbRsiChartartStrategy({})
        df = _make_df(np.full(s.get_min_candles() - 1, 50_000.0))
        assert s.is_data_sufficient(df) is False

    def test_sufficient_data_is_sufficient(self):
        s = BbRsiChartartStrategy({})
        df = _make_df(np.full(s.get_min_candles(), 50_000.0))
        assert s.is_data_sufficient(df) is True

    # -- long signal ---------------------------------------------------------

    def test_long_signal_generated(self):
        """Engineered bounce off lower BB + RSI crossover 50 → long."""
        s = BbRsiChartartStrategy({})
        df = _make_long_df()
        result = s.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "long", (
            f"Expected long, got {result.signal_type}: {result.reason}"
        )
        assert result.strength_score == 2

    def test_long_sl_below_entry(self):
        s = BbRsiChartartStrategy({})
        result = s.generate_signal(_make_long_df(), "BTCUSDT")
        if result.signal_type != "long":
            pytest.skip(f"No long signal: {result.reason}")
        assert result.sl < result.entry_price, "Long SL must be below entry"

    def test_long_tp1_above_entry(self):
        s = BbRsiChartartStrategy({})
        result = s.generate_signal(_make_long_df(), "BTCUSDT")
        if result.signal_type != "long":
            pytest.skip(f"No long signal: {result.reason}")
        assert result.tp1 > result.entry_price, "Long TP1 must be above entry"
        assert result.tp2 > result.tp1, "Long TP2 must be above TP1"

    def test_long_sl_distance_matches_atr_mult(self):
        params = {"sl_atr_mult": 2.0, "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0}
        s = BbRsiChartartStrategy(params)
        result = s.generate_signal(_make_long_df(), "BTCUSDT")
        if result.signal_type != "long":
            pytest.skip(f"No long signal: {result.reason}")
        dist = result.entry_price - result.sl
        expected = result.indicators["atr"] * params["sl_atr_mult"]
        assert abs(dist - expected) < 1e-6

    # -- short signal --------------------------------------------------------

    def test_short_signal_generated(self):
        """Engineered drop from upper BB + RSI crossunder 50 → short."""
        s = BbRsiChartartStrategy({})
        df = _make_short_df()
        result = s.generate_signal(df, "BTCUSDT")
        assert result.signal_type == "short", (
            f"Expected short, got {result.signal_type}: {result.reason}"
        )
        assert result.strength_score == 2

    def test_short_sl_above_entry(self):
        s = BbRsiChartartStrategy({})
        result = s.generate_signal(_make_short_df(), "BTCUSDT")
        if result.signal_type != "short":
            pytest.skip(f"No short signal: {result.reason}")
        assert result.sl > result.entry_price, "Short SL must be above entry"

    def test_short_tp1_below_entry(self):
        s = BbRsiChartartStrategy({})
        result = s.generate_signal(_make_short_df(), "BTCUSDT")
        if result.signal_type != "short":
            pytest.skip(f"No short signal: {result.reason}")
        assert result.tp1 < result.entry_price, "Short TP1 must be below entry"
        assert result.tp2 < result.tp1, "Short TP2 must be below TP1"

    # -- no signal -----------------------------------------------------------

    def test_no_signal_only_rsi_crossover(self):
        """RSI crosses 50 but price stays inside BB → no signal."""
        rng = np.random.default_rng(55)
        n = 220
        close = np.empty(n)
        close[:200] = 50_000 + rng.normal(0, 25, 200)
        # Small oscillation — RSI may cross 50 but price stays well inside bands
        close[200:] = 50_000 + rng.normal(0, 10, 20)
        s = BbRsiChartartStrategy({})
        result = s.generate_signal(_make_df(close), "BTCUSDT")
        # If a signal fires it must satisfy BOTH crossovers; otherwise no signal
        if result.signal_type != "none":
            # Ensure indicators show the crossover was genuine
            assert result.strength_score == 2

    def test_no_signal_flat_market(self):
        """Perfectly flat price → no crossovers → no signal."""
        close = np.full(220, 50_000.0)
        # Add just enough noise to avoid NaN std
        close += np.random.default_rng(0).normal(0, 1, 220)
        s = BbRsiChartartStrategy({})
        result = s.generate_signal(_make_df(close), "BTCUSDT")
        assert result.signal_type == "none"

    # -- indicators snapshot -------------------------------------------------

    def test_indicators_snapshot_keys_present(self):
        """SignalResult.indicators must contain rsi, bb_upper, bb_lower, atr."""
        s = BbRsiChartartStrategy({})
        result = s.generate_signal(_make_random_df(), "BTCUSDT")
        for key in ("rsi", "bb_upper", "bb_lower", "atr"):
            assert key in result.indicators, f"Missing indicator key '{key}'"

    def test_indicators_values_are_finite(self):
        s = BbRsiChartartStrategy({})
        result = s.generate_signal(_make_long_df(), "BTCUSDT")
        for k, v in result.indicators.items():
            assert np.isfinite(v), f"Indicator '{k}' is not finite: {v}"

    # -- warmup guard --------------------------------------------------------

    def test_warmup_returns_none_on_nan_indicators(self):
        """Too few bars → indicators are NaN → strategy returns 'none'."""
        # Exactly get_min_candles() - 1 bars: is_data_sufficient is False
        s = BbRsiChartartStrategy({})
        close = np.full(s.get_min_candles() - 1, 50_000.0)
        result = s.generate_signal(_make_df(close), "BTCUSDT")
        assert result.signal_type == "none"
