"""Unit tests for walk-forward optimization harness."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import WFPeriod, WFResult, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 50_000 + np.cumsum(rng.normal(0, 200, n))
    high  = close + rng.uniform(50, 300, n)
    low   = close - rng.uniform(50, 300, n)
    open_ = close + rng.normal(0, 100, n)
    ts    = [1_700_000_000_000 + i * 3_600_000 for i in range(n)]  # hourly ms
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_, "high": high, "low": low,
        "close": close, "volume": rng.uniform(100, 1_000, n),
    })


def _make_grid_result(pf: float, n: int = 35, wr: float = 0.55, **params) -> dict:
    return {"pf": pf, "n": n, "wr": wr, "net_pnl": 10.0, "max_dd": 5.0,
            "avg_bars": 20.0, "final_bal": 110.0, **params}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWFResult:
    def test_empty_result_metrics(self):
        r = WFResult(strategy="ema_crossover", symbol="BTCUSDT")
        assert r.n_periods == 0
        assert r.avg_wf_pf == 0.0
        assert r.avg_fixed_pf == 0.0
        assert r.wf_win_rate == 0.0
        assert r.pf_edge == 0.0

    def test_wf_beats_fixed(self):
        p = WFPeriod(
            period_idx=0, train_start_ms=0, train_end_ms=1,
            test_start_ms=2, test_end_ms=3,
            best_params={"adx_threshold": 25.0, "sl_atr_mult": 2.0,
                         "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0},
            train_pf=2.0, train_n=40, train_wr=0.6,
            wf_pf=1.8, wf_n=12, wf_wr=0.58,
            fixed_pf=1.4, fixed_n=10, fixed_wr=0.5,
        )
        assert p.wf_beats_fixed is True

    def test_avg_pf_computed_correctly(self):
        periods = [
            WFPeriod(0, 0, 1, 2, 3, {}, 2.0, 40, 0.6, 1.8, 12, 0.55, 1.4, 11, 0.50),
            WFPeriod(1, 4, 5, 6, 7, {}, 1.9, 38, 0.5, 1.5, 10, 0.50, 1.6, 10, 0.52),
        ]
        r = WFResult(strategy="s", symbol="BTC", periods=periods)
        assert r.avg_wf_pf == pytest.approx((1.8 + 1.5) / 2)
        assert r.avg_fixed_pf == pytest.approx((1.4 + 1.6) / 2)
        assert r.pf_edge == pytest.approx(r.avg_wf_pf - r.avg_fixed_pf)

    def test_wf_win_rate(self):
        periods = [
            WFPeriod(0, 0, 1, 2, 3, {}, 2.0, 40, 0.6, 1.8, 10, 0.5, 1.4, 10, 0.5),  # WF wins
            WFPeriod(1, 4, 5, 6, 7, {}, 2.0, 38, 0.5, 1.2, 10, 0.4, 1.5, 10, 0.5),  # WF loses
            WFPeriod(2, 8, 9, 10, 11, {}, 2.0, 35, 0.6, 1.9, 10, 0.6, 1.7, 10, 0.5), # WF wins
        ]
        r = WFResult(strategy="s", symbol="BTC", periods=periods)
        assert r.wf_win_rate == pytest.approx(2 / 3)


class TestRunFunction:
    STRATEGY = "ema_crossover"
    PARAM_KEYS = ["adx_threshold", "sl_atr_mult", "tp1_atr_mult", "tp2_atr_mult"]

    def _train_result(self, **params):
        defaults = {"adx_threshold": 30.0, "sl_atr_mult": 1.5,
                    "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0}
        defaults.update(params)
        return [_make_grid_result(pf=1.85, n=40, **defaults)]

    def _test_results(self):
        """Simulate run_grid returning all combos for the test window."""
        # Include both optimized params and fixed params
        return [
            _make_grid_result(pf=1.60, n=12,
                              adx_threshold=30.0, sl_atr_mult=1.5,
                              tp1_atr_mult=3.0, tp2_atr_mult=5.0),
            # fixed defaults
            _make_grid_result(pf=1.30, n=10,
                              adx_threshold=25.0, sl_atr_mult=2.0,
                              tp1_atr_mult=3.0, tp2_atr_mult=5.0),
        ]

    def test_insufficient_data_returns_empty(self):
        small_df = _make_df(n=100)
        with patch("src.backtest.walk_forward._load_ohlcv", return_value=small_df):
            result = run(self.STRATEGY, "BTCUSDT", train_bars=2160, test_bars=720)
        assert result.n_periods == 0

    def test_correct_number_of_periods(self):
        # df = 3600 bars; train=2160, test=720, step=720
        # t=2160: test ends at 2880 ≤ 3600 → period 1
        # t=2880: test ends at 3600 ≤ 3600 → period 2
        # t=3600: 3600+720=4320 > 3600 → stop
        df = _make_df(n=3600)
        with patch("src.backtest.walk_forward._load_ohlcv", return_value=df), \
             patch("src.backtest.walk_forward.run_grid", side_effect=[
                 self._train_result(), self._test_results(),  # period 1
                 self._train_result(), self._test_results(),  # period 2
             ]):
            result = run(self.STRATEGY, "BTCUSDT",
                         train_bars=2160, test_bars=720, step_bars=720)
        assert result.n_periods == 2

    def test_wf_pf_extracted_from_matching_params(self):
        df = _make_df(n=3000)
        with patch("src.backtest.walk_forward._load_ohlcv", return_value=df), \
             patch("src.backtest.walk_forward.run_grid", side_effect=[
                 self._train_result(),   # train grid → best params: adx=30 sl=1.5 tp1=3.0 tp2=5.0
                 self._test_results(),   # test grid → all combos
             ]):
            result = run(self.STRATEGY, "BTCUSDT",
                         train_bars=2160, test_bars=720, step_bars=720)

        assert result.n_periods == 1
        p = result.periods[0]
        assert p.wf_pf == pytest.approx(1.60)   # matched by best_params
        assert p.fixed_pf == pytest.approx(1.30) # matched by _STRATEGY_DEFAULTS

    def test_strategy_and_symbol_stored(self):
        df = _make_df(n=3000)
        with patch("src.backtest.walk_forward._load_ohlcv", return_value=df), \
             patch("src.backtest.walk_forward.run_grid", side_effect=[
                 self._train_result(), self._test_results(),
             ]):
            result = run(self.STRATEGY, "ETHUSDT",
                         train_bars=2160, test_bars=720, step_bars=720)
        assert result.strategy == self.STRATEGY
        assert result.symbol == "ETHUSDT"

    def test_zero_periods_when_train_consumes_all_data(self):
        df = _make_df(n=2500)  # train=2160, test=720 → needs 2880 → 0 periods
        with patch("src.backtest.walk_forward._load_ohlcv", return_value=df):
            result = run(self.STRATEGY, "BTCUSDT",
                         train_bars=2160, test_bars=720, step_bars=720)
        assert result.n_periods == 0

    def test_missing_train_results_skips_period(self):
        # run_grid returns [] for train → period should be skipped
        df = _make_df(n=3000)
        with patch("src.backtest.walk_forward._load_ohlcv", return_value=df), \
             patch("src.backtest.walk_forward.run_grid", return_value=[]):
            result = run(self.STRATEGY, "BTCUSDT",
                         train_bars=2160, test_bars=720, step_bars=720)
        assert result.n_periods == 0
