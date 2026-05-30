"""Unit tests for StrategyRunner."""

from __future__ import annotations

import os
import sqlite3
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import pytest

from src.signal.base_strategy import SignalResult
from src.signal.strategy_runner import StrategyRunner, _tp_scale_from_adx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50_000 + np.cumsum(rng.normal(0, 200, n))
    high = close + rng.uniform(50, 300, n)
    low = close - rng.uniform(50, 300, n)
    open_ = close + rng.normal(0, 100, n)
    volume = rng.uniform(100, 1_000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStrategyRunner:
    def test_loads_correct_strategy_by_name(self, monkeypatch):
        """ACTIVE_STRATEGY=rsi_macd → get_active_strategy_name() == 'rsi_macd'."""
        monkeypatch.setenv("ACTIVE_STRATEGY", "rsi_macd")
        monkeypatch.delenv("STRATEGY_PARAMS", raising=False)
        runner = StrategyRunner()
        assert runner.get_active_strategy_name() == "rsi_macd"

    def test_snake_to_class_name_conversion(self):
        """snake_case → PascalCaseStrategy for all three implemented strategies."""
        cases = {
            "rsi_macd": "RsiMacdStrategy",
            "zscore_reversion": "ZscoreReversionStrategy",
            "bb_breakout": "BbBreakoutStrategy",
        }
        for snake, expected in cases.items():
            result = StrategyRunner._snake_to_class_name(snake)
            assert result == expected, f"_snake_to_class_name({snake!r}) = {result!r}, expected {expected!r}"

    def test_invalid_strategy_name_raises_value_error(self, monkeypatch):
        """Unknown strategy name → ValueError on construction."""
        monkeypatch.setenv("ACTIVE_STRATEGY", "nonexistent_strategy")
        monkeypatch.delenv("STRATEGY_PARAMS", raising=False)
        with pytest.raises(ValueError):
            StrategyRunner()

    def test_strategy_exception_returns_none_signal(self, monkeypatch):
        """If generate_signal() raises, run() must return signal_type='none' without propagating."""
        monkeypatch.setenv("ACTIVE_STRATEGY", "rsi_macd")
        monkeypatch.delenv("STRATEGY_PARAMS", raising=False)
        runner = StrategyRunner()

        # Patch the inner strategy's generate_signal to always raise
        def _boom(df, symbol):
            raise RuntimeError("simulated strategy crash")

        monkeypatch.setattr(runner._strategy, "generate_signal", _boom)

        df = _make_df()
        result = runner.run(df, "BTCUSDT")
        assert result.signal_type == "none"
        assert "전략 실행 오류" in result.reason

    def test_runtime_reload(self, monkeypatch):
        """reload() hot-swaps the active strategy without restarting."""
        monkeypatch.setenv("ACTIVE_STRATEGY", "rsi_macd")
        monkeypatch.delenv("STRATEGY_PARAMS", raising=False)
        runner = StrategyRunner()
        assert runner.get_active_strategy_name() == "rsi_macd"

        runner.reload("zscore_reversion", {})
        assert runner.get_active_strategy_name() == "zscore_reversion"


# ---------------------------------------------------------------------------
# G2: Intraday regime refresh
# ---------------------------------------------------------------------------

def _make_conn_with_symbols(symbols: list[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE symbols (
            symbol TEXT PRIMARY KEY,
            is_active INTEGER NOT NULL DEFAULT 1,
            strategy TEXT
        );
    """)
    for sym in symbols:
        conn.execute("INSERT INTO symbols (symbol) VALUES (?)", (sym,))
    conn.commit()
    return conn


class TestRegimeRefresh:
    def _make_runner(self, monkeypatch, conn=None):
        monkeypatch.setenv("ACTIVE_STRATEGY", "rsi_macd")
        monkeypatch.delenv("STRATEGY_PARAMS", raising=False)
        return StrategyRunner(conn=conn)

    def test_no_refresh_before_interval(self, monkeypatch):
        """_regime_refresh_if_needed skips when interval hasn't elapsed."""
        conn = _make_conn_with_symbols(["BTCUSDT"])
        runner = self._make_runner(monkeypatch, conn=conn)
        runner._last_btc_adx = 30.0
        runner._last_regime_check = time.monotonic()  # just checked

        with patch("src.jobs.screener._compute_symbol_indicators") as mock_ind:
            runner._regime_refresh_if_needed()
            mock_ind.assert_not_called()

    def test_no_reassign_when_adx_stable(self, monkeypatch):
        """No reassignment when BTC ADX shift < threshold."""
        conn = _make_conn_with_symbols(["BTCUSDT", "ETHUSDT"])
        runner = self._make_runner(monkeypatch, conn=conn)
        runner._last_btc_adx = 30.0
        runner._last_regime_check = 0.0  # force check

        ind_stable = {"adx": 32.0, "atr_pct": 1.0, "above_sma200": False,
                      "sma_aligned": False, "sma50_slope": 0.0, "adx_change": 0.0}
        with patch("src.jobs.screener._compute_symbol_indicators", return_value=ind_stable):
            with patch("src.jobs.screener._assign_strategy") as mock_assign:
                runner._regime_refresh_if_needed()
                mock_assign.assert_not_called()

    def test_reassigns_when_adx_shifts_significantly(self, monkeypatch):
        """Re-assigns all symbols when BTC ADX shifts > threshold."""
        conn = _make_conn_with_symbols(["BTCUSDT", "ETHUSDT"])
        runner = self._make_runner(monkeypatch, conn=conn)
        runner._last_btc_adx = 20.0
        runner._last_regime_check = 0.0  # force check

        ind_shifted = {"adx": 35.0, "atr_pct": 2.0, "above_sma200": True,
                       "sma_aligned": False, "sma50_slope": 0.0, "adx_change": 0.0}
        with patch("src.jobs.screener._compute_symbol_indicators", return_value=ind_shifted):
            with patch("src.jobs.screener._assign_strategy", return_value="supertrend"):
                with patch("src.jobs.screener._discover_strategies", return_value=[]):
                    runner._regime_refresh_if_needed()

        rows = conn.execute("SELECT strategy FROM symbols WHERE strategy IS NOT NULL").fetchall()
        assert len(rows) == 2
        assert all(r[0] == "supertrend" for r in rows)

    def test_cache_cleared_after_refresh(self, monkeypatch):
        """Symbol strategy cache cleared after regime refresh."""
        conn = _make_conn_with_symbols(["BTCUSDT"])
        runner = self._make_runner(monkeypatch, conn=conn)
        runner._symbol_strategy_cache["BTCUSDT"] = (runner._strategy, time.monotonic())
        runner._last_btc_adx = 20.0
        runner._last_regime_check = 0.0

        ind_shifted = {"adx": 36.0, "atr_pct": 1.5, "above_sma200": False,
                       "sma_aligned": False, "sma50_slope": 0.0, "adx_change": 0.0}
        with patch("src.jobs.screener._compute_symbol_indicators", return_value=ind_shifted):
            with patch("src.jobs.screener._assign_strategy", return_value="ema_crossover"):
                with patch("src.jobs.screener._discover_strategies", return_value=[]):
                    runner._regime_refresh_if_needed()

        assert "BTCUSDT" not in runner._symbol_strategy_cache

    def test_updates_last_btc_adx_after_refresh(self, monkeypatch):
        """_last_btc_adx updated to current ADX after refresh."""
        conn = _make_conn_with_symbols(["BTCUSDT"])
        runner = self._make_runner(monkeypatch, conn=conn)
        runner._last_btc_adx = 20.0
        runner._last_regime_check = 0.0

        ind = {"adx": 38.0, "atr_pct": 1.5, "above_sma200": False,
               "sma_aligned": False, "sma50_slope": 0.0, "adx_change": 0.0}
        with patch("src.jobs.screener._compute_symbol_indicators", return_value=ind):
            with patch("src.jobs.screener._assign_strategy", return_value="supertrend"):
                with patch("src.jobs.screener._discover_strategies", return_value=[]):
                    runner._regime_refresh_if_needed()

        assert runner._last_btc_adx == 38.0

    def test_no_crash_without_db(self, monkeypatch):
        """reload_if_changed() is safe when no DB connection."""
        monkeypatch.setenv("ACTIVE_STRATEGY", "rsi_macd")
        monkeypatch.delenv("STRATEGY_PARAMS", raising=False)
        runner = StrategyRunner(conn=None)
        runner.reload_if_changed()  # must not raise


# ---------------------------------------------------------------------------
# Regime-adaptive TP scaling
# ---------------------------------------------------------------------------

class TestTpScaleFromAdx:
    def test_neutral_at_adx_25(self):
        assert _tp_scale_from_adx(25.0) == pytest.approx(1.0)

    def test_none_returns_one(self):
        assert _tp_scale_from_adx(None) == 1.0

    def test_trending_above_one(self):
        # ADX 45 → 1.0 + (45-25)*0.01 = 1.20
        assert _tp_scale_from_adx(45.0) == pytest.approx(1.20)

    def test_ranging_below_one(self):
        # ADX 15 → 1.0 + (15-25)*0.01 = 0.90
        assert _tp_scale_from_adx(15.0) == pytest.approx(0.90)

    def test_clamped_at_max(self):
        # ADX 200 → clamped to 1.25
        assert _tp_scale_from_adx(200.0) == pytest.approx(1.25)

    def test_clamped_at_min(self):
        # ADX 0 → 1.0 + (0-25)*0.01 = 0.75
        assert _tp_scale_from_adx(0.0) == pytest.approx(0.75)
        # ADX negative → also clamped at 0.75
        assert _tp_scale_from_adx(-50.0) == pytest.approx(0.75)


class TestApplyRegimeTpScale:
    def _make_runner(self, monkeypatch, btc_adx: float | None = None):
        monkeypatch.setenv("ACTIVE_STRATEGY", "ema_crossover")
        monkeypatch.delenv("STRATEGY_PARAMS", raising=False)
        runner = StrategyRunner(conn=None)
        runner._last_btc_adx = btc_adx
        return runner

    def test_long_signal_tp_widens_in_trend(self, monkeypatch):
        runner = self._make_runner(monkeypatch, btc_adx=45.0)  # scale = 1.20
        r = SignalResult(
            signal_type="long", strength_score=2,
            entry_price=100.0, tp1=103.0, tp2=105.0, sl=98.0,
        )
        r2 = runner._apply_regime_tp_scale(r)
        # tp1 dist = 3.0 → 3.0 * 1.20 = 3.6 → 103.6
        assert r2.tp1 == pytest.approx(103.6)
        # tp2 dist = 5.0 → 5.0 * 1.20 = 6.0 → 106.0
        assert r2.tp2 == pytest.approx(106.0)

    def test_short_signal_tp_widens_in_trend(self, monkeypatch):
        runner = self._make_runner(monkeypatch, btc_adx=45.0)  # scale = 1.20
        r = SignalResult(
            signal_type="short", strength_score=2,
            entry_price=100.0, tp1=97.0, tp2=95.0, sl=102.0,
        )
        r2 = runner._apply_regime_tp_scale(r)
        # tp1 dist = 3.0 → 3.6 → 96.4
        assert r2.tp1 == pytest.approx(96.4)
        # tp2 dist = 5.0 → 6.0 → 94.0
        assert r2.tp2 == pytest.approx(94.0)

    def test_ranging_tightens_tp(self, monkeypatch):
        runner = self._make_runner(monkeypatch, btc_adx=15.0)  # scale = 0.90
        r = SignalResult(
            signal_type="long", strength_score=2,
            entry_price=100.0, tp1=110.0, tp2=120.0, sl=95.0,
        )
        r2 = runner._apply_regime_tp_scale(r)
        assert r2.tp1 == pytest.approx(109.0)   # 100 + 10*0.90
        assert r2.tp2 == pytest.approx(118.0)   # 100 + 20*0.90

    def test_no_scale_when_btc_adx_none(self, monkeypatch):
        runner = self._make_runner(monkeypatch, btc_adx=None)
        r = SignalResult(
            signal_type="long", strength_score=2,
            entry_price=100.0, tp1=103.0, tp2=105.0, sl=98.0,
        )
        r2 = runner._apply_regime_tp_scale(r)
        assert r2.tp1 == pytest.approx(103.0)
        assert r2.tp2 == pytest.approx(105.0)

    def test_no_scale_at_neutral_adx(self, monkeypatch):
        runner = self._make_runner(monkeypatch, btc_adx=25.0)  # scale = 1.0
        r = SignalResult(
            signal_type="long", strength_score=2,
            entry_price=100.0, tp1=103.0, tp2=106.0, sl=98.0,
        )
        r2 = runner._apply_regime_tp_scale(r)
        assert r2.tp1 == pytest.approx(103.0)
        assert r2.tp2 == pytest.approx(106.0)

    def test_tp_none_safe(self, monkeypatch):
        runner = self._make_runner(monkeypatch, btc_adx=40.0)
        r = SignalResult(
            signal_type="long", strength_score=2,
            entry_price=100.0, tp1=None, tp2=None, sl=98.0,
        )
        r2 = runner._apply_regime_tp_scale(r)
        assert r2.tp1 is None
        assert r2.tp2 is None


# ---------------------------------------------------------------------------
# Directional concentration guard
# ---------------------------------------------------------------------------

def _make_conn_with_positions(open_positions: list[tuple[str, str]]) -> sqlite3.Connection:
    """Create in-memory DB with positions table. open_positions: [(symbol, side), ...]"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            side TEXT,
            status TEXT DEFAULT 'open'
        );
    """)
    for sym, side in open_positions:
        conn.execute("INSERT INTO positions (symbol, side, status) VALUES (?, ?, 'open')", (sym, side))
    conn.commit()
    return conn


class TestDirectionalConcentration:
    def _make_runner(self, monkeypatch, conn, btc_adx=30.0):
        monkeypatch.setenv("ACTIVE_STRATEGY", "rsi_macd")
        monkeypatch.delenv("STRATEGY_PARAMS", raising=False)
        runner = StrategyRunner(conn=conn)
        runner._last_btc_adx = btc_adx
        return runner

    def test_blocks_at_threshold(self, monkeypatch):
        """4 open longs, limit=5 → max_same=4 → block new long."""
        conn = _make_conn_with_positions([(f"COIN{i}USDT", "long") for i in range(4)])
        runner = self._make_runner(monkeypatch, conn, btc_adx=30.0)
        with patch("src.utils.config.load_config") as mock_cfg:
            mock_cfg.return_value.max_positions = 5
            blocked, reason = runner._check_directional_concentration("long")
        assert blocked
        assert "directional_concentration" in reason
        assert "long" in reason

    def test_allows_opposite_direction(self, monkeypatch):
        """4 open longs → short still allowed."""
        conn = _make_conn_with_positions([(f"COIN{i}USDT", "long") for i in range(4)])
        runner = self._make_runner(monkeypatch, conn, btc_adx=30.0)
        with patch("src.utils.config.load_config") as mock_cfg:
            mock_cfg.return_value.max_positions = 5
            blocked, _ = runner._check_directional_concentration("short")
        assert not blocked

    def test_allows_below_threshold(self, monkeypatch):
        """3 open longs, limit=5, max_same=4 → allow new long."""
        conn = _make_conn_with_positions([(f"COIN{i}USDT", "long") for i in range(3)])
        runner = self._make_runner(monkeypatch, conn, btc_adx=30.0)
        with patch("src.utils.config.load_config") as mock_cfg:
            mock_cfg.return_value.max_positions = 5
            blocked, _ = runner._check_directional_concentration("long")
        assert not blocked

    def test_ranging_lowers_threshold(self, monkeypatch):
        """ADX=15 → limit=3, max_same=2. 2 longs → block."""
        conn = _make_conn_with_positions([("COIN1USDT", "long"), ("COIN2USDT", "long")])
        runner = self._make_runner(monkeypatch, conn, btc_adx=15.0)
        with patch("src.utils.config.load_config") as mock_cfg:
            mock_cfg.return_value.max_positions = 5
            blocked, reason = runner._check_directional_concentration("long")
        assert blocked
        assert "2" in reason

    def test_no_conn_returns_false(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_STRATEGY", "rsi_macd")
        monkeypatch.delenv("STRATEGY_PARAMS", raising=False)
        runner = StrategyRunner(conn=None)
        blocked, _ = runner._check_directional_concentration("long")
        assert not blocked

    def test_adx_info_in_reason(self, monkeypatch):
        """BTC ADX appears in block reason for observability."""
        conn = _make_conn_with_positions([(f"COIN{i}USDT", "short") for i in range(4)])
        runner = self._make_runner(monkeypatch, conn, btc_adx=35.0)
        with patch("src.utils.config.load_config") as mock_cfg:
            mock_cfg.return_value.max_positions = 5
            blocked, reason = runner._check_directional_concentration("short")
        assert blocked
        assert "35.0" in reason
