"""Unit tests for position_sizer — atr_position_size and kelly_risk_pct."""

from __future__ import annotations

import sqlite3

import pytest

from src.risk.position_sizer import atr_position_size, calc_position_margin, kelly_risk_pct


# ---------------------------------------------------------------------------
# atr_position_size
# ---------------------------------------------------------------------------

class TestAtrPositionSize:
    def test_basic_calculation(self):
        qty = atr_position_size(account_balance=10_000, risk_pct=0.01, atr=500.0)
        assert qty == pytest.approx(10_000 * 0.01 / (500.0 * 2.0))

    def test_atr_zero_raises(self):
        with pytest.raises(ValueError, match="atr must be positive"):
            atr_position_size(10_000, 0.01, atr=0.0)

    def test_atr_negative_raises(self):
        with pytest.raises(ValueError, match="atr must be positive"):
            atr_position_size(10_000, 0.01, atr=-100.0)

    def test_leverage_param_does_not_change_qty(self):
        qty1 = atr_position_size(10_000, 0.01, atr=500.0, leverage=1)
        qty3 = atr_position_size(10_000, 0.01, atr=500.0, leverage=3)
        assert qty1 == pytest.approx(qty3)


# ---------------------------------------------------------------------------
# kelly_risk_pct helpers
# ---------------------------------------------------------------------------

def _make_conn(rows: list[tuple[str, str, str | None]]) -> sqlite3.Connection:
    """rows = [(strategy_name, status, realized_pnl), ...]"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE positions (
            strategy_name TEXT,
            status TEXT,
            realized_pnl TEXT,
            closed_at TEXT DEFAULT '2026-01-01T00:00:00'
        );
    """)
    conn.executemany(
        "INSERT INTO positions (strategy_name, status, realized_pnl) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    return conn


def _closed(strategy: str, pnl: float) -> tuple[str, str, str]:
    return (strategy, "closed", str(pnl))


# ---------------------------------------------------------------------------
# kelly_risk_pct tests
# ---------------------------------------------------------------------------

class TestKellyRiskPct:
    FALLBACK = 0.005

    def test_fewer_than_min_trades_returns_fallback(self):
        rows = [_closed("ema_crossover", 10.0) for _ in range(9)]
        conn = _make_conn(rows)
        result = kelly_risk_pct(conn, "ema_crossover", self.FALLBACK, min_trades=10)
        assert result == self.FALLBACK

    def test_exactly_min_trades_uses_kelly(self):
        # 10 trades: 6 wins (+100 each), 4 losses (-50 each)
        rows = [_closed("s", 100.0)] * 6 + [_closed("s", -50.0)] * 4
        conn = _make_conn(rows)
        result = kelly_risk_pct(conn, "s", self.FALLBACK, min_trades=10)
        # W=0.6, R=2.0 → f*=0.6 - 0.4/2 = 0.4, half=0.2, clamped to 0.008
        assert result == pytest.approx(0.008)

    def test_all_wins_returns_fallback(self):
        rows = [_closed("s", 100.0)] * 15
        conn = _make_conn(rows)
        result = kelly_risk_pct(conn, "s", self.FALLBACK, min_trades=10)
        assert result == self.FALLBACK

    def test_all_losses_returns_fallback(self):
        rows = [_closed("s", -100.0)] * 15
        conn = _make_conn(rows)
        result = kelly_risk_pct(conn, "s", self.FALLBACK, min_trades=10)
        assert result == self.FALLBACK

    def test_negative_kelly_returns_fallback(self):
        # W=0.2, R=0.5 → f* = 0.2 - 0.8/0.5 = 0.2 - 1.6 = -1.4 < 0
        rows = [_closed("s", 50.0)] * 4 + [_closed("s", -100.0)] * 16
        conn = _make_conn(rows)
        result = kelly_risk_pct(conn, "s", self.FALLBACK, min_trades=10)
        assert result == self.FALLBACK

    def test_respects_min_rpt_floor(self):
        # W=0.52, R=1.05 → f*=0.52 - 0.48/1.05 ≈ 0.063, half≈0.031, should be >> 0.002
        # Use a case where kelly is tiny: W=0.51, R=1.01 → f*=0.51 - 0.49/1.01 ≈ 0.024, half=0.012
        # Want below min_rpt: force a very marginal edge → use custom min_rpt=0.05
        rows = [_closed("s", 10.1)] * 11 + [_closed("s", -10.0)] * 9
        conn = _make_conn(rows)
        result = kelly_risk_pct(conn, "s", self.FALLBACK, min_trades=10, min_rpt=0.05)
        assert result >= 0.05

    def test_respects_max_rpt_ceiling(self):
        # Strong edge: W=0.8, R=4 → f*=0.8 - 0.2/4=0.75, half=0.375 → clamped to max_rpt
        rows = [_closed("s", 400.0)] * 16 + [_closed("s", -100.0)] * 4
        conn = _make_conn(rows)
        result = kelly_risk_pct(conn, "s", self.FALLBACK, min_trades=10, max_rpt=0.008)
        assert result == pytest.approx(0.008)

    def test_unknown_strategy_returns_fallback(self):
        rows = [_closed("other_strategy", 100.0)] * 15
        conn = _make_conn(rows)
        result = kelly_risk_pct(conn, "ema_crossover", self.FALLBACK, min_trades=10)
        assert result == self.FALLBACK

    def test_bad_conn_returns_fallback(self):
        """DB error (e.g. missing table) must not raise."""
        conn = sqlite3.connect(":memory:")  # no positions table
        result = kelly_risk_pct(conn, "s", self.FALLBACK)
        assert result == self.FALLBACK

    def test_only_open_positions_ignored(self):
        """open positions don't count toward Kelly."""
        rows = [("s", "open", "100.0")] * 20 + [_closed("s", -50.0)] * 5
        conn = _make_conn(rows)
        result = kelly_risk_pct(conn, "s", self.FALLBACK, min_trades=10)
        assert result == self.FALLBACK  # only 5 closed → insufficient

    def test_valid_kelly_within_bounds(self):
        # W=0.6, R=1.5 → f*=0.6 - 0.4/1.5=0.333, half=0.167 → clamped 0.008
        rows = [_closed("s", 150.0)] * 12 + [_closed("s", -100.0)] * 8
        conn = _make_conn(rows)
        result = kelly_risk_pct(conn, "s", self.FALLBACK)
        assert 0.002 <= result <= 0.008
