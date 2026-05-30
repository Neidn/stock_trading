"""Bollinger + RSI Double Strategy (ChartArt v1.1).

Pine Script original: https://kr.tradingview.com/v/uCV8I4xA/
Author: ChartArt (January 18, 2015)

Entry logic (ported from Pine v2):
    LONG  — RSI crosses above 50  AND  close crosses above lower BB
    SHORT — RSI crosses below 50  AND  close crosses below upper BB

Both conditions must fire on the same bar (same as Pine crossover simultaneous check).
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_atr, calc_bollinger, calc_rsi


class BbRsiChartartStrategy(BaseStrategy):
    """Bollinger + RSI Double Strategy by ChartArt v1.1.

    Parameters (read from self.params with defaults):
        rsi_period (int):   RSI lookback. Default 6 (original).
        rsi_level (float):  Crossover/under threshold. Default 50 (original).
        bb_period (int):    Bollinger SMA period. Default 200 (original).
        bb_std (float):     Bollinger std-dev multiplier. Default 2.0 (original).
        sl_atr_mult (float):  ATR multiplier for stop-loss. Default 2.0.
        tp1_atr_mult (float): ATR multiplier for TP1. Default 3.0.
        tp2_atr_mult (float): ATR multiplier for TP2. Default 5.0.
    """

    DEFAULTS: dict = {
        "rsi_period":    6,
        "rsi_level":     50.0,
        "bb_period":     200,
        "bb_std":        2.0,
        "sl_atr_mult":   2.0,
        "tp1_atr_mult":  3.0,
        "tp2_atr_mult":  5.0,
    }

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        return frozenset({"ranging"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        """Best in ranging/low-trend markets. Score falls as ADX rises.

        Flat SMA50 (slope ≈ 0) confirms true ranging; aligned SMAs signal
        a structured trend where mean-reversion trades get stopped out.
        """
        adx         = float(indicators.get("adx", 25))
        sma_aligned = bool(indicators.get("sma_aligned", False))
        sma50_slope = float(indicators.get("sma50_slope", 0.0))
        base          = max(0.0, 1.0 - adx / 50.0)
        align_penalty = -0.10 if sma_aligned else 0.05
        ranging_bonus = 0.08 if abs(sma50_slope) < 0.002 else 0.0
        return round(max(0.0, min(base + align_penalty + ranging_bonus, 1.0)), 4)

    def get_name(self) -> str:
        return "bb_rsi_chartart"

    def get_min_candles(self) -> int:
        # BB period dominates; add headroom for RSI warmup
        return self.params.get("bb_period", 200) + 10

    def get_timeframe(self) -> str:
        return "1h"

    def _validate_params(self) -> None:
        sl = self.params.get("sl_atr_mult", 2.0)
        tp1 = self.params.get("tp1_atr_mult", 3.0)
        if sl <= 0:
            raise ValueError(f"sl_atr_mult({sl}) must be > 0")
        if tp1 <= sl:
            raise ValueError(f"tp1_atr_mult({tp1}) must be > sl_atr_mult({sl})")

    def generate_signal(self, df, symbol: str) -> SignalResult:
        """Generate long/short signal from BB + RSI crossover confluence.

        Args:
            df: OHLCV DataFrame sorted ascending.
            symbol: Trading pair identifier.

        Returns:
            SignalResult with strength_score=2 when both crossovers align,
            otherwise signal_type='none'.
        """
        rsi_period   = self.params.get("rsi_period", 6)
        rsi_level    = float(self.params.get("rsi_level", 50))
        bb_period    = self.params.get("bb_period", 200)
        bb_std       = float(self.params.get("bb_std", 2.0))
        sl_atr_mult  = float(self.params.get("sl_atr_mult", 2.0))
        tp1_atr_mult = float(self.params.get("tp1_atr_mult", 3.0))
        tp2_atr_mult = float(self.params.get("tp2_atr_mult", 5.0))

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values

        rsi               = calc_rsi(close, rsi_period)
        bb_upper, _, bb_lower = calc_bollinger(close, bb_period, bb_std)
        atr               = calc_atr(high, low, close, 14)

        # Need at least 2 bars of valid indicator values
        if any(np.isnan(v) for v in [rsi[-1], rsi[-2], bb_upper[-1], bb_lower[-1],
                                      bb_upper[-2], bb_lower[-2], atr[-1]]):
            return SignalResult(signal_type="none", reason="지표 워밍업 중")

        cur_close  = float(close[-1])
        prev_close = float(close[-2])
        cur_rsi    = float(rsi[-1])
        prev_rsi   = float(rsi[-2])
        cur_upper  = float(bb_upper[-1])
        cur_lower  = float(bb_lower[-1])
        prev_upper = float(bb_upper[-2])
        prev_lower = float(bb_lower[-2])
        cur_atr    = float(atr[-1])

        # Pine crossover(a, b)  = prev_a < b  AND cur_a >= b
        # Pine crossunder(a, b) = prev_a > b  AND cur_a <= b
        rsi_cross_up    = prev_rsi   <  rsi_level and cur_rsi   >= rsi_level
        rsi_cross_down  = prev_rsi   >  rsi_level and cur_rsi   <= rsi_level
        price_cross_up  = prev_close <  prev_lower and cur_close >= cur_lower
        price_cross_dn  = prev_close >  prev_upper and cur_close <= cur_upper

        indicators = {
            "rsi": cur_rsi,
            "bb_upper": cur_upper,
            "bb_lower": cur_lower,
            "atr": cur_atr,
        }

        # --- LONG: both crossovers up ---
        if rsi_cross_up and price_cross_up:
            entry = cur_close
            sl    = entry - cur_atr * sl_atr_mult
            if sl <= 0:
                return SignalResult(
                    signal_type="hold", strength_score=0,
                    reason=f"ATR({cur_atr:.6f}) exceeds entry({entry:.6f}); SL would be negative",
                )
            tp1   = entry + cur_atr * tp1_atr_mult
            tp2   = entry + cur_atr * tp2_atr_mult
            return SignalResult(
                signal_type="long",
                strength_score=2,
                entry_price=entry,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                indicators=indicators,
                reason=(
                    f"롱: RSI {prev_rsi:.1f}→{cur_rsi:.1f} crossed {rsi_level} | "
                    f"close {prev_close:.4f}→{cur_close:.4f} crossed lower BB {cur_lower:.4f}"
                ),
            )

        # --- SHORT: both crossovers down ---
        if rsi_cross_down and price_cross_dn:
            entry = cur_close
            sl    = entry + cur_atr * sl_atr_mult
            tp1   = entry - cur_atr * tp1_atr_mult
            tp2   = entry - cur_atr * tp2_atr_mult
            return SignalResult(
                signal_type="short",
                strength_score=2,
                entry_price=entry,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                indicators=indicators,
                reason=(
                    f"숏: RSI {prev_rsi:.1f}→{cur_rsi:.1f} crossed {rsi_level} | "
                    f"close {prev_close:.4f}→{cur_close:.4f} crossed upper BB {cur_upper:.4f}"
                ),
            )

        # --- No signal ---
        rsi_dir = "↑" if cur_rsi > rsi_level else "↓"
        return SignalResult(
            signal_type="none",
            strength_score=0,
            indicators=indicators,
            reason=(
                f"조건 미충족 | RSI={cur_rsi:.1f}{rsi_dir} "
                f"close={cur_close:.4f} BB[{cur_lower:.4f},{cur_upper:.4f}]"
            ),
        )
