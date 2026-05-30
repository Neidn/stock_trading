"""MACD + SMA 200 Strategy (by ChartArt).

Pine Script original: https://kr.tradingview.com/v/uCV8I4xA/ (v1.0, 2015-11-30)
Author: ChartArt

NOTE: This MACD uses SMA (not EMA) for fast/slow/signal lines — intentional ChartArt variant.

Entry logic (ported from Pine v2):
    LONG  — hist crosses above 0  AND  macd > 0  AND  fastSMA > slowSMA
            AND  close[slowLength bars ago] > SMA200
    SHORT — hist crosses below 0  AND  macd < 0  AND  fastSMA < slowSMA
            AND  close[slowLength bars ago] < SMA200
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_atr, calc_sma


class MacdSma200ChartartStrategy(BaseStrategy):
    """MACD + SMA200 strategy by ChartArt.

    Parameters (read from self.params with defaults):
        fast_period (int):    SMA fast period. Default 12.
        slow_period (int):    SMA slow period. Default 26.
        signal_period (int):  SMA of MACD period. Default 9.
        sma200_period (int):  Very slow SMA period. Default 200.
        sl_atr_mult (float):  ATR multiplier for stop-loss. Default 2.0.
        tp1_atr_mult (float): ATR multiplier for TP1. Default 3.0.
        tp2_atr_mult (float): ATR multiplier for TP2. Default 5.0.
    """

    DEFAULTS: dict = {
        "fast_period":   12,
        "slow_period":   26,
        "signal_period": 9,
        "sma200_period": 200,
        "sl_atr_mult":   2.0,
        "tp1_atr_mult":  3.0,
        "tp2_atr_mult":  5.0,
    }

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        return frozenset({"trending"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        """Peaks at ADX 55 — very strong sustained trends only.

        SMA200 needs long-duration stable trends to be meaningful.
        ema_pullback_rsi wins in ADX 32-54 range.
        sma_aligned is a strong signal here: SMA200 filter is only valid when
        all timeframes agree on direction.
        """
        adx          = float(indicators.get("adx", 25))
        atr_pct      = float(indicators.get("atr_pct", 1.0))
        sma_aligned  = bool(indicators.get("sma_aligned", False))
        sma50_slope  = float(indicators.get("sma50_slope", 0.0))
        above_sma200 = bool(indicators.get("above_sma200", False))
        # Bell curve peaking at ADX 55: rises linearly below, falls above
        trend_score = adx / 55.0 if adx <= 55.0 else max(0.0, 1.0 - (adx - 55.0) / 55.0)
        vol_score   = min(atr_pct / 5.0, 1.0)
        base         = (trend_score + vol_score) / 2.0
        align_bonus  = 0.18 if sma_aligned else 0.0
        slope_bonus  = min(abs(sma50_slope) / 0.02, 1.0) * 0.07
        above_bonus  = 0.08 if above_sma200 else 0.0
        return round(min(base + align_bonus + slope_bonus + above_bonus, 1.0), 4)

    def get_name(self) -> str:
        return "macd_sma200_chartart"

    def get_min_candles(self) -> int:
        p = {**self.DEFAULTS, **self.params}
        # SMA200 dominates; add signal warmup on top of MACD series
        return int(p["sma200_period"]) + int(p["signal_period"]) + 10

    def get_timeframe(self) -> str:
        return "1h"

    def _validate_params(self) -> None:
        p = {**self.DEFAULTS, **self.params}
        if p["fast_period"] >= p["slow_period"]:
            raise ValueError(
                f"fast_period({p['fast_period']}) must be < slow_period({p['slow_period']})"
            )
        if p["sl_atr_mult"] <= 0:
            raise ValueError(f"sl_atr_mult({p['sl_atr_mult']}) must be > 0")
        if p["tp1_atr_mult"] <= p["sl_atr_mult"]:
            raise ValueError(
                f"tp1_atr_mult({p['tp1_atr_mult']}) must be > sl_atr_mult({p['sl_atr_mult']})"
            )

    def generate_signal(self, df, symbol: str) -> SignalResult:
        p = {**self.DEFAULTS, **self.params}

        fast_p   = int(p["fast_period"])
        slow_p   = int(p["slow_period"])
        sig_p    = int(p["signal_period"])
        sma200_p = int(p["sma200_period"])
        sl_mult  = float(p["sl_atr_mult"])
        tp1_mult = float(p["tp1_atr_mult"])
        tp2_mult = float(p["tp2_atr_mult"])

        close = df["close"].to_numpy(dtype=float)
        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)

        fast_sma   = calc_sma(close, fast_p)
        slow_sma   = calc_sma(close, slow_p)
        very_slow  = calc_sma(close, sma200_p)
        macd_line  = fast_sma - slow_sma
        signal_arr = calc_sma(macd_line, sig_p)
        hist       = macd_line - signal_arr
        atr        = calc_atr(high, low, close)

        # Need at least 2 bars for crossover + slow_p bars back for close[slowLength]
        min_idx = slow_p + 1
        if len(close) < min_idx + 1:
            return SignalResult(signal_type="hold", strength_score=0, reason="insufficient data")

        cur_hist  = hist[-1]
        prev_hist = hist[-2]
        cur_macd  = macd_line[-1]
        cur_fast  = fast_sma[-1]
        cur_slow  = slow_sma[-1]
        cur_vslow = very_slow[-1]
        cur_atr   = atr[-1]
        cur_close = close[-1]

        # close[slowLength] = close slow_p bars ago (Pine: close[26])
        close_lagged = close[-(slow_p + 1)]

        hist_cross_up   = prev_hist <= 0 and cur_hist > 0
        hist_cross_down = prev_hist >= 0 and cur_hist < 0

        indicators = {
            "fast_sma":    cur_fast,
            "slow_sma":    cur_slow,
            "very_slow":   cur_vslow,
            "macd":        cur_macd,
            "hist":        cur_hist,
            "atr":         cur_atr,
        }

        # LONG
        if (hist_cross_up
                and cur_macd > 0
                and cur_fast > cur_slow
                and close_lagged > cur_vslow):
            entry = cur_close
            sl    = entry - cur_atr * sl_mult
            if sl <= 0:
                return SignalResult(
                    signal_type="hold", strength_score=0,
                    reason=f"ATR({cur_atr:.6f}) exceeds entry({entry:.6f}); SL would be negative",
                )
            return SignalResult(
                signal_type="long",
                strength_score=3,
                entry_price=entry,
                sl=sl,
                tp1=entry + cur_atr * tp1_mult,
                tp2=entry + cur_atr * tp2_mult,
                indicators=indicators,
                reason=(
                    f"롱: MACD hist crossed 0↑ macd={cur_macd:.4f} "
                    f"fast({cur_fast:.4f})>slow({cur_slow:.4f}) "
                    f"close[{slow_p}]({close_lagged:.4f})>SMA200({cur_vslow:.4f})"
                ),
            )

        # SHORT
        if (hist_cross_down
                and cur_macd < 0
                and cur_fast < cur_slow
                and close_lagged < cur_vslow):
            entry = cur_close
            sl    = entry + cur_atr * sl_mult
            return SignalResult(
                signal_type="short",
                strength_score=3,
                entry_price=entry,
                sl=sl,
                tp1=entry - cur_atr * tp1_mult,
                tp2=entry - cur_atr * tp2_mult,
                indicators=indicators,
                reason=(
                    f"숏: MACD hist crossed 0↓ macd={cur_macd:.4f} "
                    f"fast({cur_fast:.4f})<slow({cur_slow:.4f}) "
                    f"close[{slow_p}]({close_lagged:.4f})<SMA200({cur_vslow:.4f})"
                ),
            )

        return SignalResult(signal_type="hold", strength_score=0, reason="no signal")
