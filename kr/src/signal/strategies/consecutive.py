"""Consecutive Rising Days Strategy (KRX long-only).

Signal: N consecutive sessions where close > previous close, confirmed by
SMA trend and volume.

Long conditions (strength +1 each):
  1. Last n_days all have close[i] > close[i-1]  — consecutive up closes
  2. close > SMA(sma_period)                      — above trend
  3. volume_ratio >= vol_threshold                — volume confirms

Minimum strength_score 2 required to be actionable.
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_atr, calc_sma, calc_volume_ratio


class ConsecutiveStrategy(BaseStrategy):
    """N consecutive up-closes with SMA trend + volume confirmation.

    Parameters:
        n_days (int):           Required consecutive up-close count. Default 3.
        sma_period (int):       Trend filter SMA period. Default 20.
        vol_threshold (float):  Min volume ratio. Default 1.1.
        sl_atr_mult (float):    ATR multiplier for stop-loss. Default 2.0.
        tp1_atr_mult (float):   ATR multiplier for TP1. Default 3.0.
        tp2_atr_mult (float):   ATR multiplier for TP2. Default 5.0.
    """

    DEFAULTS: dict = {
        "n_days":        3,
        "sma_period":    20,
        "vol_threshold": 1.1,
        "sl_atr_mult":   2.0,
        "tp1_atr_mult":  3.0,
        "tp2_atr_mult":  5.0,
    }

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        return frozenset({"trending"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        adx          = float(indicators.get("adx", 25))
        above_sma200 = bool(indicators.get("above_sma200", False))
        atr_pct      = float(indicators.get("atr_pct", 1.0))
        base         = min(adx / 50.0, 1.0) * 0.50
        align_bonus  = 0.20 if above_sma200 else 0.0
        atr_bonus    = 0.05 if atr_pct <= 3.0 else 0.0
        return round(min(base + align_bonus + atr_bonus, 0.80), 4)

    def get_name(self) -> str:
        return "consecutive"

    def get_min_candles(self) -> int:
        n_days     = int(self.params.get("n_days",     self.DEFAULTS["n_days"]))
        sma_period = int(self.params.get("sma_period", self.DEFAULTS["sma_period"]))
        return max(n_days + 1, sma_period) + 20

    def get_timeframe(self) -> str:
        return "D"

    def generate_signal(self, df, symbol: str) -> SignalResult:
        p = {**self.DEFAULTS, **self.params}
        n_days        = int(p["n_days"])
        sma_period    = int(p["sma_period"])
        vol_threshold = float(p["vol_threshold"])
        sl_atr_mult   = float(p["sl_atr_mult"])
        tp1_atr_mult  = float(p["tp1_atr_mult"])
        tp2_atr_mult  = float(p["tp2_atr_mult"])

        close  = df["close"].to_numpy(dtype=float)
        high   = df["high"].to_numpy(dtype=float)
        low    = df["low"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)

        needed = max(n_days + 1, sma_period) + 1
        if len(close) < needed:
            return SignalResult(signal_type="hold", strength_score=0, reason="insufficient data")

        window = close[-(n_days + 1):]   # n_days+1 values → n_days comparisons
        consecutive = all(window[i + 1] > window[i] for i in range(n_days))

        sma_arr   = calc_sma(close, sma_period)
        atr_arr   = calc_atr(high, low, close, 14)
        vol_ratio = calc_volume_ratio(volume, 20)

        cur_close = close[-1]
        cur_sma   = float(sma_arr[-1]) if not np.isnan(sma_arr[-1]) else 0.0
        cur_atr   = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        cur_vol   = float(vol_ratio[-1]) if not np.isnan(vol_ratio[-1]) else 0.0

        trend_ok = cur_sma > 0 and cur_close > cur_sma
        vol_ok   = cur_vol >= vol_threshold
        strength = sum([consecutive, trend_ok, vol_ok])

        indicators = {
            "consecutive": consecutive,
            "n_days":      n_days,
            "sma":         cur_sma,
            "cur_close":   cur_close,
            "atr":         cur_atr,
            "vol_ratio":   cur_vol,
        }

        if not consecutive:
            return SignalResult(
                signal_type="hold", strength_score=0,
                indicators=indicators,
                reason=f"{n_days}일 연속 상승 미달",
            )

        if strength < 2:
            return SignalResult(
                signal_type="hold", strength_score=strength,
                indicators=indicators,
                reason=f"{n_days}일 연속 상승이나 확인 부족 (SMA위={trend_ok}, vol={cur_vol:.2f})",
            )

        if cur_atr <= 0:
            return SignalResult(
                signal_type="hold", strength_score=0,
                indicators=indicators,
                reason="ATR=0, SL 계산 불가",
            )

        entry = cur_close
        sl    = entry - cur_atr * sl_atr_mult
        if sl <= 0:
            return SignalResult(
                signal_type="hold", strength_score=0,
                indicators=indicators,
                reason=f"SL non-positive (ATR={cur_atr:.2f})",
            )

        return SignalResult(
            signal_type="long",
            strength_score=strength,
            entry_price=entry,
            sl=sl,
            tp1=entry + cur_atr * tp1_atr_mult,
            tp2=entry + cur_atr * tp2_atr_mult,
            indicators=indicators,
            reason=(
                f"롱: {n_days}일 연속 상승, "
                f"SMA{sma_period} 위, 거래량비율={cur_vol:.2f}"
            ),
        )
