"""Strong Close Strategy (KRX long-only).

Signal: close is in the top portion of the candle range (near the high),
indicating sustained buying pressure throughout the session.

close_pct = (close - low) / (high - low)

Long conditions (strength +1 each):
  1. close_pct >= close_threshold   — close near session high
  2. volume_ratio >= vol_threshold  — volume confirms buying
  3. close > SMA(sma_period)        — above trend filter

Minimum strength_score 2 required to be actionable.
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_atr, calc_sma, calc_volume_ratio


class StrongCloseStrategy(BaseStrategy):
    """Close-position-within-range momentum with volume + SMA confirmation.

    Parameters:
        close_threshold (float): Min (close-low)/(high-low) ratio. Default 0.75.
        sma_period (int):        Trend filter SMA period. Default 20.
        vol_threshold (float):   Min volume ratio. Default 1.2.
        sl_atr_mult (float):     ATR multiplier for stop-loss. Default 1.5.
        tp1_atr_mult (float):    ATR multiplier for TP1. Default 2.5.
        tp2_atr_mult (float):    ATR multiplier for TP2. Default 4.0.
    """

    DEFAULTS: dict = {
        "close_threshold": 0.75,
        "sma_period":      20,
        "vol_threshold":   1.2,
        "sl_atr_mult":     1.5,
        "tp1_atr_mult":    2.5,
        "tp2_atr_mult":    4.0,
    }

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        return frozenset({"any"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        adx          = float(indicators.get("adx", 25))
        above_sma200 = bool(indicators.get("above_sma200", False))
        atr_pct      = float(indicators.get("atr_pct", 1.0))
        base         = min(adx / 60.0, 1.0) * 0.45
        align_bonus  = 0.15 if above_sma200 else 0.0
        atr_bonus    = 0.10 if 0.5 <= atr_pct <= 4.0 else 0.0
        return round(min(base + align_bonus + atr_bonus, 0.75), 4)

    def get_name(self) -> str:
        return "strong_close"

    def get_min_candles(self) -> int:
        sma = int(self.params.get("sma_period", self.DEFAULTS["sma_period"]))
        return sma + 20

    def get_timeframe(self) -> str:
        return "1d"

    def generate_signal(self, df, symbol: str) -> SignalResult:
        p = {**self.DEFAULTS, **self.params}
        close_threshold = float(p["close_threshold"])
        sma_period      = int(p["sma_period"])
        vol_threshold   = float(p["vol_threshold"])
        sl_atr_mult     = float(p["sl_atr_mult"])
        tp1_atr_mult    = float(p["tp1_atr_mult"])
        tp2_atr_mult    = float(p["tp2_atr_mult"])

        close  = df["close"].to_numpy(dtype=float)
        high   = df["high"].to_numpy(dtype=float)
        low    = df["low"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)

        if len(close) < sma_period + 1:
            return SignalResult(signal_type="hold", strength_score=0, reason="insufficient data")

        cur_close = close[-1]
        cur_high  = high[-1]
        cur_low   = low[-1]
        candle_range = cur_high - cur_low

        if candle_range > 0:
            close_pct = (cur_close - cur_low) / candle_range
        else:
            close_pct = 0.5  # doji — neutral

        sma_arr   = calc_sma(close, sma_period)
        atr_arr   = calc_atr(high, low, close, 14)
        vol_ratio = calc_volume_ratio(volume, 20)

        cur_sma = float(sma_arr[-1]) if not np.isnan(sma_arr[-1]) else 0.0
        cur_atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        cur_vol = float(vol_ratio[-1]) if not np.isnan(vol_ratio[-1]) else 0.0

        strong_close = close_pct >= close_threshold
        trend_ok     = cur_sma > 0 and cur_close > cur_sma
        vol_ok       = cur_vol >= vol_threshold
        strength     = sum([strong_close, trend_ok, vol_ok])

        indicators = {
            "close_pct": close_pct,
            "sma":       cur_sma,
            "cur_close": cur_close,
            "atr":       cur_atr,
            "vol_ratio": cur_vol,
        }

        if not strong_close:
            return SignalResult(
                signal_type="hold", strength_score=0,
                indicators=indicators,
                reason=f"종가위치={close_pct:.2f} < {close_threshold} 기준",
            )

        if strength < 2:
            return SignalResult(
                signal_type="hold", strength_score=strength,
                indicators=indicators,
                reason=f"강한 종가이나 확인 부족 (SMA위={trend_ok}, vol={cur_vol:.2f})",
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
                f"롱: 강한 종가 {close_pct:.0%} (고가 대비), "
                f"SMA{sma_period} 위, 거래량비율={cur_vol:.2f}"
            ),
        )
