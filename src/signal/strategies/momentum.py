"""N-Day Momentum Strategy (KRX long-only).

Signal: N-day price return exceeds threshold, confirmed by SMA trend filter
and volume surge.

Long conditions (strength +1 each):
  1. (close[-1] - close[-lookback-1]) / close[-lookback-1] >= min_return_pct
  2. close > SMA(sma_period)         — price above trend
  3. volume_ratio >= vol_threshold   — volume confirms move

Minimum strength_score 2 required to be actionable.
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_atr, calc_sma, calc_volume_ratio


class MomentumStrategy(BaseStrategy):
    """N-day return momentum with SMA trend filter and volume confirmation.

    Parameters:
        lookback (int):         Return calculation window. Default 20 days.
        min_return_pct (float): Minimum N-day return (%) to trigger. Default 8.0.
        sma_period (int):       Trend filter SMA period. Default 50.
        vol_threshold (float):  Min volume ratio to confirm. Default 1.2.
        sl_atr_mult (float):    ATR multiplier for stop-loss. Default 2.0.
        tp1_atr_mult (float):   ATR multiplier for TP1. Default 3.0.
        tp2_atr_mult (float):   ATR multiplier for TP2. Default 5.0.
    """

    DEFAULTS: dict = {
        "lookback":       20,
        "min_return_pct": 8.0,
        "sma_period":     50,
        "vol_threshold":  1.2,
        "sl_atr_mult":    2.0,
        "tp1_atr_mult":   3.0,
        "tp2_atr_mult":   5.0,
    }

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        return frozenset({"trending"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        adx          = float(indicators.get("adx", 25))
        above_sma200 = bool(indicators.get("above_sma200", False))
        atr_pct      = float(indicators.get("atr_pct", 1.0))
        base         = min(adx / 50.0, 1.0) * 0.55
        align_bonus  = 0.25 if above_sma200 else 0.0
        atr_bonus    = 0.10 if 1.0 <= atr_pct <= 3.0 else 0.0
        return round(min(base + align_bonus + atr_bonus, 0.90), 4)

    def get_name(self) -> str:
        return "momentum"

    def get_min_candles(self) -> int:
        lookback   = int(self.params.get("lookback",   self.DEFAULTS["lookback"]))
        sma_period = int(self.params.get("sma_period", self.DEFAULTS["sma_period"]))
        return max(lookback, sma_period) + 20

    def get_timeframe(self) -> str:
        return "1d"

    def generate_signal(self, df, symbol: str) -> SignalResult:
        p = {**self.DEFAULTS, **self.params}
        lookback       = int(p["lookback"])
        min_return_pct = float(p["min_return_pct"])
        sma_period     = int(p["sma_period"])
        vol_threshold  = float(p["vol_threshold"])
        sl_atr_mult    = float(p["sl_atr_mult"])
        tp1_atr_mult   = float(p["tp1_atr_mult"])
        tp2_atr_mult   = float(p["tp2_atr_mult"])

        close  = df["close"].to_numpy(dtype=float)
        high   = df["high"].to_numpy(dtype=float)
        low    = df["low"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)

        if len(close) < max(lookback, sma_period) + 1:
            return SignalResult(signal_type="hold", strength_score=0, reason="insufficient data")

        cur_close  = close[-1]
        past_close = close[-(lookback + 1)]
        ret_pct    = (cur_close - past_close) / past_close * 100.0 if past_close > 0 else 0.0

        sma_arr   = calc_sma(close, sma_period)
        atr_arr   = calc_atr(high, low, close, 14)
        vol_ratio = calc_volume_ratio(volume, 20)

        cur_sma = float(sma_arr[-1]) if not np.isnan(sma_arr[-1]) else 0.0
        cur_atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        cur_vol = float(vol_ratio[-1]) if not np.isnan(vol_ratio[-1]) else 0.0

        momentum_ok = ret_pct >= min_return_pct
        trend_ok    = cur_sma > 0 and cur_close > cur_sma
        vol_ok      = cur_vol >= vol_threshold
        strength    = sum([momentum_ok, trend_ok, vol_ok])

        indicators = {
            "ret_pct":   ret_pct,
            "sma":       cur_sma,
            "cur_close": cur_close,
            "atr":       cur_atr,
            "vol_ratio": cur_vol,
        }

        if not momentum_ok:
            return SignalResult(
                signal_type="hold", strength_score=0,
                indicators=indicators,
                reason=f"{lookback}일 수익률={ret_pct:.1f}% < {min_return_pct}% 기준",
            )

        if strength < 2:
            return SignalResult(
                signal_type="hold", strength_score=strength,
                indicators=indicators,
                reason=f"모멘텀 있으나 확인 부족 (SMA위={trend_ok}, vol={cur_vol:.2f})",
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
                f"롱: {lookback}일 수익률={ret_pct:.1f}%, "
                f"SMA{sma_period} 위, 거래량비율={cur_vol:.2f}"
            ),
        )
