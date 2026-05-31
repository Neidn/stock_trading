"""52-Week High Breakout Strategy (KRX long-only).

Signal: close breaks above N-day rolling high with volume + ADX confirmation.

Long conditions (strength +1 each):
  1. close > max(close[-n_days:-1])  — new N-day high
  2. volume_ratio >= vol_threshold   — volume surge confirms breakout
  3. ADX >= adx_min                  — trend strength present

Minimum strength_score 2 required to be actionable.
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_adx, calc_atr, calc_volume_ratio


class Week52HighStrategy(BaseStrategy):
    """N-day high breakout with volume + ADX confirmation.

    Parameters:
        n_days (int):           Rolling high lookback. Default 100.
        vol_threshold (float):  Min volume ratio to confirm. Default 1.3.
        adx_min (float):        Min ADX for trend strength point. Default 20.
        sl_atr_mult (float):    ATR multiplier for stop-loss. Default 2.0.
        tp1_atr_mult (float):   ATR multiplier for TP1. Default 3.0.
        tp2_atr_mult (float):   ATR multiplier for TP2. Default 5.0.
    """

    DEFAULTS: dict = {
        "n_days":        100,
        "vol_threshold": 1.3,
        "adx_min":       20.0,
        "sl_atr_mult":   2.0,
        "tp1_atr_mult":  3.0,
        "tp2_atr_mult":  5.0,
    }

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        return frozenset({"trending", "volatile"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        adx          = float(indicators.get("adx", 25))
        above_sma200 = bool(indicators.get("above_sma200", False))
        atr_pct      = float(indicators.get("atr_pct", 1.0))
        base         = min(adx / 50.0, 1.0) * 0.65
        align_bonus  = 0.20 if above_sma200 else 0.0
        vol_penalty  = -0.10 if atr_pct > 5.0 else 0.0
        return round(min(max(base + align_bonus + vol_penalty, 0.0), 0.90), 4)

    def get_name(self) -> str:
        return "week52_high"

    def get_min_candles(self) -> int:
        n = int(self.params.get("n_days", self.DEFAULTS["n_days"]))
        return n + 28  # ADX needs 2 × period warm-up

    def get_timeframe(self) -> str:
        return "D"

    def generate_signal(self, df, symbol: str) -> SignalResult:
        p = {**self.DEFAULTS, **self.params}
        n_days        = int(p["n_days"])
        vol_threshold = float(p["vol_threshold"])
        adx_min       = float(p["adx_min"])
        sl_atr_mult   = float(p["sl_atr_mult"])
        tp1_atr_mult  = float(p["tp1_atr_mult"])
        tp2_atr_mult  = float(p["tp2_atr_mult"])

        close  = df["close"].to_numpy(dtype=float)
        high   = df["high"].to_numpy(dtype=float)
        low    = df["low"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)

        if len(close) < n_days + 1:
            return SignalResult(signal_type="hold", strength_score=0, reason="insufficient data")

        cur_close  = close[-1]
        prior_high = float(np.max(close[-(n_days + 1):-1]))

        adx_arr   = calc_adx(high, low, close, 14)
        atr_arr   = calc_atr(high, low, close, 14)
        vol_ratio = calc_volume_ratio(volume, 20)

        cur_adx = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0
        cur_atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        cur_vol = float(vol_ratio[-1]) if not np.isnan(vol_ratio[-1]) else 0.0

        new_high = cur_close > prior_high
        vol_ok   = cur_vol >= vol_threshold
        trend_ok = cur_adx >= adx_min
        strength = sum([new_high, vol_ok, trend_ok])

        indicators = {
            "prior_high": prior_high,
            "cur_close":  cur_close,
            "adx":        cur_adx,
            "atr":        cur_atr,
            "vol_ratio":  cur_vol,
        }

        if not new_high:
            return SignalResult(
                signal_type="hold", strength_score=0,
                indicators=indicators,
                reason=f"close({cur_close:.0f}) <= {n_days}일 고점({prior_high:.0f})",
            )

        if strength < 2:
            return SignalResult(
                signal_type="hold", strength_score=strength,
                indicators=indicators,
                reason=f"돌파했으나 확인 부족 (vol_ratio={cur_vol:.2f}, ADX={cur_adx:.1f})",
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
                f"롱: {n_days}일 신고가 돌파 {cur_close:.0f} > {prior_high:.0f}, "
                f"거래량비율={cur_vol:.2f}, ADX={cur_adx:.1f}"
            ),
        )
