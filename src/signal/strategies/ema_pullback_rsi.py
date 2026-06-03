"""EMA Pullback + RSI Zone strategy.

Unlike flip-based strategies (SuperTrend, EMA crossover) that fire once per
trend change, this fires multiple times within a sustained trend — every time
RSI pulls back to the neutral zone after a momentum surge.

Entry: Multi-EMA alignment (trend confirmed) + RSI reset to neutral zone
       + confirming candle color.
  Long  — EMA20 > EMA50 > EMA200  AND  ADX > threshold
           AND  RSI in [rsi_low, rsi_high]  AND  close > open
  Short — EMA20 < EMA50 < EMA200  AND  ADX > threshold
           AND  RSI in [rsi_low, rsi_high]  AND  close < open

Signal frequency: 3–5× per week per symbol vs 1× per 2 weeks for flip strategies.
Best in strong trending markets (ADX 30–60).

SL: ATR multiple from entry.
TP: ATR multiples from entry.
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_adx, calc_atr, calc_ema, calc_rsi


class EmaPullbackRsiStrategy(BaseStrategy):
    """Multi-EMA trend continuation with RSI pullback entry.

    Parameters (read from self.params with defaults):
        ema_fast (int):       Fast EMA period. Default 20.
        ema_mid (int):        Mid EMA period. Default 50.
        ema_slow (int):       Slow EMA period (macro trend filter). Default 200.
        rsi_period (int):     RSI lookback. Default 14.
        rsi_low (float):      Lower bound of RSI neutral zone. Default 45.
        rsi_high (float):     Upper bound of RSI neutral zone. Default 60.
        adx_period (int):     ADX lookback. Default 14.
        adx_threshold (float): Minimum ADX for trend confirmation. Default 30.
        sl_atr_mult (float):  ATR multiplier for stop-loss. Default 2.0.
        tp1_atr_mult (float): ATR multiplier for TP1 (full close). Default 2.5.
    """

    DEFAULTS: dict = {
        "ema_fast":      20,
        "ema_mid":       50,
        "ema_slow":      200,
        "rsi_period":    14,
        "rsi_low":       40.0,
        "rsi_high":      55.0,
        "adx_period":    14,
        "adx_threshold": 20.0,
        "sl_atr_mult":   2.5,
        "tp1_atr_mult":  3.0,
    }

    def get_name(self) -> str:
        return "ema_pullback_rsi"

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        return frozenset({"trending", "volatile"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        """Peaks at ADX 38 — established trending markets.

        Takes over from rsi_supertrend above ADX 32 (trend established,
        continuation entries more valuable than waiting for a flip).
        Hands off to macd_sma200 above ADX 55 (extreme trend strength).
        sma_aligned bonus: multi-TF structure = more frequent pullback entries.
        """
        adx          = float(indicators.get("adx", 25))
        atr_pct      = float(indicators.get("atr_pct", 1.0))
        sma_aligned  = bool(indicators.get("sma_aligned", False))
        sma50_slope  = float(indicators.get("sma50_slope", 0.0))
        adx_change   = float(indicators.get("adx_change", 0.0))
        adx_score = min(adx / 38.0, 1.0) * max(0.0, 1.0 - (adx - 38.0) / 38.0)
        vol_score = min(atr_pct / 4.0, 1.0) * max(0.0, 1.0 - (atr_pct - 4.0) / 20.0)
        base = (adx_score + vol_score) / 2.0
        align_bonus    = 0.12 if sma_aligned else 0.0
        slope_bonus    = min(abs(sma50_slope) / 0.02, 1.0) * 0.08
        momentum_bonus = 0.05 if adx_change > 2 else 0.0
        return round(min(base + align_bonus + slope_bonus + momentum_bonus, 1.0), 4)

    def get_min_candles(self) -> int:
        p = {**self.DEFAULTS, **self.params}
        return int(p["ema_slow"]) + int(p["rsi_period"]) + 10

    def get_timeframe(self) -> str:
        return "1h"

    def _validate_params(self) -> None:
        p = {**self.DEFAULTS, **self.params}
        if not (int(p["ema_fast"]) < int(p["ema_mid"]) < int(p["ema_slow"])):
            raise ValueError(
                f"EMA periods must satisfy ema_fast < ema_mid < ema_slow, "
                f"got {p['ema_fast']} / {p['ema_mid']} / {p['ema_slow']}"
            )
        if not (0 < float(p["rsi_low"]) < float(p["rsi_high"]) < 100):
            raise ValueError(
                f"rsi_low({p['rsi_low']}) < rsi_high({p['rsi_high']}) required, both in (0,100)"
            )
        if float(p["adx_threshold"]) <= 0:
            raise ValueError(f"adx_threshold({p['adx_threshold']}) must be > 0")
        if float(p["sl_atr_mult"]) <= 0:
            raise ValueError(f"sl_atr_mult({p['sl_atr_mult']}) must be > 0")
        if float(p["tp1_atr_mult"]) <= float(p["sl_atr_mult"]):
            raise ValueError(
                f"tp1_atr_mult({p['tp1_atr_mult']}) must be > sl_atr_mult({p['sl_atr_mult']})"
            )
        if float(p["tp2_atr_mult"]) <= float(p["tp1_atr_mult"]):
            raise ValueError(
                f"tp2_atr_mult({p['tp2_atr_mult']}) must be > tp1_atr_mult({p['tp1_atr_mult']})"
            )

    def generate_signal(self, df, symbol: str) -> SignalResult:
        p = {**self.DEFAULTS, **self.params}

        ema_fast_p    = int(p["ema_fast"])
        ema_mid_p     = int(p["ema_mid"])
        ema_slow_p    = int(p["ema_slow"])
        rsi_period    = int(p["rsi_period"])
        rsi_low       = float(p["rsi_low"])
        rsi_high      = float(p["rsi_high"])
        adx_period    = int(p["adx_period"])
        adx_threshold = float(p["adx_threshold"])
        sl_atr_mult   = float(p["sl_atr_mult"])
        tp1_atr_mult  = float(p["tp1_atr_mult"])
        tp2_atr_mult  = float(p["tp2_atr_mult"])

        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        open_ = df["open"].to_numpy(dtype=float)

        ema_fast = calc_ema(close, ema_fast_p)
        ema_mid  = calc_ema(close, ema_mid_p)
        ema_slow = calc_ema(close, ema_slow_p)
        rsi      = calc_rsi(close, rsi_period)
        adx      = calc_adx(high, low, close, adx_period)
        atr      = calc_atr(high, low, close, adx_period)

        cur_ema_fast = float(ema_fast[-1])
        cur_ema_mid  = float(ema_mid[-1])
        cur_ema_slow = float(ema_slow[-1])
        cur_rsi      = float(rsi[-1])
        cur_adx      = float(adx[-1])
        cur_atr      = float(atr[-1])
        cur_close    = float(close[-1])
        cur_open     = float(open_[-1])

        if np.isnan(cur_rsi) or np.isnan(cur_adx) or np.isnan(cur_atr):
            return SignalResult(signal_type="none", reason="지표 계산값 NaN")

        indicators_snapshot = {
            "ema_fast": cur_ema_fast,
            "ema_mid":  cur_ema_mid,
            "ema_slow": cur_ema_slow,
            "rsi":      cur_rsi,
            "adx":      cur_adx,
            "atr":      cur_atr,
        }

        trend_up   = cur_ema_fast > cur_ema_mid > cur_ema_slow
        trend_down = cur_ema_fast < cur_ema_mid < cur_ema_slow
        rsi_neutral  = rsi_low <= cur_rsi <= rsi_high
        adx_ok       = cur_adx >= adx_threshold
        candle_green = cur_close > cur_open
        candle_red   = cur_close < cur_open

        if not adx_ok:
            return SignalResult(
                signal_type="none",
                indicators=indicators_snapshot,
                reason=f"ADX={cur_adx:.1f} < {adx_threshold} — 추세 약함",
            )

        if trend_up and rsi_neutral and candle_green:
            entry = cur_close
            sl    = entry - cur_atr * sl_atr_mult
            if sl <= 0:
                return SignalResult(
                    signal_type="none",
                    indicators=indicators_snapshot,
                    reason=f"SL 계산 오류: sl={sl:.6f}",
                )
            return SignalResult(
                signal_type="long",
                strength_score=3,
                entry_price=entry,
                sl=sl,
                tp1=entry + cur_atr * tp1_atr_mult,
                indicators=indicators_snapshot,
                reason=(
                    f"롱: EMA정렬상승 + RSI={cur_rsi:.1f}∈[{rsi_low},{rsi_high}] + 양봉 | "
                    f"ADX={cur_adx:.1f} fast={cur_ema_fast:.4f} mid={cur_ema_mid:.4f} slow={cur_ema_slow:.4f}"
                ),
            )

        if trend_down and rsi_neutral and candle_red:
            entry = cur_close
            sl    = entry + cur_atr * sl_atr_mult
            return SignalResult(
                signal_type="short",
                strength_score=3,
                entry_price=entry,
                sl=sl,
                tp1=entry - cur_atr * tp1_atr_mult,
                indicators=indicators_snapshot,
                reason=(
                    f"숏: EMA정렬하락 + RSI={cur_rsi:.1f}∈[{rsi_low},{rsi_high}] + 음봉 | "
                    f"ADX={cur_adx:.1f} fast={cur_ema_fast:.4f} mid={cur_ema_mid:.4f} slow={cur_ema_slow:.4f}"
                ),
            )

        # Build no-signal reason
        if trend_up:
            reason = f"상승추세 but RSI={cur_rsi:.1f} 중립대 밖" if not rsi_neutral else f"상승추세+RSI중립 but 음봉"
        elif trend_down:
            reason = f"하락추세 but RSI={cur_rsi:.1f} 중립대 밖" if not rsi_neutral else f"하락추세+RSI중립 but 양봉"
        else:
            reason = (
                f"EMA정렬없음: fast={cur_ema_fast:.4f} mid={cur_ema_mid:.4f} slow={cur_ema_slow:.4f} | "
                f"RSI={cur_rsi:.1f} ADX={cur_adx:.1f}"
            )

        return SignalResult(
            signal_type="none",
            indicators=indicators_snapshot,
            reason=reason,
        )
