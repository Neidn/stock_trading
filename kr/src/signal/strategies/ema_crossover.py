"""EMA Crossover Strategy.

Classic dual-EMA trend-following system:
    LONG  — fast EMA crosses above slow EMA  AND  ADX > adx_threshold (confirmed trend)
    SHORT — fast EMA crosses below slow EMA  AND  ADX > adx_threshold (confirmed trend)

ADX filter prevents trading in choppy/ranging markets.
"""

from __future__ import annotations

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_adx, calc_atr, calc_ema


class EmaCrossoverStrategy(BaseStrategy):
    """EMA crossover with ADX trend filter.

    Parameters (read from self.params with defaults):
        ema_fast (int):       Fast EMA period. Default 20.
        ema_slow (int):       Slow EMA period. Default 50.
        adx_period (int):     ADX lookback. Default 14.
        adx_threshold (float): Min ADX to confirm trend. Default 25.
        sl_atr_mult (float):  ATR multiplier for stop-loss. Default 2.0.
        tp1_atr_mult (float): ATR multiplier for TP1. Default 3.0.
        tp2_atr_mult (float): ATR multiplier for TP2. Default 5.0.
    """

    DEFAULTS: dict = {
        "ema_fast":      20,
        "ema_slow":      50,
        "adx_period":    14,
        "adx_threshold": 25.0,
        "sl_atr_mult":   2.0,
        "tp1_atr_mult":  3.0,
        "tp2_atr_mult":  5.0,
    }

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        return frozenset({"trending"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        """Secondary strategy — base capped at 0.65; sma_aligned can lift to 0.85."""
        adx         = float(indicators.get("adx", 25))
        sma_aligned = bool(indicators.get("sma_aligned", False))
        adx_change  = float(indicators.get("adx_change", 0.0))
        base           = min(adx / 50.0, 1.0) * 0.65
        align_bonus    = 0.15 if sma_aligned else 0.0
        momentum_bonus = 0.05 if adx_change > 2 else 0.0
        return round(min(base + align_bonus + momentum_bonus, 0.85), 4)

    def get_name(self) -> str:
        return "ema_crossover"

    def get_min_candles(self) -> int:
        slow = self.params.get("ema_slow", self.DEFAULTS["ema_slow"])
        adx  = self.params.get("adx_period", self.DEFAULTS["adx_period"])
        return max(slow, adx) * 2 + 10

    def get_timeframe(self) -> str:
        return "1d"

    def _validate_params(self) -> None:
        p = {**self.DEFAULTS, **self.params}
        if p["ema_fast"] >= p["ema_slow"]:
            raise ValueError(
                f"ema_fast({p['ema_fast']}) must be < ema_slow({p['ema_slow']})"
            )
        if p["sl_atr_mult"] <= 0:
            raise ValueError(f"sl_atr_mult({p['sl_atr_mult']}) must be > 0")
        if p["tp1_atr_mult"] <= p["sl_atr_mult"]:
            raise ValueError(
                f"tp1_atr_mult({p['tp1_atr_mult']}) must be > sl_atr_mult({p['sl_atr_mult']})"
            )

    def generate_signal(self, df, symbol: str) -> SignalResult:
        p = {**self.DEFAULTS, **self.params}

        ema_fast_p    = int(p["ema_fast"])
        ema_slow_p    = int(p["ema_slow"])
        adx_period    = int(p["adx_period"])
        adx_threshold = float(p["adx_threshold"])
        sl_atr_mult   = float(p["sl_atr_mult"])
        tp1_atr_mult  = float(p["tp1_atr_mult"])
        tp2_atr_mult  = float(p["tp2_atr_mult"])

        close  = df["close"].to_numpy(dtype=float)
        high   = df["high"].to_numpy(dtype=float)
        low    = df["low"].to_numpy(dtype=float)

        ema_fast = calc_ema(close, ema_fast_p)
        ema_slow = calc_ema(close, ema_slow_p)
        adx      = calc_adx(high, low, close, adx_period)
        atr      = calc_atr(high, low, close)

        if len(close) < 2:
            return SignalResult(signal_type="hold", strength_score=0, reason="insufficient data")

        cur_fast,  prev_fast  = ema_fast[-1],  ema_fast[-2]
        cur_slow,  prev_slow  = ema_slow[-1],  ema_slow[-2]
        cur_adx               = adx[-1]
        cur_atr               = atr[-1]
        cur_close             = close[-1]

        cross_up   = prev_fast <= prev_slow and cur_fast > cur_slow
        cross_down = prev_fast >= prev_slow and cur_fast < cur_slow
        trend_ok   = cur_adx >= adx_threshold

        indicators = {
            "ema_fast":  cur_fast,
            "ema_slow":  cur_slow,
            "adx":       cur_adx,
            "atr":       cur_atr,
        }

        if cross_up and trend_ok:
            entry = cur_close
            sl    = entry - cur_atr * sl_atr_mult
            if sl <= 0:
                return SignalResult(
                    signal_type="hold", strength_score=0,
                    reason=f"ATR({cur_atr:.6f}) exceeds entry({entry:.6f}); SL would be negative",
                )
            return SignalResult(
                signal_type="long",
                strength_score=2,
                entry_price=entry,
                sl=sl,
                tp1=entry + cur_atr * tp1_atr_mult,
                tp2=entry + cur_atr * tp2_atr_mult,
                indicators=indicators,
                reason=(
                    f"롱: EMA{ema_fast_p}({cur_fast:.4f}) crossed above "
                    f"EMA{ema_slow_p}({cur_slow:.4f}), ADX={cur_adx:.1f}"
                ),
            )

        if cross_down and trend_ok:
            entry = cur_close
            sl    = entry + cur_atr * sl_atr_mult
            return SignalResult(
                signal_type="short",
                strength_score=2,
                entry_price=entry,
                sl=sl,
                tp1=entry - cur_atr * tp1_atr_mult,
                tp2=entry - cur_atr * tp2_atr_mult,
                indicators=indicators,
                reason=(
                    f"숏: EMA{ema_fast_p}({cur_fast:.4f}) crossed below "
                    f"EMA{ema_slow_p}({cur_slow:.4f}), ADX={cur_adx:.1f}"
                ),
            )

        adx_reason = f"ADX={cur_adx:.1f} < {adx_threshold} (no trend)" if not trend_ok else ""
        return SignalResult(
            signal_type="hold",
            strength_score=0,
            reason=adx_reason or "no crossover",
        )
