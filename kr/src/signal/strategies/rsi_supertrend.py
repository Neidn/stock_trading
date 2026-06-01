"""RSI + SuperTrend hybrid strategy.

Entry: SuperTrend direction flip confirmed by RSI momentum.
  Long  — trend flips to +1 AND RSI > rsi_threshold (default 50)
  Short — trend flips to -1 AND RSI < (100 - rsi_threshold)

RSI filter prevents entering against momentum:
  a SuperTrend flip with RSI in the wrong zone is likely a fakeout.

SL: SuperTrend band at entry (natural exit level), floored/capped by ATR multiple.
TP: ATR multiples from entry.
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_atr, calc_rsi
from src.signal.strategies.supertrend import _calc_supertrend


class RsiSupertrendStrategy(BaseStrategy):
    """RSI-confirmed SuperTrend strategy.

    Parameters (read from self.params with defaults):
        atr_period (int):      ATR lookback for SuperTrend bands. Default 10.
        multiplier (float):    Band width multiplier. Default 3.0.
        rsi_period (int):      RSI lookback. Default 14.
        rsi_threshold (float): RSI level for momentum confirmation.
            Long  when RSI >  rsi_threshold.
            Short when RSI < (100 - rsi_threshold). Default 50.
        sl_atr_mult (float):   ATR multiplier for stop-loss. Default 2.0.
        tp1_atr_mult (float):  ATR multiplier for TP1. Default 3.0.
        tp2_atr_mult (float):  ATR multiplier for TP2. Default 5.0.
    """

    DEFAULTS: dict = {
        "atr_period":    10,
        "multiplier":    3.0,
        "rsi_period":    14,
        "rsi_threshold": 50.0,
        "sl_atr_mult":   2.0,
        "tp1_atr_mult":  3.0,
        "tp2_atr_mult":  5.0,
    }

    def get_name(self) -> str:
        return "rsi_supertrend"

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        # Best in transitioning/trending markets; RSI filter reduces whipsaws in volatile
        return frozenset({"trending", "volatile", "ranging"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        """Peak score at ADX 28 (transitioning markets, trend just forming).

        Wins in ADX 20-32 zone. ema_pullback_rsi takes over above ADX 32.
        sma_aligned penalty: fully aligned SMAs = trend already established,
        not transitioning — reduces score to yield to ema_pullback_rsi.
        """
        adx         = float(indicators.get("adx", 25))
        atr_pct     = float(indicators.get("atr_pct", 1.0))
        sma_aligned = bool(indicators.get("sma_aligned", False))
        adx_change  = float(indicators.get("adx_change", 0.0))
        adx_score = min(adx / 28.0, 1.0) * max(0.0, 1.0 - (adx - 28.0) / 28.0)
        vol_score = min(atr_pct / 4.0, 1.0) * max(0.0, 1.0 - (atr_pct - 4.0) / 20.0)
        base = (adx_score + vol_score) / 2.0
        # Trend just forming (ADX rising, SMAs not yet aligned) = ideal zone
        align_adj      = -0.08 if sma_aligned else 0.05
        momentum_bonus = 0.05 if 0 < adx_change <= 5 else 0.0
        return round(max(0.0, min(base + align_adj + momentum_bonus, 1.0)), 4)

    def get_min_candles(self) -> int:
        atr_period = int(self.params.get("atr_period", self.DEFAULTS["atr_period"]))
        rsi_period = int(self.params.get("rsi_period", self.DEFAULTS["rsi_period"]))
        return max(atr_period * 3, rsi_period * 2) + 10

    def get_timeframe(self) -> str:
        return "1d"

    def _validate_params(self) -> None:
        p = {**self.DEFAULTS, **self.params}
        if float(p["multiplier"]) <= 0:
            raise ValueError(f"multiplier({p['multiplier']}) must be > 0")
        if not (0 < float(p["rsi_threshold"]) < 100):
            raise ValueError(f"rsi_threshold({p['rsi_threshold']}) must be between 0 and 100")
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

        atr_period    = int(p["atr_period"])
        multiplier    = float(p["multiplier"])
        rsi_period    = int(p["rsi_period"])
        rsi_threshold = float(p["rsi_threshold"])
        sl_atr_mult   = float(p["sl_atr_mult"])
        tp1_atr_mult  = float(p["tp1_atr_mult"])
        tp2_atr_mult  = float(p["tp2_atr_mult"])

        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)

        up, dn, trend = _calc_supertrend(high, low, close, atr_period, multiplier)
        atr = calc_atr(high, low, close, atr_period)
        rsi = calc_rsi(close, rsi_period)

        cur_trend  = trend[-1]
        prev_trend = trend[-2]
        cur_rsi    = float(rsi[-1])
        cur_atr    = float(atr[-1])
        cur_up     = float(up[-1])
        cur_dn     = float(dn[-1])
        entry      = float(close[-1])

        if np.isnan(cur_rsi) or np.isnan(cur_atr):
            return SignalResult(signal_type="none", reason="지표 계산값 NaN")

        indicators_snapshot = {
            "supertrend_up": cur_up,
            "supertrend_dn": cur_dn,
            "trend":         int(cur_trend),
            "rsi":           cur_rsi,
            "atr":           cur_atr,
        }

        trend_flipped_long  = cur_trend == 1  and prev_trend == -1
        trend_flipped_short = cur_trend == -1 and prev_trend == 1
        rsi_bullish = cur_rsi > rsi_threshold
        rsi_bearish = cur_rsi < (100.0 - rsi_threshold)

        if trend_flipped_long:
            if not rsi_bullish:
                return SignalResult(
                    signal_type="none",
                    indicators=indicators_snapshot,
                    reason=f"SuperTrend 상향 전환 but RSI 미확인 | RSI={cur_rsi:.1f} < {rsi_threshold}",
                )
            band_sl = cur_up
            atr_sl  = entry - cur_atr * sl_atr_mult
            sl      = max(band_sl, atr_sl)
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
                tp2=entry + cur_atr * tp2_atr_mult,
                indicators=indicators_snapshot,
                reason=(
                    f"롱: SuperTrend 상향 전환 + RSI={cur_rsi:.1f} > {rsi_threshold} | "
                    f"up={cur_up:.4f} ATR={cur_atr:.4f}"
                ),
            )

        if trend_flipped_short:
            if not rsi_bearish:
                return SignalResult(
                    signal_type="none",
                    indicators=indicators_snapshot,
                    reason=f"SuperTrend 하향 전환 but RSI 미확인 | RSI={cur_rsi:.1f} > {100.0 - rsi_threshold}",
                )
            band_sl = cur_dn
            atr_sl  = entry + cur_atr * sl_atr_mult
            sl      = min(band_sl, atr_sl)
            return SignalResult(
                signal_type="short",
                strength_score=3,
                entry_price=entry,
                sl=sl,
                tp1=entry - cur_atr * tp1_atr_mult,
                tp2=entry - cur_atr * tp2_atr_mult,
                indicators=indicators_snapshot,
                reason=(
                    f"숏: SuperTrend 하향 전환 + RSI={cur_rsi:.1f} < {100.0 - rsi_threshold} | "
                    f"dn={cur_dn:.4f} ATR={cur_atr:.4f}"
                ),
            )

        direction = "bullish" if cur_trend == 1 else "bearish"
        return SignalResult(
            signal_type="none",
            indicators=indicators_snapshot,
            reason=f"trend={direction}, no flip | RSI={cur_rsi:.1f}",
        )
