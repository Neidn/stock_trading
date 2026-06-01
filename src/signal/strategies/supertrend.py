"""SuperTrend Strategy.

Port of KivancOzbilgic's SuperTrend STRATEGY (Pine Script v4).

SuperTrend bands:
    up  = hl2 - (multiplier * ATR)   # support band (ratchets upward)
    dn  = hl2 + (multiplier * ATR)   # resistance band (ratchets downward)

Trend flips:
    LONG  — trend flips from -1 to +1  (close breaks above dn band)
    SHORT — trend flips from +1 to -1  (close breaks below up band)

SL: the active SuperTrend band at entry (natural SuperTrend exit level).
TP: ATR-based multiples from entry.
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_atr


def _calc_supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (up_band, dn_band, trend) arrays.

    trend[i] == 1  → bullish (up band is active support)
    trend[i] == -1 → bearish (dn band is active resistance)
    """
    n = len(close)
    hl2 = (high + low) / 2.0
    atr = calc_atr(high, low, close, period)

    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    trend = np.ones(n)

    for i in range(1, n):
        if np.isnan(atr[i]):
            trend[i] = trend[i - 1]
            up[i] = up[i - 1] if not np.isnan(up[i - 1]) else hl2[i] - multiplier * 0
            dn[i] = dn[i - 1] if not np.isnan(dn[i - 1]) else hl2[i] + multiplier * 0
            continue

        basic_up = hl2[i] - multiplier * atr[i]
        basic_dn = hl2[i] + multiplier * atr[i]

        prev_up = up[i - 1] if not np.isnan(up[i - 1]) else basic_up
        prev_dn = dn[i - 1] if not np.isnan(dn[i - 1]) else basic_dn

        # Ratchet: up only moves up when previous close was above it
        up[i] = max(basic_up, prev_up) if close[i - 1] > prev_up else basic_up
        # Ratchet: dn only moves down when previous close was below it
        dn[i] = min(basic_dn, prev_dn) if close[i - 1] < prev_dn else basic_dn

        prev_trend = trend[i - 1]
        if prev_trend == -1 and close[i] > prev_dn:
            trend[i] = 1
        elif prev_trend == 1 and close[i] < prev_up:
            trend[i] = -1
        else:
            trend[i] = prev_trend

    return up, dn, trend


class SupertrendStrategy(BaseStrategy):
    """SuperTrend trend-following strategy.

    Parameters (read from self.params with defaults):
        atr_period (int):     ATR lookback for SuperTrend bands. Default 10.
        multiplier (float):   Band width multiplier. Default 3.0.
        sl_atr_mult (float):  ATR multiplier for TP stop-loss. Default 2.0.
                              SL is max(band_sl, entry - sl_atr_mult * ATR).
        tp1_atr_mult (float): ATR multiplier for TP1. Default 3.0.
        tp2_atr_mult (float): ATR multiplier for TP2. Default 5.0.
    """

    DEFAULTS: dict = {
        "atr_period":    10,
        "multiplier":    3.0,
        "sl_atr_mult":   2.0,
        "tp1_atr_mult":  3.0,
        "tp2_atr_mult":  5.0,
    }

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        return frozenset({"trending", "volatile"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        """Peaks at ADX 35. sma_aligned confirms band direction is trustworthy."""
        adx         = float(indicators.get("adx", 25))
        atr_pct     = float(indicators.get("atr_pct", 1.0))
        sma_aligned = bool(indicators.get("sma_aligned", False))
        adx_change  = float(indicators.get("adx_change", 0.0))
        trend_score = min(adx / 35.0, 1.0) * max(0.0, 1.0 - (adx - 35.0) / 35.0)
        vol_score   = min(atr_pct / 5.0, 1.0)
        base           = (trend_score + vol_score) / 2.0
        align_bonus    = 0.10 if sma_aligned else 0.0
        momentum_bonus = 0.05 if adx_change > 2 else 0.0
        # Strong trend + high volatility: supertrend bands are most effective here
        high_vol_bonus = 0.15 if atr_pct > 3.0 and adx > 30 else 0.0
        return round(min(base + align_bonus + momentum_bonus + high_vol_bonus, 0.95), 4)

    def get_name(self) -> str:
        return "supertrend"

    def get_min_candles(self) -> int:
        period = self.params.get("atr_period", self.DEFAULTS["atr_period"])
        return int(period) * 3 + 10

    def get_timeframe(self) -> str:
        return "1d"

    def _validate_params(self) -> None:
        p = {**self.DEFAULTS, **self.params}
        if float(p["multiplier"]) <= 0:
            raise ValueError(f"multiplier({p['multiplier']}) must be > 0")
        if float(p["sl_atr_mult"]) <= 0:
            raise ValueError(f"sl_atr_mult({p['sl_atr_mult']}) must be > 0")
        if float(p["tp1_atr_mult"]) <= float(p["sl_atr_mult"]):
            raise ValueError(
                f"tp1_atr_mult({p['tp1_atr_mult']}) must be > sl_atr_mult({p['sl_atr_mult']})"
            )

    def generate_signal(self, df, symbol: str) -> SignalResult:
        p = {**self.DEFAULTS, **self.params}

        atr_period   = int(p["atr_period"])
        multiplier   = float(p["multiplier"])
        sl_atr_mult  = float(p["sl_atr_mult"])
        tp1_atr_mult = float(p["tp1_atr_mult"])
        tp2_atr_mult = float(p["tp2_atr_mult"])

        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)

        if len(close) < self.get_min_candles():
            return SignalResult(signal_type="hold", strength_score=0,
                                reason="insufficient candles")

        up, dn, trend = _calc_supertrend(high, low, close, atr_period, multiplier)
        atr = calc_atr(high, low, close, atr_period)

        cur_trend  = trend[-1]
        prev_trend = trend[-2]
        cur_atr    = atr[-1]
        cur_up     = up[-1]
        cur_dn     = dn[-1]
        entry      = close[-1]

        buy_signal  = cur_trend == 1 and prev_trend == -1   # trend flipped up
        sell_signal = cur_trend == -1 and prev_trend == 1   # trend flipped down

        indicators = {
            "supertrend_up":  cur_up,
            "supertrend_dn":  cur_dn,
            "trend":          int(cur_trend),
            "atr":            cur_atr,
        }

        if buy_signal:
            # SL = SuperTrend support band (up line), floored by ATR multiple
            band_sl = cur_up
            atr_sl  = entry - cur_atr * sl_atr_mult
            sl      = max(band_sl, atr_sl)  # tighter of the two
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
                tp1=entry + cur_atr * tp1_atr_mult,
                tp2=entry + cur_atr * tp2_atr_mult,
                indicators=indicators,
                reason=f"롱: SuperTrend 상향 전환, up={cur_up:.6f}, ATR={cur_atr:.6f}",
            )

        if sell_signal:
            # SL = SuperTrend resistance band (dn line), capped by ATR multiple
            band_sl = cur_dn
            atr_sl  = entry + cur_atr * sl_atr_mult
            sl      = min(band_sl, atr_sl)  # tighter of the two
            return SignalResult(
                signal_type="short",
                strength_score=3,
                entry_price=entry,
                sl=sl,
                tp1=entry - cur_atr * tp1_atr_mult,
                tp2=entry - cur_atr * tp2_atr_mult,
                indicators=indicators,
                reason=f"숏: SuperTrend 하향 전환, dn={cur_dn:.6f}, ATR={cur_atr:.6f}",
            )

        return SignalResult(
            signal_type="hold",
            strength_score=0,
            reason=f"trend={'bullish' if cur_trend == 1 else 'bearish'}, no flip",
        )
