"""RSI + MACD confluence strategy.

Enters long when RSI is oversold, MACD histogram is turning up, price is
above EMA(50), and volume is elevated.  Mirror conditions apply for short.
Requires at least 2 of 4 conditions to generate an actionable signal.
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import (
    calc_atr,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_volume_ratio,
)


class RsiMacdStrategy(BaseStrategy):
    """RSI + MACD confluence strategy.

    Parameters (read from self.params with defaults):
        rsi_period (int): RSI lookback period. Default 14.
        rsi_oversold (int): RSI threshold for long entry. Default 35.
        rsi_overbought (int): RSI threshold for short entry. Default 65.
        fast_period (int): MACD fast EMA period. Default 12.
        slow_period (int): MACD slow EMA period. Default 26.
        signal_period (int): MACD signal EMA period. Default 9.
        vol_multiplier (float): Volume ratio required to confirm signal. Default 1.3.
        sl_atr_mult (float): ATR multiplier for stop-loss distance. Default 2.0.
        tp1_atr_mult (float): ATR multiplier for first take-profit. Default 3.0.
        tp2_atr_mult (float): ATR multiplier for second take-profit. Default 5.0.
    """

    def get_name(self) -> str:
        """Return strategy identifier."""
        return "rsi_macd"

    def get_min_candles(self) -> int:
        """Return minimum candle count needed.

        Uses slow_period * 3 to ensure all indicators have sufficient warmup.
        """
        return self.params.get("slow_period", 26) * 3

    def get_timeframe(self) -> str:
        return "1h"

    def _validate_params(self) -> None:
        """Validate parameter relationships.

        Raises:
            ValueError: If any parameter constraint is violated.
        """
        oversold = self.params.get("rsi_oversold", 35)
        overbought = self.params.get("rsi_overbought", 65)
        sl_mult = self.params.get("sl_atr_mult", 2.0)
        tp1_mult = self.params.get("tp1_atr_mult", 3.0)

        if oversold >= overbought:
            raise ValueError(
                f"rsi_oversold({oversold}) must be less than rsi_overbought({overbought})"
            )
        if sl_mult <= 0:
            raise ValueError(f"sl_atr_mult({sl_mult}) must be > 0")
        if tp1_mult <= sl_mult:
            raise ValueError(
                f"tp1_atr_mult({tp1_mult}) must be greater than sl_atr_mult({sl_mult})"
            )

    def generate_signal(self, df, symbol: str) -> SignalResult:
        """Generate a long or short signal from RSI + MACD confluence.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume],
                sorted ascending by time.
            symbol: Trading pair identifier (used for reason string).

        Returns:
            :class:`SignalResult` with signal_type, strength_score, entry/tp/sl
            prices, indicator snapshot, and a human-readable reason.
        """
        # --- Load parameters ---
        rsi_period = self.params.get("rsi_period", 14)
        rsi_oversold = self.params.get("rsi_oversold", 35)
        rsi_overbought = self.params.get("rsi_overbought", 65)
        fast_period = self.params.get("fast_period", 12)
        slow_period = self.params.get("slow_period", 26)
        signal_period = self.params.get("signal_period", 9)
        vol_multiplier = self.params.get("vol_multiplier", 1.3)
        sl_atr_mult = self.params.get("sl_atr_mult", 2.0)
        tp1_atr_mult = self.params.get("tp1_atr_mult", 3.0)
        tp2_atr_mult = self.params.get("tp2_atr_mult", 5.0)

        # --- Compute indicators ---
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        rsi = calc_rsi(close, rsi_period)
        _, _, hist = calc_macd(close, fast_period, slow_period, signal_period)
        ema50 = calc_ema(close, 50)
        atr = calc_atr(high, low, close, 14)
        vol_ratio = calc_volume_ratio(volume, 20)

        # Need at least 2 bars of histogram to detect direction change
        if np.isnan(hist[-1]) or np.isnan(hist[-2]):
            return SignalResult(signal_type="none", reason="지표 워밍업 중")

        cur_rsi = float(rsi[-1])
        cur_hist = float(hist[-1])
        prev_hist = float(hist[-2])
        cur_close = float(close[-1])
        cur_ema50 = float(ema50[-1])
        cur_atr = float(atr[-1])
        cur_vol_ratio = float(vol_ratio[-1])

        # Guard: NaN in any required value
        if any(np.isnan(v) for v in [cur_rsi, cur_ema50, cur_atr, cur_vol_ratio]):
            return SignalResult(signal_type="none", reason="지표 계산값 NaN")

        indicators_snapshot = {
            "rsi": cur_rsi,
            "macd_hist": cur_hist,
            "atr": cur_atr,
            "vol_ratio": cur_vol_ratio,
        }

        # --- Evaluate long conditions ---
        long_conds = [
            cur_rsi < rsi_oversold,                   # 1. RSI oversold
            cur_hist > prev_hist,                     # 2. MACD hist turning up
            cur_close > cur_ema50,                    # 3. Price above EMA50
            cur_vol_ratio > vol_multiplier,           # 4. Volume confirmation
        ]

        # --- Evaluate short conditions ---
        short_conds = [
            cur_rsi > rsi_overbought,                 # 1. RSI overbought
            cur_hist < prev_hist,                     # 2. MACD hist turning down
            cur_close < cur_ema50,                    # 3. Price below EMA50
            cur_vol_ratio > vol_multiplier,           # 4. Volume confirmation
        ]

        long_score = min(sum(long_conds), 3)
        short_score = min(sum(short_conds), 3)

        # --- Determine direction (long takes priority on tie) ---
        if long_score >= short_score and long_score >= 2:
            entry = cur_close
            sl = entry - cur_atr * sl_atr_mult
            tp1 = entry + cur_atr * tp1_atr_mult
            tp2 = entry + cur_atr * tp2_atr_mult
            met = [i + 1 for i, c in enumerate(long_conds) if c]
            reason = f"롱 조건 충족: {met} | RSI={cur_rsi:.1f} hist={cur_hist:.4f}"
            return SignalResult(
                signal_type="long",
                strength_score=long_score,
                entry_price=entry,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                indicators=indicators_snapshot,
                reason=reason,
            )

        if short_score >= 2:
            entry = cur_close
            sl = entry + cur_atr * sl_atr_mult
            tp1 = entry - cur_atr * tp1_atr_mult
            tp2 = entry - cur_atr * tp2_atr_mult
            met = [i + 1 for i, c in enumerate(short_conds) if c]
            reason = f"숏 조건 충족: {met} | RSI={cur_rsi:.1f} hist={cur_hist:.4f}"
            return SignalResult(
                signal_type="short",
                strength_score=short_score,
                entry_price=entry,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                indicators=indicators_snapshot,
                reason=reason,
            )

        # --- Insufficient conditions ---
        reason = (
            f"조건 미충족 | 롱:{long_score}/4 숏:{short_score}/4 "
            f"RSI={cur_rsi:.1f} hist={cur_hist:.4f} vol={cur_vol_ratio:.2f}"
        )
        return SignalResult(
            signal_type="none",
            strength_score=0,
            indicators=indicators_snapshot,
            reason=reason,
        )
