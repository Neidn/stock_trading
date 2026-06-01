"""Bollinger Band squeeze-and-breakout strategy.

Waits for a volatility squeeze (band width narrows below its recent average),
then enters in the direction of the first clean breakout above the upper band
(long) or below the lower band (short), confirmed by elevated volume.
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_atr, calc_bollinger, calc_volume_ratio


class BbBreakoutStrategy(BaseStrategy):
    """Bollinger Band squeeze-and-breakout strategy.

    Parameters (read from self.params with defaults):
        bb_period (int): Bollinger Band SMA period. Default 20.
        bb_std (float): Standard deviation multiplier for bands. Default 2.0.
        squeeze_window (int): Number of bars used to compute average band width
            for squeeze detection. Default 20.
        squeeze_pct (float): Current band width must be below
            ``avg_width * squeeze_pct`` to qualify as a squeeze. Default 0.8.
        vol_multiplier (float): Volume ratio required to confirm a breakout.
            Default 1.5.
        sl_atr_mult (float): ATR multiplier for stop-loss. Default 1.5.
        tp1_atr_mult (float): ATR multiplier for first take-profit. Default 2.5.
        tp2_atr_mult (float): ATR multiplier for second take-profit. Default 4.5.
    """

    def get_name(self) -> str:
        """Return strategy identifier."""
        return "bb_breakout"

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        return frozenset({"ranging", "volatile"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        """Score high when ATR% is low (squeeze setup) and ADX is moderate.

        Moderate SMA50 slope (not flat, not extreme) suggests directional bias
        is building — ideal pre-breakout condition.
        """
        atr_pct     = float(indicators.get("atr_pct", 1.0))
        adx         = float(indicators.get("adx", 25))
        sma50_slope = float(indicators.get("sma50_slope", 0.0))
        adx_change  = float(indicators.get("adx_change", 0.0))
        squeeze_score = max(0.0, 1.0 - atr_pct / 2.0)
        trend_score   = min(adx / 30.0, 1.0) * max(0.0, 1.0 - (adx - 30) / 30.0)
        base = (squeeze_score + trend_score) / 2.0
        # Moderate slope = direction building but not yet trending
        slope_norm   = abs(sma50_slope) / 0.02
        slope_bonus  = 0.08 if 0.05 < slope_norm < 0.5 else 0.0
        momentum_bonus = 0.05 if 0 < adx_change <= 3 else 0.0
        return round(min(base + slope_bonus + momentum_bonus, 1.0), 4)

    def get_min_candles(self) -> int:
        """Return minimum candle count: bb_period + squeeze_window + 10."""
        bb_period = self.params.get("bb_period", 20)
        squeeze_window = self.params.get("squeeze_window", 20)
        return bb_period + squeeze_window + 10

    def get_timeframe(self) -> str:
        return "1d"

    def _validate_params(self) -> None:
        """Validate parameter constraints.

        Raises:
            ValueError: If any constraint is violated.
        """
        sl_mult = self.params.get("sl_atr_mult", 1.5)
        tp1_mult = self.params.get("tp1_atr_mult", 2.5)
        tp2_mult = self.params.get("tp2_atr_mult", 4.5)
        squeeze_pct = self.params.get("squeeze_pct", 0.8)

        if sl_mult <= 0:
            raise ValueError(f"sl_atr_mult({sl_mult}) must be > 0")
        if tp1_mult <= sl_mult:
            raise ValueError(
                f"tp1_atr_mult({tp1_mult}) must be greater than sl_atr_mult({sl_mult})"
            )
        if tp2_mult <= tp1_mult:
            raise ValueError(
                f"tp2_atr_mult({tp2_mult}) must be greater than tp1_atr_mult({tp1_mult})"
            )
        if not (0 < squeeze_pct < 1):
            raise ValueError(f"squeeze_pct({squeeze_pct}) must be between 0 and 1")

    def generate_signal(self, df, symbol: str) -> SignalResult:
        """Generate a breakout signal after a Bollinger Band squeeze.

        Logic:
            1. Compute band width array = (upper - lower) / middle.
            2. Squeeze detected if current width < rolling mean of last
               ``squeeze_window`` widths × ``squeeze_pct``.
            3. No squeeze → return 'none'.
            4. Squeeze + upper breakout (first bar above band) + volume → long.
            5. Squeeze + lower breakout (first bar below band) + volume → short.
            6. Squeeze but no clean breakout or insufficient volume → 'none'.

        Args:
            df: OHLCV DataFrame sorted ascending by time.
            symbol: Trading pair identifier.

        Returns:
            :class:`SignalResult` with strength_score=3 when actionable.
        """
        # --- Load parameters ---
        bb_period = self.params.get("bb_period", 20)
        bb_std = self.params.get("bb_std", 2.0)
        squeeze_window = self.params.get("squeeze_window", 20)
        squeeze_pct = self.params.get("squeeze_pct", 0.8)
        vol_multiplier = self.params.get("vol_multiplier", 1.5)
        sl_atr_mult = self.params.get("sl_atr_mult", 1.5)
        tp1_atr_mult = self.params.get("tp1_atr_mult", 2.5)
        tp2_atr_mult = self.params.get("tp2_atr_mult", 4.5)

        # --- Compute indicators ---
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        upper, middle, lower = calc_bollinger(close, bb_period, bb_std)
        atr = calc_atr(high, low, close, 14)
        vol_ratio = calc_volume_ratio(volume, 20)

        # Need at least 2 bars of bands + squeeze_window bars of width history
        required = squeeze_window + 1
        if np.isnan(upper[-required:]).any() or np.isnan(middle[-required:]).any():
            return SignalResult(signal_type="none", reason="지표 워밍업 중")

        cur_close = float(close[-1])
        prev_close = float(close[-2])
        cur_upper = float(upper[-1])
        cur_lower = float(lower[-1])
        prev_upper = float(upper[-2])
        prev_lower = float(lower[-2])
        cur_atr = float(atr[-1])
        cur_vol_ratio = float(vol_ratio[-1])

        if any(np.isnan(v) for v in [cur_atr, cur_vol_ratio]):
            return SignalResult(signal_type="none", reason="지표 계산값 NaN")

        # --- Band width array (vectorized, no loops) ---
        # Use only the tail we need: squeeze_window bars ending at [-1]
        bb_width_arr = (upper - lower) / np.where(middle == 0, np.nan, middle)
        cur_bb_width = bb_width_arr[-1]
        # Average over the last squeeze_window bars (excluding the current bar)
        avg_bb_width = float(np.nanmean(bb_width_arr[-(squeeze_window + 1):-1]))

        if np.isnan(cur_bb_width) or np.isnan(avg_bb_width):
            return SignalResult(signal_type="none", reason="밴드폭 계산 불가")

        # --- 1. Squeeze detection ---
        squeeze_detected = cur_bb_width < avg_bb_width * squeeze_pct

        indicators_snapshot = {
            "bb_width": float(cur_bb_width),
            "squeeze_detected": squeeze_detected,
            "upper": cur_upper,
            "lower": cur_lower,
            "atr": cur_atr,
        }

        if not squeeze_detected:
            reason = (
                f"스퀴즈 미감지 | width={cur_bb_width:.4f} "
                f"avg={avg_bb_width:.4f} 기준={avg_bb_width * squeeze_pct:.4f}"
            )
            return SignalResult(
                signal_type="none",
                indicators=indicators_snapshot,
                reason=reason,
            )

        # --- 2. Long: first bar to close above upper band ---
        long_breakout = cur_close > cur_upper and prev_close <= prev_upper
        # --- 3. Short: first bar to close below lower band ---
        short_breakout = cur_close < cur_lower and prev_close >= prev_lower

        vol_ok = cur_vol_ratio > vol_multiplier

        if long_breakout and vol_ok:
            entry = cur_close
            sl = entry - cur_atr * sl_atr_mult
            tp1 = entry + cur_atr * tp1_atr_mult
            tp2 = entry + cur_atr * tp2_atr_mult
            reason = (
                f"BB 상단 돌파 (스퀴즈 후) | "
                f"close={cur_close:.2f} upper={cur_upper:.2f} "
                f"vol_ratio={cur_vol_ratio:.2f}"
            )
            return SignalResult(
                signal_type="long",
                strength_score=3,
                entry_price=entry,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                indicators=indicators_snapshot,
                reason=reason,
            )

        if short_breakout and vol_ok:
            entry = cur_close
            sl = entry + cur_atr * sl_atr_mult
            tp1 = entry - cur_atr * tp1_atr_mult
            tp2 = entry - cur_atr * tp2_atr_mult
            reason = (
                f"BB 하단 돌파 (스퀴즈 후) | "
                f"close={cur_close:.2f} lower={cur_lower:.2f} "
                f"vol_ratio={cur_vol_ratio:.2f}"
            )
            return SignalResult(
                signal_type="short",
                strength_score=3,
                entry_price=entry,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                indicators=indicators_snapshot,
                reason=reason,
            )

        # Squeeze detected but no clean breakout or volume insufficient
        if long_breakout or short_breakout:
            reason = f"돌파 감지 but 거래량 부족 | vol_ratio={cur_vol_ratio:.2f} < {vol_multiplier}"
        else:
            reason = f"스퀴즈 감지 but 돌파 없음 | width={cur_bb_width:.4f}"

        return SignalResult(
            signal_type="none",
            indicators=indicators_snapshot,
            reason=reason,
        )
