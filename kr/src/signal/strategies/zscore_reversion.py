"""Z-score mean-reversion strategy.

Enters long when price is statistically oversold (z < -threshold) and short
when overbought (z > +threshold), but only in ranging markets (ADX <= max_adx).
Mean-reversion is invalid in trending conditions; ADX acts as the market-regime
filter that blocks signals automatically.
"""

from __future__ import annotations

import numpy as np

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_adx, calc_atr, calc_sma, calc_zscore


class ZscoreReversionStrategy(BaseStrategy):
    """Z-score mean-reversion strategy.

    Only operates in ranging (sideways) markets. ADX above ``max_adx``
    suppresses all signals to avoid fading a strong trend.

    Parameters (read from self.params with defaults):
        window (int): Rolling window for Z-score and SMA calculation. Default 20.
        zscore_threshold (float): Absolute Z-score required to trigger entry. Default 2.0.
        max_adx (float): ADX ceiling — signals blocked above this value. Default 25.
        sl_atr_mult (float): ATR multiplier for stop-loss. Default 1.5.
        tp1_atr_mult (float): ATR multiplier for tp1 (informational; tp1 uses mean). Default 2.0.
        tp2_atr_mult (float): ATR multiplier for second take-profit. Default 4.0.
    """

    def get_name(self) -> str:
        """Return strategy identifier."""
        return "zscore_reversion"

    def get_min_candles(self) -> int:
        """Return minimum candle count: window * 3 for ADX + Z-score warmup."""
        return self.params.get("window", 20) * 3

    def get_timeframe(self) -> str:
        return "1h"

    def _validate_params(self) -> None:
        """Validate parameter constraints.

        Raises:
            ValueError: If any constraint is violated.
        """
        threshold = self.params.get("zscore_threshold", 2.0)
        max_adx = self.params.get("max_adx", 25)
        sl_mult = self.params.get("sl_atr_mult", 1.5)
        tp2_mult = self.params.get("tp2_atr_mult", 4.0)

        if threshold <= 0:
            raise ValueError(f"zscore_threshold({threshold}) must be > 0")
        if max_adx <= 0:
            raise ValueError(f"max_adx({max_adx}) must be > 0")
        if sl_mult <= 0:
            raise ValueError(f"sl_atr_mult({sl_mult}) must be > 0")
        if tp2_mult <= sl_mult:
            raise ValueError(
                f"tp2_atr_mult({tp2_mult}) must be greater than sl_atr_mult({sl_mult})"
            )

    def generate_signal(self, df, symbol: str) -> SignalResult:
        """Generate a mean-reversion signal based on Z-score and ADX filter.

        Logic:
            1. If ADX > max_adx → trending market, return 'none'.
            2. If |zscore| < threshold → no extreme deviation, return 'none'.
            3. z < -threshold → long (price below mean, expect reversion up).
            4. z > +threshold → short (price above mean, expect reversion down).
            tp1 is always set to the rolling mean (reversion target).

        Args:
            df: OHLCV DataFrame sorted ascending by time.
            symbol: Trading pair identifier.

        Returns:
            :class:`SignalResult` with strength_score always 3 when actionable.
        """
        # --- Load parameters ---
        window = self.params.get("window", 20)
        zscore_threshold = self.params.get("zscore_threshold", 2.0)
        max_adx = self.params.get("max_adx", 25)
        sl_atr_mult = self.params.get("sl_atr_mult", 1.5)
        tp2_atr_mult = self.params.get("tp2_atr_mult", 4.0)

        # --- Compute indicators ---
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        zscore = calc_zscore(close, window)
        mean = calc_sma(close, window)
        atr = calc_atr(high, low, close, 14)
        adx = calc_adx(high, low, close, 14)

        cur_zscore = float(zscore[-1])
        cur_mean = float(mean[-1])
        cur_atr = float(atr[-1])
        cur_adx = float(adx[-1])
        cur_close = float(close[-1])

        # Guard: NaN in any required indicator
        if any(np.isnan(v) for v in [cur_zscore, cur_mean, cur_atr, cur_adx]):
            return SignalResult(signal_type="none", reason="지표 워밍업 중")

        indicators_snapshot = {
            "zscore": cur_zscore,
            "adx": cur_adx,
            "atr": cur_atr,
            "mean": cur_mean,
        }

        # --- 1. Market-regime filter: block signals in trending markets ---
        if cur_adx > max_adx:
            reason = (
                f"추세장 감지 — 신호 차단 | ADX={cur_adx:.1f} > max_adx={max_adx}"
            )
            return SignalResult(
                signal_type="none",
                indicators=indicators_snapshot,
                reason=reason,
            )

        # --- 2. Threshold filter: no extreme deviation ---
        if abs(cur_zscore) < zscore_threshold:
            reason = (
                f"Z-score 임계값 미달 | z={cur_zscore:.2f} "
                f"(threshold=±{zscore_threshold})"
            )
            return SignalResult(
                signal_type="none",
                indicators=indicators_snapshot,
                reason=reason,
            )

        # --- 3. Long: price statistically oversold, expect reversion upward ---
        if cur_zscore < -zscore_threshold:
            entry = cur_close
            tp1 = cur_mean                            # reversion target = mean
            tp2 = entry + cur_atr * tp2_atr_mult
            sl = entry - cur_atr * sl_atr_mult
            reason = (
                f"Z-score 과매도 진입 | z={cur_zscore:.2f} "
                f"ADX={cur_adx:.1f} mean={cur_mean:.2f}"
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

        # --- 4. Short: price statistically overbought, expect reversion downward ---
        entry = cur_close
        tp1 = cur_mean                                # reversion target = mean
        tp2 = entry - cur_atr * tp2_atr_mult
        sl = entry + cur_atr * sl_atr_mult
        reason = (
            f"Z-score 과매수 진입 | z={cur_zscore:.2f} "
            f"ADX={cur_adx:.1f} mean={cur_mean:.2f}"
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
