from __future__ import annotations

from datetime import timedelta

import pandas as pd

from stock_trading.indicators import average_true_range, relative_volume, rolling_high
from stock_trading.models import Signal, SignalDirection, SignalStatus
from stock_trading.risk.sizing import calculate_stock_position
from stock_trading.strategies.base import BaseStrategy, StrategyContext


class MomentumBreakoutStrategy(BaseStrategy):
    def get_name(self) -> str:
        return "momentum_breakout"

    def generate(self, symbol: str, frame: pd.DataFrame, context: StrategyContext) -> Signal | None:
        lookback = int(context.params.get("breakout_lookback", 20))
        atr_period = int(context.params.get("atr_period", 14))
        atr_stop_multiple = float(context.params.get("atr_stop_multiple", 2.0))
        reward_risk = float(context.params.get("reward_risk", 2.0))
        signal_expiry_days = int(context.params.get("signal_expiry_days", 5))

        required_rows = max(lookback, atr_period, 20) + 1
        if len(frame) < required_rows:
            return None

        latest = frame.iloc[-1]
        close = float(latest["close"])
        previous_high = float(rolling_high(frame["high"], lookback).shift(1).iloc[-1])
        atr = float(average_true_range(frame, atr_period).iloc[-1])
        rel_volume = float(relative_volume(frame["volume"], 20).iloc[-1])

        if pd.isna(previous_high) or pd.isna(atr) or atr <= 0:
            return None

        as_of = pd.Timestamp(latest["timestamp"]).to_pydatetime()
        if close <= previous_high:
            return Signal(
                symbol=symbol,
                strategy=self.get_name(),
                as_of=as_of,
                direction=SignalDirection.HOLD,
                confidence=0.0,
                reason=f"no breakout: close {close:.2f} <= prior {lookback}-bar high {previous_high:.2f}",
            )

        stop = max(0.01, close - (atr * atr_stop_multiple))
        target = close + ((close - stop) * reward_risk)
        risk_plan = calculate_stock_position(
            account_equity=context.account_equity,
            entry=close,
            stop=stop,
            target=target,
            config=context.risk_config,
        )
        confidence = min(1.0, 0.65 + max(0.0, rel_volume - 1.0) * 0.15)
        if risk_plan.shares <= 0:
            return Signal(
                symbol=symbol,
                strategy=self.get_name(),
                as_of=as_of,
                direction=SignalDirection.HOLD,
                status=SignalStatus.CANCELLED,
                confidence=0.0,
                reason=(
                    f"breakout above prior {lookback}-bar high {previous_high:.2f}; "
                    "risk limits allow 0 shares"
                ),
                risk=risk_plan,
            )

        return Signal(
            symbol=symbol,
            strategy=self.get_name(),
            as_of=as_of,
            direction=SignalDirection.BUY,
            confidence=round(confidence, 3),
            reason=f"breakout above prior {lookback}-bar high {previous_high:.2f}; relative volume {rel_volume:.2f}",
            expiry=as_of + timedelta(days=signal_expiry_days),
            risk=risk_plan,
        )
