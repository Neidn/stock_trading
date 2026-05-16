from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_trading.indicators import relative_volume, sma
from stock_trading.models import ScreenerResult


@dataclass(frozen=True)
class ScreenerConfig:
    min_price: float = 5.0
    min_avg_dollar_volume: float = 20_000_000
    min_relative_volume: float = 0.8
    require_uptrend: bool = True
    fast_sma: int = 50
    slow_sma: int = 200
    liquidity_lookback: int = 20

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "ScreenerConfig":
        return cls(
            min_price=float(values.get("min_price", cls.min_price)),
            min_avg_dollar_volume=float(values.get("min_avg_dollar_volume", cls.min_avg_dollar_volume)),
            min_relative_volume=float(values.get("min_relative_volume", cls.min_relative_volume)),
            require_uptrend=bool(values.get("require_uptrend", cls.require_uptrend)),
            fast_sma=int(values.get("fast_sma", cls.fast_sma)),
            slow_sma=int(values.get("slow_sma", cls.slow_sma)),
            liquidity_lookback=int(values.get("liquidity_lookback", cls.liquidity_lookback)),
        )


def evaluate_symbol(symbol: str, frame: pd.DataFrame, config: ScreenerConfig) -> ScreenerResult:
    if frame.empty:
        return ScreenerResult(symbol=symbol, as_of=pd.Timestamp.utcnow().to_pydatetime(), passed=False, score=0, reasons=["no data"])

    minimum_rows = max(config.slow_sma if config.require_uptrend else config.liquidity_lookback, config.liquidity_lookback) + 1
    latest = frame.iloc[-1]
    as_of = pd.Timestamp(latest["timestamp"]).to_pydatetime()
    reasons: list[str] = []
    score = 0.0

    if len(frame) < minimum_rows:
        return ScreenerResult(
            symbol=symbol,
            as_of=as_of,
            passed=False,
            score=0,
            reasons=[f"insufficient bars: {len(frame)} < {minimum_rows}"],
        )

    close = float(latest["close"])
    avg_dollar_volume = float((frame["close"] * frame["volume"]).tail(config.liquidity_lookback).mean())
    rel_volume = float(relative_volume(frame["volume"], config.liquidity_lookback).iloc[-1])
    fast = float(sma(frame["close"], config.fast_sma).iloc[-1])
    slow = float(sma(frame["close"], config.slow_sma).iloc[-1])

    if close >= config.min_price:
        score += 1
        reasons.append(f"price ok: {close:.2f}")
    else:
        reasons.append(f"price below minimum: {close:.2f} < {config.min_price:.2f}")

    if avg_dollar_volume >= config.min_avg_dollar_volume:
        score += 1
        reasons.append(f"liquidity ok: avg dollar volume {avg_dollar_volume:,.0f}")
    else:
        reasons.append(f"liquidity low: avg dollar volume {avg_dollar_volume:,.0f}")

    if rel_volume >= config.min_relative_volume:
        score += 1
        reasons.append(f"relative volume ok: {rel_volume:.2f}")
    else:
        reasons.append(f"relative volume low: {rel_volume:.2f}")

    trend_ok = close > fast > slow
    if not config.require_uptrend or trend_ok:
        score += 1
        reasons.append(f"trend ok: close {close:.2f}, SMA{config.fast_sma} {fast:.2f}, SMA{config.slow_sma} {slow:.2f}")
    else:
        reasons.append(f"trend weak: close {close:.2f}, SMA{config.fast_sma} {fast:.2f}, SMA{config.slow_sma} {slow:.2f}")

    required_score = 4 if config.require_uptrend else 3
    return ScreenerResult(
        symbol=symbol,
        as_of=as_of,
        passed=score >= required_score,
        score=score,
        reasons=reasons,
    )
