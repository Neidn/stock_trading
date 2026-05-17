from __future__ import annotations

from stock_trading.strategies.base import BaseStrategy
from stock_trading.strategies.momentum_breakout import MomentumBreakoutStrategy


def build_strategy(name: str) -> BaseStrategy:
    normalized = name.strip().lower()
    strategies: dict[str, BaseStrategy] = {
        "momentum_breakout": MomentumBreakoutStrategy(),
    }
    try:
        return strategies[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(strategies))
        raise ValueError(f"unknown strategy {name!r}; available: {available}") from exc
