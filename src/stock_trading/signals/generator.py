from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_trading.db import insert_signals
from stock_trading.models import Signal, SignalDirection
from stock_trading.risk.sizing import RiskConfig
from stock_trading.strategies.base import BaseStrategy, StrategyContext


def generate_signals(
    db_path: Path,
    strategy: BaseStrategy,
    bars_by_symbol: dict[str, pd.DataFrame],
    account_equity: float,
    risk_config: RiskConfig,
    strategy_params: dict[str, object],
    persist: bool = True,
    actionable_only: bool = False,
) -> list[Signal]:
    context = StrategyContext(
        account_equity=account_equity,
        risk_config=risk_config,
        params=strategy_params,
    )
    signals: list[Signal] = []
    for symbol, frame in sorted(bars_by_symbol.items()):
        signal = strategy.generate(symbol, frame, context)
        if signal is None:
            continue
        if actionable_only and signal.direction not in {
            SignalDirection.BUY,
            SignalDirection.SELL,
            SignalDirection.EXIT_WATCH,
        }:
            continue
        signals.append(signal)

    if persist:
        insert_signals(db_path, signals)
    return signals
