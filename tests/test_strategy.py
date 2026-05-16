from datetime import datetime, timedelta

import pandas as pd

from stock_trading.models import SignalDirection, SignalStatus
from stock_trading.risk.sizing import RiskConfig
from stock_trading.strategies.base import StrategyContext
from stock_trading.strategies.momentum_breakout import MomentumBreakoutStrategy


def make_breakout_frame(rows: int = 60) -> pd.DataFrame:
    start = datetime(2025, 1, 1)
    records = []
    for index in range(rows):
        close = 100 + index * 0.2
        if index == rows - 1:
            close += 8
        records.append(
            {
                "timestamp": start + timedelta(days=index),
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1_000_000 if index < rows - 1 else 2_000_000,
            }
        )
    return pd.DataFrame(records)


def test_momentum_breakout_generates_buy_signal() -> None:
    strategy = MomentumBreakoutStrategy()
    signal = strategy.generate(
        "TEST",
        make_breakout_frame(),
        StrategyContext(
            account_equity=20_000,
            risk_config=RiskConfig(risk_per_trade=0.01, max_position_pct=0.10),
            params={"breakout_lookback": 20, "atr_period": 14},
        ),
    )

    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.action == "BUY"
    assert signal.status == SignalStatus.NEW
    assert signal.expiry == signal.as_of + timedelta(days=5)
    assert signal.risk is not None
    assert signal.risk.shares > 0
