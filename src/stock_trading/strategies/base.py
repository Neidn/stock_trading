from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from stock_trading.models import Signal
from stock_trading.risk.sizing import RiskConfig


@dataclass(frozen=True)
class StrategyContext:
    account_equity: float
    risk_config: RiskConfig
    params: dict[str, object]


class BaseStrategy(ABC):
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, symbol: str, frame: pd.DataFrame, context: StrategyContext) -> Signal | None:
        raise NotImplementedError
