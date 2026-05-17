from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class SignalDirection(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT_WATCH = "exit_watch"


class SignalStatus(StrEnum):
    NEW = "new"
    WATCHING = "watching"
    ENTERED_MANUAL = "entered_manual"
    CLOSED_MANUAL = "closed_manual"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ScreenerResult:
    symbol: str
    as_of: datetime
    passed: bool
    score: float
    reasons: list[str]


@dataclass(frozen=True)
class RiskPlan:
    entry: float
    stop: float
    target: float
    risk_per_share: float
    shares: int
    capital_at_risk: float
    notional: float


@dataclass(frozen=True)
class Signal:
    symbol: str
    strategy: str
    as_of: datetime
    direction: SignalDirection
    confidence: float
    reason: str
    status: SignalStatus = SignalStatus.NEW
    expiry: datetime | None = None
    risk: RiskPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", SignalDirection(self.direction))
        object.__setattr__(self, "status", SignalStatus(self.status))

    @property
    def action(self) -> str:
        return self.direction.value.upper()
