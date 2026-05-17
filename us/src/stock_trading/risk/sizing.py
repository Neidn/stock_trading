from __future__ import annotations

from dataclasses import dataclass

from stock_trading.models import RiskPlan


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.005
    max_position_pct: float = 0.10
    max_open_positions: int = 5


def calculate_stock_position(
    account_equity: float,
    entry: float,
    stop: float,
    target: float,
    config: RiskConfig,
) -> RiskPlan:
    if account_equity <= 0:
        raise ValueError("account_equity must be positive")
    if entry <= 0:
        raise ValueError("entry must be positive")
    if stop <= 0 or stop >= entry:
        raise ValueError("long stock stop must be below entry and positive")

    risk_budget = account_equity * config.risk_per_trade
    max_notional = account_equity * config.max_position_pct
    risk_per_share = entry - stop

    shares_by_risk = int(risk_budget // risk_per_share)
    shares_by_notional = int(max_notional // entry)
    shares = max(0, min(shares_by_risk, shares_by_notional))

    return RiskPlan(
        entry=round(entry, 4),
        stop=round(stop, 4),
        target=round(target, 4),
        risk_per_share=round(risk_per_share, 4),
        shares=shares,
        capital_at_risk=round(shares * risk_per_share, 2),
        notional=round(shares * entry, 2),
    )
