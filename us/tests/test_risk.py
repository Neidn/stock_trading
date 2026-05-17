from stock_trading.risk.sizing import RiskConfig, calculate_stock_position


def test_calculate_stock_position_caps_by_risk_and_notional() -> None:
    plan = calculate_stock_position(
        account_equity=10_000,
        entry=100,
        stop=95,
        target=110,
        config=RiskConfig(risk_per_trade=0.01, max_position_pct=0.10),
    )

    assert plan.shares == 10
    assert plan.capital_at_risk == 50
    assert plan.notional == 1000
