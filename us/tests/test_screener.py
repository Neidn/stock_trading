from datetime import datetime, timedelta

import pandas as pd

from stock_trading.screener.rules import ScreenerConfig, evaluate_symbol


def make_trending_frame(rows: int = 220) -> pd.DataFrame:
    start = datetime(2025, 1, 1)
    records = []
    for index in range(rows):
        close = 50 + index * 0.4
        records.append(
            {
                "timestamp": start + timedelta(days=index),
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1_000_000 + index * 1000,
            }
        )
    return pd.DataFrame(records)


def test_evaluate_symbol_passes_liquid_uptrend() -> None:
    result = evaluate_symbol("TEST", make_trending_frame(), ScreenerConfig(min_avg_dollar_volume=1_000_000))

    assert result.passed is True
    assert result.score == 4
