from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_trading.db import insert_screener_results
from stock_trading.models import ScreenerResult
from stock_trading.screener.rules import ScreenerConfig, evaluate_symbol


def run_screener(
    db_path: Path,
    bars_by_symbol: dict[str, pd.DataFrame],
    config: ScreenerConfig,
    persist: bool = True,
) -> list[ScreenerResult]:
    results = [
        evaluate_symbol(symbol=symbol, frame=frame, config=config)
        for symbol, frame in sorted(bars_by_symbol.items())
    ]
    if persist:
        insert_screener_results(db_path, results)
    return results


def passed_symbols(results: list[ScreenerResult]) -> list[str]:
    return [result.symbol for result in results if result.passed]
