from __future__ import annotations

from pathlib import Path

from stock_trading.data.providers import MarketDataProvider
from stock_trading.db import upsert_bars


def ingest_bars(
    db_path: Path,
    provider: MarketDataProvider,
    symbols: list[str],
    start: str,
    end: str | None,
    interval: str,
) -> int:
    bars = provider.fetch_daily_bars(symbols=symbols, start=start, end=end, interval=interval)
    return upsert_bars(db_path, bars)
