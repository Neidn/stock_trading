from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_daily_bars(
        self,
        symbols: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError


class YFinanceProvider(MarketDataProvider):
    def fetch_daily_bars(
        self,
        symbols: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        bars: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            bars[symbol] = normalize_ohlcv(symbol, frame)
        return bars


class CsvProvider(MarketDataProvider):
    def __init__(self, csv_dir: Path) -> None:
        self.csv_dir = csv_dir

    def fetch_daily_bars(
        self,
        symbols: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        bars: dict[str, pd.DataFrame] = {}
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else None

        for symbol in symbols:
            path = self.csv_dir / f"{symbol.upper()}.csv"
            frame = pd.read_csv(path)
            frame = normalize_ohlcv(symbol, frame)
            frame = frame[frame["timestamp"] >= start_ts]
            if end_ts is not None:
                frame = frame[frame["timestamp"] < end_ts]
            bars[symbol.upper()] = frame.reset_index(drop=True)
        return bars


def normalize_ohlcv(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [column[0] for column in normalized.columns]

    if "Date" in normalized.columns:
        normalized = normalized.rename(columns={"Date": "timestamp"})
    elif "Datetime" in normalized.columns:
        normalized = normalized.rename(columns={"Datetime": "timestamp"})
    elif "timestamp" not in normalized.columns:
        normalized = normalized.reset_index().rename(columns={"index": "timestamp"})

    column_map = {column: str(column).strip().lower().replace(" ", "_") for column in normalized.columns}
    normalized = normalized.rename(columns=column_map)
    aliases = {
        "date": "timestamp",
        "adj_close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "timestamp": "timestamp",
    }
    normalized = normalized.rename(columns={key: value for key, value in aliases.items() if key in normalized.columns})

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"{symbol} OHLCV data missing columns: {missing}")

    normalized = normalized[required].dropna()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna().sort_values("timestamp").reset_index(drop=True)
    return normalized


def build_provider(name: str) -> MarketDataProvider:
    provider = name.lower().strip()
    if provider == "yfinance":
        return YFinanceProvider()
    if provider == "csv":
        return CsvProvider(Path("data/raw"))
    raise ValueError(f"unsupported market data provider: {name}")
