from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    db_path: Path
    watchlist_path: Path
    provider: str
    bar_interval: str
    risk_per_trade: float
    max_position_pct: float
    max_open_positions: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=Path(os.getenv("STOCK_TRADING_DB_PATH", "data/stock_trading.db")),
            watchlist_path=Path(os.getenv("STOCK_TRADING_WATCHLIST", "config/watchlist.yml")),
            provider=os.getenv("STOCK_TRADING_PROVIDER", "yfinance"),
            bar_interval=os.getenv("STOCK_TRADING_BAR_INTERVAL", "1d"),
            risk_per_trade=float(os.getenv("STOCK_TRADING_RISK_PER_TRADE", "0.005")),
            max_position_pct=float(os.getenv("STOCK_TRADING_MAX_POSITION_PCT", "0.10")),
            max_open_positions=int(os.getenv("STOCK_TRADING_MAX_OPEN_POSITIONS", "5")),
        )


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_watchlist(path: Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"watchlist file not found: {resolved}")

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load watchlist files. Run: pip install -e .") from exc

    with resolved.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    symbols = loaded.get("symbols") or []
    normalized = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    if not normalized:
        raise ValueError("watchlist must contain at least one symbol")

    loaded["symbols"] = normalized
    loaded.setdefault("screener", {})
    loaded.setdefault("strategy", {})
    return loaded
