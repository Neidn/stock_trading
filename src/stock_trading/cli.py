from __future__ import annotations

import argparse
from pathlib import Path

from stock_trading.alerts.console import format_screener_results, format_signals
from stock_trading.config import Settings, load_watchlist, resolve_project_path
from stock_trading.data.ingestion import ingest_bars
from stock_trading.data.providers import build_provider
from stock_trading.db import init_db, load_all_bars
from stock_trading.risk.sizing import RiskConfig
from stock_trading.screener.engine import passed_symbols, run_screener
from stock_trading.screener.rules import ScreenerConfig
from stock_trading.signals.generator import generate_signals
from stock_trading.strategies.registry import build_strategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock screener and signal CLI")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--watchlist", type=Path, default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db")

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--start", required=True)
    ingest.add_argument("--end", default=None)
    ingest.add_argument("--provider", default=None)
    ingest.add_argument("--interval", default=None)

    subparsers.add_parser("screen")

    signals = subparsers.add_parser("signals")
    signals.add_argument("--equity", type=float, required=True)
    signals.add_argument("--actionable-only", action="store_true")

    daily = subparsers.add_parser("run-daily")
    daily.add_argument("--start", required=True)
    daily.add_argument("--end", default=None)
    daily.add_argument("--equity", type=float, required=True)
    daily.add_argument("--provider", default=None)
    daily.add_argument("--interval", default=None)
    daily.add_argument("--actionable-only", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    db_path = args.db_path or settings.db_path
    watchlist_path = args.watchlist or settings.watchlist_path

    if args.command == "init-db":
        init_db(db_path)
        print(f"initialized database: {resolve_project_path(db_path)}")
        return 0

    watchlist = load_watchlist(watchlist_path)
    symbols = watchlist["symbols"]
    screener_config = ScreenerConfig.from_dict(watchlist.get("screener", {}))
    strategy_config = dict(watchlist.get("strategy", {}))
    active_strategy = str(strategy_config.get("active", "momentum_breakout"))
    risk_config = RiskConfig(
        risk_per_trade=settings.risk_per_trade,
        max_position_pct=settings.max_position_pct,
        max_open_positions=settings.max_open_positions,
    )

    if args.command in {"ingest", "run-daily"}:
        provider = build_provider(args.provider or settings.provider)
        interval = args.interval or settings.bar_interval
        init_db(db_path)
        count = ingest_bars(
            db_path=db_path,
            provider=provider,
            symbols=symbols,
            start=args.start,
            end=args.end,
            interval=interval,
        )
        print(f"ingested bars: {count}")
        if args.command == "ingest":
            return 0

    bars_by_symbol = load_all_bars(db_path, symbols)
    screen_results = run_screener(
        db_path=db_path,
        bars_by_symbol=bars_by_symbol,
        config=screener_config,
        persist=True,
    )

    if args.command == "screen":
        print(format_screener_results(screen_results))
        return 0

    screened = passed_symbols(screen_results)
    screened_bars = {symbol: bars_by_symbol[symbol] for symbol in screened}
    strategy = build_strategy(active_strategy)
    signals = generate_signals(
        db_path=db_path,
        strategy=strategy,
        bars_by_symbol=screened_bars,
        account_equity=args.equity,
        risk_config=risk_config,
        strategy_params=strategy_config,
        persist=True,
        actionable_only=args.actionable_only,
    )

    if args.command == "run-daily":
        print(format_screener_results(screen_results))
        print()
    print(format_signals(signals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
