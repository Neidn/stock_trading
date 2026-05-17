from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_trading.config import resolve_project_path
from stock_trading.models import ScreenerResult, Signal


SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS screener_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    as_of TEXT NOT NULL,
    passed INTEGER NOT NULL,
    score REAL NOT NULL,
    reasons_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    as_of TEXT NOT NULL,
    action TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT 'hold',
    status TEXT NOT NULL DEFAULT 'new',
    expiry TEXT,
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    entry REAL,
    stop REAL,
    target REAL,
    shares INTEGER,
    capital_at_risk REAL,
    notional REAL,
    created_at TEXT NOT NULL
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    resolved = resolve_project_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved)
    connection.row_factory = sqlite3.Row
    return connection


def _migrate_signal_columns(connection: sqlite3.Connection) -> None:
    table_info = connection.execute("PRAGMA table_info(signals)").fetchall()
    if not table_info:
        return

    columns = {row["name"] for row in table_info}
    migrations = {
        "direction": "ALTER TABLE signals ADD COLUMN direction TEXT NOT NULL DEFAULT 'hold'",
        "status": "ALTER TABLE signals ADD COLUMN status TEXT NOT NULL DEFAULT 'new'",
        "expiry": "ALTER TABLE signals ADD COLUMN expiry TEXT",
    }
    for column, statement in migrations.items():
        if column not in columns:
            connection.execute(statement)

    connection.execute(
        """
        UPDATE signals
        SET direction = CASE lower(action)
            WHEN 'buy' THEN 'buy'
            WHEN 'sell' THEN 'sell'
            WHEN 'hold' THEN 'hold'
            WHEN 'watch' THEN 'hold'
            WHEN 'exit_watch' THEN 'exit_watch'
            WHEN 'exit-watch' THEN 'exit_watch'
            ELSE direction
        END
        WHERE action IS NOT NULL
        """
    )
    connection.execute(
        """
        UPDATE signals
        SET status = CASE lower(action)
            WHEN 'watch' THEN 'watching'
            ELSE status
        END
        WHERE action IS NOT NULL
        """
    )


def init_db(db_path: Path) -> None:
    with connect(db_path) as connection:
        connection.executescript(SCHEMA)
        _migrate_signal_columns(connection)


def upsert_bars(db_path: Path, bars_by_symbol: dict[str, pd.DataFrame]) -> int:
    rows: list[tuple[str, str, float, float, float, float, float]] = []
    for symbol, frame in bars_by_symbol.items():
        if frame.empty:
            continue
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{symbol} bars missing columns: {sorted(missing)}")

        for record in frame.to_dict("records"):
            timestamp = pd.Timestamp(record["timestamp"]).to_pydatetime().isoformat()
            rows.append(
                (
                    symbol.upper(),
                    timestamp,
                    float(record["open"]),
                    float(record["high"]),
                    float(record["low"]),
                    float(record["close"]),
                    float(record["volume"]),
                )
            )

    if not rows:
        return 0

    with connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO bars (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timestamp) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
            """,
            rows,
        )
    return len(rows)


def load_bars(db_path: Path, symbol: str, limit: int | None = None) -> pd.DataFrame:
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM bars
        WHERE symbol = ?
        ORDER BY timestamp ASC
    """
    with connect(db_path) as connection:
        frame = pd.read_sql_query(query, connection, params=(symbol.upper(),), parse_dates=["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if limit:
        return frame.tail(limit).reset_index(drop=True)
    return frame


def load_all_bars(db_path: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    return {symbol.upper(): load_bars(db_path, symbol.upper()) for symbol in symbols}


def insert_screener_results(db_path: Path, results: Iterable[ScreenerResult]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            result.symbol.upper(),
            result.as_of.isoformat(),
            int(result.passed),
            float(result.score),
            json.dumps(result.reasons),
            now,
        )
        for result in results
    ]
    if not rows:
        return 0

    with connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO screener_results
                (symbol, as_of, passed, score, reasons_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def insert_signals(db_path: Path, signals: Iterable[Signal]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for signal in signals:
        risk = signal.risk
        rows.append(
            (
                signal.symbol.upper(),
                signal.strategy,
                signal.as_of.isoformat(),
                signal.action,
                signal.direction.value,
                signal.status.value,
                signal.expiry.isoformat() if signal.expiry else None,
                float(signal.confidence),
                signal.reason,
                risk.entry if risk else None,
                risk.stop if risk else None,
                risk.target if risk else None,
                risk.shares if risk else None,
                risk.capital_at_risk if risk else None,
                risk.notional if risk else None,
                now,
            )
        )
    if not rows:
        return 0

    with connect(db_path) as connection:
        _migrate_signal_columns(connection)
        connection.executemany(
            """
            INSERT INTO signals
                (symbol, strategy, as_of, action, direction, status, expiry,
                 confidence, reason, entry, stop, target, shares, capital_at_risk,
                 notional, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)
