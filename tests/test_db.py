import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from stock_trading.db import init_db, insert_signals, load_bars, upsert_bars
from stock_trading.models import Signal, SignalDirection, SignalStatus


def test_upsert_and_load_bars_with_limit(tmp_path) -> None:
    db_path = tmp_path / "stock_trading.db"
    init_db(db_path)

    start = datetime(2025, 1, 1)
    frame = pd.DataFrame(
        [
            {
                "timestamp": start + timedelta(days=index),
                "open": 10 + index,
                "high": 11 + index,
                "low": 9 + index,
                "close": 10.5 + index,
                "volume": 1000 + index,
            }
            for index in range(5)
        ]
    )

    inserted = upsert_bars(db_path, {"TEST": frame})
    loaded = load_bars(db_path, "TEST", limit=2)

    assert inserted == 5
    assert len(loaded) == 2
    assert loaded.iloc[0]["close"] == 13.5
    assert loaded.iloc[1]["close"] == 14.5


def test_insert_signals_persists_state_fields(tmp_path) -> None:
    db_path = tmp_path / "stock_trading.db"
    init_db(db_path)
    expiry = datetime(2025, 1, 6)
    signal = Signal(
        symbol="test",
        strategy="unit",
        as_of=datetime(2025, 1, 1),
        direction=SignalDirection.BUY,
        status=SignalStatus.NEW,
        expiry=expiry,
        confidence=0.9,
        reason="unit test",
    )

    inserted = insert_signals(db_path, [signal])

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT action, direction, status, expiry FROM signals WHERE symbol = ?",
            ("TEST",),
        ).fetchone()

    assert inserted == 1
    assert row == ("BUY", "buy", "new", expiry.isoformat())


def test_init_db_migrates_legacy_signals_table(tmp_path) -> None:
    db_path = tmp_path / "stock_trading.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                as_of TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                entry REAL,
                stop REAL,
                target REAL,
                shares INTEGER,
                capital_at_risk REAL,
                notional REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO signals
                (symbol, strategy, as_of, action, confidence, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("TEST", "legacy", "2025-01-01T00:00:00", "WATCH", 0.5, "legacy row", "2025-01-01T00:00:00"),
        )

    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(signals)").fetchall()}
        row = connection.execute("SELECT direction, status, expiry FROM signals WHERE symbol = ?", ("TEST",)).fetchone()

    assert {"direction", "status", "expiry"}.issubset(columns)
    assert row == ("hold", "watching", None)
