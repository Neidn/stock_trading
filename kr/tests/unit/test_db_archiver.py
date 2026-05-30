"""Tests for DbArchiverJob rolling-window pruning."""

from __future__ import annotations

import sqlite3

import pytest

from src.jobs.db_archiver import DbArchiverJob, KEEP_CANDLES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE klines (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            interval_type TEXT NOT NULL,
            open_time TEXT NOT NULL,
            open TEXT, high TEXT, low TEXT, close TEXT,
            volume TEXT, close_time TEXT,
            UNIQUE (symbol, interval_type, open_time)
        )"""
    )
    conn.commit()
    return conn


_BASE_TS = 1_700_000_000_000  # 13-digit ms epoch — text sort == numeric sort


def _insert_klines(conn, symbol: str, interval: str, count: int, start: int = 1) -> None:
    """Insert `count` rows with open_time = BASE_TS + start, BASE_TS + start+1, ..."""
    rows = [
        (
            f"{symbol}_{interval}_{i}",
            symbol, interval,
            str(_BASE_TS + i),
            "1", "2", "0.5", "1.5", "100",
            str(_BASE_TS + i),
        )
        for i in range(start, start + count)
    ]
    conn.executemany(
        "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Basic pruning
# ---------------------------------------------------------------------------

def test_prunes_to_keep_limit():
    conn = _make_db()
    _insert_klines(conn, "BTCUSDT", "1h", 700)
    job = DbArchiverJob(conn, keep_candles=500)
    result = job.run()

    remaining = conn.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol='BTCUSDT' AND interval_type='1h'"
    ).fetchone()[0]
    assert remaining == 500
    assert result["deleted_rows"] == 200


def test_no_delete_when_below_limit():
    conn = _make_db()
    _insert_klines(conn, "BTCUSDT", "1h", 300)
    job = DbArchiverJob(conn, keep_candles=500)
    result = job.run()

    remaining = conn.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
    assert remaining == 300
    assert result["deleted_rows"] == 0


def test_no_delete_when_exactly_at_limit():
    conn = _make_db()
    _insert_klines(conn, "BTCUSDT", "1h", 500)
    job = DbArchiverJob(conn, keep_candles=500)
    result = job.run()

    assert result["deleted_rows"] == 0


# ---------------------------------------------------------------------------
# Multiple symbols and intervals
# ---------------------------------------------------------------------------

def test_prunes_independently_per_symbol():
    conn = _make_db()
    _insert_klines(conn, "BTCUSDT", "1h", 700)
    _insert_klines(conn, "ETHUSDT", "1h", 200)
    job = DbArchiverJob(conn, keep_candles=500)
    result = job.run()

    btc = conn.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol='BTCUSDT'"
    ).fetchone()[0]
    eth = conn.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol='ETHUSDT'"
    ).fetchone()[0]
    assert btc == 500
    assert eth == 200          # untouched — below limit
    assert result["deleted_rows"] == 200


def test_prunes_independently_per_interval():
    conn = _make_db()
    _insert_klines(conn, "BTCUSDT", "1h", 700)
    _insert_klines(conn, "BTCUSDT", "4h", 600)
    job = DbArchiverJob(conn, keep_candles=500)
    result = job.run()

    h1 = conn.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol='BTCUSDT' AND interval_type='1h'"
    ).fetchone()[0]
    h4 = conn.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol='BTCUSDT' AND interval_type='4h'"
    ).fetchone()[0]
    assert h1 == 500
    assert h4 == 500
    assert result["deleted_rows"] == 300  # 200 + 100


def test_keeps_newest_candles():
    """Newest open_time values must survive pruning."""
    conn = _make_db()
    _insert_klines(conn, "BTCUSDT", "1h", 700, start=1)
    job = DbArchiverJob(conn, keep_candles=500)
    job.run()

    min_remaining = conn.execute(
        "SELECT MIN(CAST(open_time AS INTEGER)) FROM klines"
    ).fetchone()[0]
    # open_times BASE+1..BASE+700, keep newest 500 → min should be BASE+201
    assert min_remaining == _BASE_TS + 201


# ---------------------------------------------------------------------------
# Empty table
# ---------------------------------------------------------------------------

def test_empty_table():
    conn = _make_db()
    job = DbArchiverJob(conn, keep_candles=500)
    result = job.run()
    assert result["deleted_rows"] == 0
    assert result["vacuumed"] is True


# ---------------------------------------------------------------------------
# Default constant
# ---------------------------------------------------------------------------

def test_default_keep_candles():
    assert KEEP_CANDLES == 500
