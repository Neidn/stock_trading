"""Tests for OrderSyncJob."""

from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import MagicMock

from src.jobs.order_sync import OrderSyncJob


_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id          TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4)))),
    binance_order_id  INTEGER,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL DEFAULT 'buy',
    position_side     TEXT NOT NULL DEFAULT 'both',
    order_type        TEXT NOT NULL DEFAULT 'limit',
    price             TEXT,
    quantity          TEXT NOT NULL DEFAULT '0',
    filled_qty        TEXT NOT NULL DEFAULT '0',
    avg_fill_price    TEXT,
    status            TEXT NOT NULL DEFAULT 'open',
    fee               TEXT NOT NULL DEFAULT '0',
    fee_asset         TEXT,
    trading_mode      TEXT NOT NULL DEFAULT 'testnet',
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT
);
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _insert_order(conn: sqlite3.Connection, binance_id: int, symbol: str = "BTCUSDT") -> None:
    conn.execute(
        "INSERT INTO orders (order_id, binance_order_id, symbol, quantity) VALUES (?,?,?,?)",
        (f"ord-{binance_id}", binance_id, symbol, "0.01"),
    )
    conn.commit()


def _make_exchange(open_order_ids: list[int]) -> MagicMock:
    ex = MagicMock()
    ex.fetch_open_orders.return_value = [{"id": oid} for oid in open_order_ids]
    return ex


def _make_exchange_per_symbol(symbol_to_ids: dict[str, list[int]]) -> MagicMock:
    """Exchange that returns different orders per symbol (per-symbol fetch)."""
    ex = MagicMock()
    def _fetch(symbol=None):
        if symbol is None:
            return []
        # Convert ccxt format back to DB symbol for lookup
        db_sym = symbol.replace("/USDT:USDT", "USDT").replace("/", "")
        ids = symbol_to_ids.get(db_sym, [])
        return [{"id": oid} for oid in ids]
    ex.fetch_open_orders.side_effect = _fetch
    return ex


class TestOrderSyncJob(unittest.TestCase):

    def test_stale_order_marked_canceled(self):
        conn = _make_conn()
        _insert_order(conn, 101, "BTCUSDT")
        ex = _make_exchange_per_symbol({"BTCUSDT": []})  # nothing on Binance
        result = OrderSyncJob(exchange=ex, conn=conn).run()
        self.assertEqual(result["canceled"], 1)
        row = conn.execute("SELECT status FROM orders WHERE binance_order_id=101").fetchone()
        self.assertEqual(row["status"], "canceled")

    def test_live_order_kept(self):
        conn = _make_conn()
        _insert_order(conn, 202, "BTCUSDT")
        ex = _make_exchange_per_symbol({"BTCUSDT": [202]})
        result = OrderSyncJob(exchange=ex, conn=conn).run()
        self.assertEqual(result["kept"], 1)
        self.assertEqual(result["canceled"], 0)
        row = conn.execute("SELECT status FROM orders WHERE binance_order_id=202").fetchone()
        self.assertEqual(row["status"], "open")

    def test_mixed_orders(self):
        conn = _make_conn()
        _insert_order(conn, 301, "BTCUSDT")  # stale
        _insert_order(conn, 302, "BTCUSDT")  # live
        _insert_order(conn, 303, "ETHUSDT")  # stale, different symbol
        ex = _make_exchange_per_symbol({"BTCUSDT": [302], "ETHUSDT": []})
        result = OrderSyncJob(exchange=ex, conn=conn).run()
        self.assertEqual(result["canceled"], 2)
        self.assertEqual(result["kept"], 1)

    def test_stop_market_order_detected_per_symbol(self):
        """Per-symbol fetch returns STOP_MARKET orders that no-symbol fetch misses."""
        conn = _make_conn()
        _insert_order(conn, 3000001623843742, "BEATUSDT")  # SL order, large ID
        ex = _make_exchange_per_symbol({"BEATUSDT": [3000001623843742]})
        result = OrderSyncJob(exchange=ex, conn=conn).run()
        self.assertEqual(result["kept"], 1)
        self.assertEqual(result["canceled"], 0)

    def test_already_canceled_orders_not_touched(self):
        conn = _make_conn()
        _insert_order(conn, 401, "BTCUSDT")
        conn.execute("UPDATE orders SET status='canceled' WHERE binance_order_id=401")
        conn.commit()
        ex = _make_exchange_per_symbol({})
        result = OrderSyncJob(exchange=ex, conn=conn).run()
        self.assertEqual(result["canceled"], 0)

    def test_per_symbol_fetch_error_skipped(self):
        """Symbol fetch failure skips that symbol — others still processed."""
        conn = _make_conn()
        _insert_order(conn, 501, "BTCUSDT")
        _insert_order(conn, 502, "ETHUSDT")
        ex = MagicMock()
        def _fetch(symbol=None):
            if "BTC" in (symbol or ""):
                raise Exception("rate limit")
            return []
        ex.fetch_open_orders.side_effect = _fetch
        result = OrderSyncJob(exchange=ex, conn=conn).run()
        # BTCUSDT skipped (error) → kept; ETHUSDT fetched, not on Binance → canceled
        self.assertEqual(result["canceled"], 1)
        self.assertEqual(result["kept"], 1)

    def test_null_binance_id_marked_canceled(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO orders (order_id, binance_order_id, symbol, quantity) VALUES ('x', NULL, 'BTCUSDT', '0.01')"
        )
        conn.commit()
        ex = _make_exchange_per_symbol({"BTCUSDT": []})
        result = OrderSyncJob(exchange=ex, conn=conn).run()
        self.assertEqual(result["canceled"], 1)
