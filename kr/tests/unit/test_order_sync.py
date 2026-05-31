"""Tests for KRX OrderSyncJob.

KIS API calls are mocked. DB uses in-memory SQLite with broker_order_id schema.
"""

from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import AsyncMock, MagicMock

from src.jobs.order_sync import OrderSyncJob


_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id          TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4)))),
    broker_order_id   TEXT,
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
    trading_mode      TEXT NOT NULL DEFAULT 'paper',
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


def _insert_order(conn: sqlite3.Connection, broker_id: str | None, symbol: str = "005930") -> None:
    conn.execute(
        "INSERT INTO orders (order_id, broker_order_id, symbol, quantity) VALUES (?,?,?,?)",
        (f"ord-{broker_id}", broker_id, symbol, "10"),
    )
    conn.commit()


def _make_kis(open_order_nos: list[str]) -> MagicMock:
    kis = MagicMock()
    orders = [{"order_no": ono, "symbol": "005930"} for ono in open_order_nos]
    kis.fetch_unfilled_orders = AsyncMock(return_value=orders)
    return kis


class TestOrderSyncJob(unittest.TestCase):

    def test_stale_order_marked_canceled(self):
        conn = _make_conn()
        _insert_order(conn, "KIS001")
        kis = _make_kis([])  # nothing on KIS
        result = OrderSyncJob(kis=kis, conn=conn).run()
        self.assertEqual(result["canceled"], 1)
        row = conn.execute("SELECT status FROM orders WHERE broker_order_id='KIS001'").fetchone()
        self.assertEqual(row["status"], "canceled")

    def test_live_order_kept(self):
        conn = _make_conn()
        _insert_order(conn, "KIS002")
        kis = _make_kis(["KIS002"])
        result = OrderSyncJob(kis=kis, conn=conn).run()
        self.assertEqual(result["kept"], 1)
        self.assertEqual(result["canceled"], 0)
        row = conn.execute("SELECT status FROM orders WHERE broker_order_id='KIS002'").fetchone()
        self.assertEqual(row["status"], "open")

    def test_mixed_orders(self):
        conn = _make_conn()
        _insert_order(conn, "K301", "005930")  # stale
        _insert_order(conn, "K302", "005930")  # live
        _insert_order(conn, "K303", "000660")  # stale, different symbol
        kis = _make_kis(["K302"])
        result = OrderSyncJob(kis=kis, conn=conn).run()
        self.assertEqual(result["canceled"], 2)
        self.assertEqual(result["kept"], 1)

    def test_already_canceled_orders_not_touched(self):
        conn = _make_conn()
        _insert_order(conn, "K401")
        conn.execute("UPDATE orders SET status='canceled' WHERE broker_order_id='K401'")
        conn.commit()
        kis = _make_kis([])
        result = OrderSyncJob(kis=kis, conn=conn).run()
        self.assertEqual(result["canceled"], 0)

    def test_null_broker_id_marked_canceled(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO orders (order_id, broker_order_id, symbol, quantity) VALUES ('x', NULL, '005930', '10')"
        )
        conn.commit()
        kis = _make_kis([])
        result = OrderSyncJob(kis=kis, conn=conn).run()
        self.assertEqual(result["canceled"], 1)

    def test_kis_api_error_returns_error_dict(self):
        conn = _make_conn()
        kis = MagicMock()
        kis.fetch_unfilled_orders = AsyncMock(side_effect=RuntimeError("KIS API down"))
        result = OrderSyncJob(kis=kis, conn=conn).run()
        self.assertEqual(result["canceled"], 0)
        self.assertIn("KIS API down", result["error"])

    def test_empty_orders_table(self):
        conn = _make_conn()
        kis = _make_kis([])
        result = OrderSyncJob(kis=kis, conn=conn).run()
        self.assertEqual(result["canceled"], 0)
        self.assertEqual(result["kept"], 0)


if __name__ == "__main__":
    unittest.main()
