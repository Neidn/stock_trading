"""Unit tests for OrderManager.

All ccxt exchange calls are replaced with MagicMock.
DB uses in-memory SQLite with the orders schema.
Async tests use IsolatedAsyncioTestCase.
"""

from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from src.execution.order_manager import MAX_SLIPPAGE_PCT, OrderManager, OrderTimeoutError
from src.utils.config import TradingMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ORDERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id            TEXT PRIMARY KEY,
    binance_order_id    INTEGER,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL CHECK (side IN ('buy','sell')),
    position_side       TEXT NOT NULL CHECK (position_side IN ('long','short','both')),
    order_type          TEXT NOT NULL,
    price               TEXT,
    quantity            TEXT NOT NULL,
    filled_qty          TEXT NOT NULL DEFAULT '0',
    avg_fill_price      TEXT,
    status              TEXT NOT NULL,
    signal_id           TEXT,
    fee                 TEXT NOT NULL DEFAULT '0',
    fee_asset           TEXT,
    trading_mode        TEXT NOT NULL DEFAULT 'testnet'
                            CHECK (trading_mode IN ('testnet','live')),
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT
);
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_ORDERS_SCHEMA)
    conn.commit()
    return conn


def _ccxt_response(binance_id: int = 12345, order_type: str = "limit") -> dict:
    """Minimal ccxt create_order response."""
    return {
        "id": binance_id,
        "symbol": "BTCUSDT",
        "type": order_type,
        "side": "buy",
        "price": 50_000.0,
        "amount": 0.01,
        "filled": 0.0,
        "average": None,
        "status": "open",
        "fee": {"cost": 0.05, "currency": "USDT"},
    }


def _make_exchange(binance_id: int = 12345, order_type: str = "limit") -> MagicMock:
    ex = MagicMock()
    ex.create_order.return_value = _ccxt_response(binance_id, order_type)
    ex.cancel_order.return_value = {"id": binance_id, "status": "canceled"}
    return ex


def _make_config():
    cfg = MagicMock()
    cfg.trading_mode = TradingMode.TESTNET
    return cfg


def _make_manager(
    conn=None,
    exchange=None,
    order_stream=None,
    telegram_bot=None,
) -> OrderManager:
    return OrderManager(
        conn=conn or _make_conn(),
        exchange=exchange or _make_exchange(),
        order_stream=order_stream,
        telegram_bot=telegram_bot,
        config=_make_config(),
    )


def _order_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM orders ORDER BY created_at").fetchall()


# ---------------------------------------------------------------------------
# create_order
# ---------------------------------------------------------------------------

class TestCreateOrder(unittest.TestCase):

    def test_calls_exchange_create_order(self):
        ex = _make_exchange()
        mgr = _make_manager(exchange=ex)
        mgr.create_order("BTCUSDT", "buy", "limit", 0.01, price=50_000.0)
        ex.create_order.assert_called_once()

    def test_saves_row_to_db(self):
        conn = _make_conn()
        mgr = _make_manager(conn=conn)
        mgr.create_order("BTCUSDT", "buy", "limit", 0.01, price=50_000.0)
        rows = _order_rows(conn)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["symbol"], "BTCUSDT")
        self.assertEqual(row["side"], "buy")
        self.assertEqual(row["order_type"], "limit")
        self.assertEqual(row["quantity"], "0.01")
        self.assertEqual(row["status"], "open")
        self.assertEqual(row["trading_mode"], "testnet")

    def test_result_has_internal_order_id(self):
        mgr = _make_manager()
        result = mgr.create_order("BTCUSDT", "buy", "limit", 0.01, price=50_000.0)
        self.assertIn("internal_order_id", result)

    def test_market_order_passes_none_price(self):
        ex = _make_exchange(order_type="market")
        ex.create_order.return_value["type"] = "market"
        mgr = _make_manager(exchange=ex)
        mgr.create_order("BTCUSDT", "buy", "market", 0.01)
        args = ex.create_order.call_args
        # 5th positional arg is price — should be None for market
        self.assertIsNone(args[0][4])

    def test_stop_market_uses_stop_price_param(self):
        ex = _make_exchange(order_type="stop_market")
        ex.create_order.return_value["type"] = "STOP_MARKET"
        mgr = _make_manager(exchange=ex)
        mgr.create_order("BTCUSDT", "sell", "stop_market", 0.01, price=48_000.0)
        _, kwargs = ex.create_order.call_args
        params = ex.create_order.call_args[0][5]
        self.assertEqual(params["stopPrice"], 48_000.0)

    def test_position_side_forwarded_as_uppercase(self):
        ex = _make_exchange()
        mgr = _make_manager(exchange=ex)
        mgr.create_order("BTCUSDT", "buy", "limit", 0.01, price=50_000.0, position_side="long")
        params = ex.create_order.call_args[0][5]
        self.assertEqual(params["positionSide"], "LONG")


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------

class TestCancelOrder(unittest.TestCase):

    def test_calls_exchange_cancel_order(self):
        ex = _make_exchange()
        mgr = _make_manager(exchange=ex)
        mgr.cancel_order("BTCUSDT", "12345")
        ex.cancel_order.assert_called_once_with("12345", "BTCUSDT")

    def test_updates_db_status_to_canceled(self):
        conn = _make_conn()
        ex = _make_exchange(binance_id=99)
        mgr = _make_manager(conn=conn, exchange=ex)
        # Insert an order first
        mgr.create_order("BTCUSDT", "buy", "limit", 0.01, price=50_000.0)
        mgr.cancel_order("BTCUSDT", "99")
        row = conn.execute(
            "SELECT status FROM orders WHERE binance_order_id=99"
        ).fetchone()
        self.assertEqual(row["status"], "canceled")


# ---------------------------------------------------------------------------
# market_close
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# cancel_symbol_orders
# ---------------------------------------------------------------------------

class TestCancelSymbolOrders(unittest.TestCase):

    def test_calls_cancel_all_orders(self):
        ex = _make_exchange()
        mgr = _make_manager(exchange=ex)
        mgr.cancel_symbol_orders("BTCUSDT")
        ex.cancel_all_orders.assert_called_once_with("BTCUSDT")

    def test_swallows_exchange_error(self):
        ex = _make_exchange()
        ex.cancel_all_orders.side_effect = Exception("network error")
        mgr = _make_manager(exchange=ex)
        mgr.cancel_symbol_orders("BTCUSDT")  # must not raise


# ---------------------------------------------------------------------------
# market_close
# ---------------------------------------------------------------------------

class TestMarketClose(unittest.TestCase):

    def test_creates_market_order_with_correct_side(self):
        ex = _make_exchange(order_type="market")
        ex.create_order.return_value["type"] = "market"
        mgr = _make_manager(exchange=ex)
        mgr.market_close("BTCUSDT", "sell", 0.01)
        call_args = ex.create_order.call_args[0]
        self.assertEqual(call_args[2], "sell")   # side
        self.assertEqual(call_args[1], "market")  # order_type

    def test_cancels_symbol_orders_before_market_close(self):
        ex = _make_exchange(order_type="market")
        mgr = _make_manager(exchange=ex)
        mgr.market_close("BTCUSDT", "sell", 0.01)
        ex.cancel_all_orders.assert_called_once_with("BTCUSDT")


# ---------------------------------------------------------------------------
# submit_and_confirm
# ---------------------------------------------------------------------------

class TestSubmitAndConfirm(unittest.IsolatedAsyncioTestCase):

    async def test_fill_returned_from_stream(self):
        """Stream returns fill immediately → result matches fill dict."""
        fill = {"id": 12345, "average": 50_100.0, "filled": 0.01, "status": "filled"}
        stream = AsyncMock()
        stream.wait_for_fill.return_value = fill

        conn = _make_conn()
        mgr = _make_manager(conn=conn, order_stream=stream)
        order = {"symbol": "BTCUSDT", "side": "buy", "type": "limit",
                 "quantity": 0.01, "price": 50_000.0}
        result = await mgr.submit_and_confirm(order, timeout_sec=5)
        self.assertEqual(result["average"], 50_100.0)

    async def test_fill_updates_db_status(self):
        fill = {"id": 12345, "average": 50_100.0, "filled": 0.01, "status": "filled"}
        stream = AsyncMock()
        stream.wait_for_fill.return_value = fill

        conn = _make_conn()
        mgr = _make_manager(conn=conn, order_stream=stream)
        order = {"symbol": "BTCUSDT", "side": "buy", "type": "limit",
                 "quantity": 0.01, "price": 50_000.0}
        await mgr.submit_and_confirm(order, timeout_sec=5)
        row = conn.execute(
            "SELECT status, avg_fill_price FROM orders WHERE binance_order_id=12345"
        ).fetchone()
        self.assertEqual(row["status"], "filled")
        self.assertEqual(float(row["avg_fill_price"]), 50_100.0)

    async def test_telegram_notified_on_fill(self):
        fill = {"id": 12345, "average": 50_100.0, "filled": 0.01, "status": "filled"}
        stream = AsyncMock()
        stream.wait_for_fill.return_value = fill
        bot = MagicMock()

        mgr = _make_manager(order_stream=stream, telegram_bot=bot)
        order = {"symbol": "BTCUSDT", "side": "buy", "type": "limit",
                 "quantity": 0.01, "price": 50_000.0}
        await mgr.submit_and_confirm(order, timeout_sec=5)
        bot.send_alert.assert_called_once()
        self.assertIn("FILL", bot.send_alert.call_args[0][0])

    async def test_timeout_limit_order_retries_as_market(self):
        """Limit order timeout → cancel → retry with market order."""
        # First call (limit) times out, second call (market) fills
        call_count = 0
        original_id = 12345
        market_id = 99999

        def fake_create_order(symbol, order_type, side, qty, price=None, params=None):
            nonlocal call_count
            call_count += 1
            bid = original_id if call_count == 1 else market_id
            return {
                "id": bid,
                "type": order_type,
                "side": side,
                "price": price,
                "amount": qty,
                "filled": 0.0,
                "average": None,
                "status": "open",
                "fee": {"cost": 0.0, "currency": "USDT"},
            }

        ex = MagicMock()
        ex.create_order.side_effect = fake_create_order
        ex.cancel_order.return_value = {"id": original_id, "status": "canceled"}

        market_fill = {
            "id": market_id, "average": 50_050.0, "filled": 0.01, "status": "filled"
        }
        stream = AsyncMock()
        # First wait: timeout (None); second wait: fill
        stream.wait_for_fill.side_effect = [None, market_fill]

        conn = _make_conn()
        mgr = _make_manager(conn=conn, exchange=ex, order_stream=stream)
        order = {"symbol": "BTCUSDT", "side": "buy", "type": "limit",
                 "quantity": 0.01, "price": 50_000.0}
        result = await mgr.submit_and_confirm(order, timeout_sec=5)

        # Cancel called for the original limit order
        ex.cancel_order.assert_called_once_with(str(original_id), "BTCUSDT")
        # Market retry returned the fill
        self.assertEqual(result["average"], 50_050.0)
        # Two exchange.create_order calls: limit + market
        self.assertEqual(ex.create_order.call_count, 2)

    async def test_timeout_market_order_raises(self):
        """Market order also times out → OrderTimeoutError."""
        stream = AsyncMock()
        stream.wait_for_fill.return_value = None  # always timeout

        ex = _make_exchange(binance_id=11111, order_type="market")
        mgr = _make_manager(exchange=ex, order_stream=stream)
        order = {"symbol": "BTCUSDT", "side": "buy", "type": "market",
                 "quantity": 0.01}
        with self.assertRaises(OrderTimeoutError):
            await mgr.submit_and_confirm(order, timeout_sec=5)

    async def test_tp_sl_registered_after_fill(self):
        """If tp1/tp2/sl in order → register_tp_sl called after fill."""
        fill = {"id": 12345, "average": 50_100.0, "filled": 0.1, "status": "filled"}
        stream = AsyncMock()
        stream.wait_for_fill.return_value = fill

        # exchange must handle the 3 TP/SL create_order calls too
        call_count = [0]
        def fake_create(symbol, order_type, side, qty, price=None, params=None):
            call_count[0] += 1
            return {
                "id": 10000 + call_count[0],
                "type": order_type, "side": side,
                "price": price, "amount": qty,
                "filled": 0.0, "average": None,
                "status": "open",
                "fee": {"cost": 0.0, "currency": "USDT"},
            }

        ex = MagicMock()
        ex.create_order.side_effect = fake_create

        conn = _make_conn()
        mgr = _make_manager(conn=conn, exchange=ex, order_stream=stream)
        order = {
            "symbol": "BTCUSDT", "side": "buy", "type": "limit",
            "quantity": 0.1, "price": 50_000.0,
            "tp1": 52_000.0, "tp2": 54_000.0, "sl": 48_500.0,
        }
        await mgr.submit_and_confirm(order, timeout_sec=5)
        # 1 entry + 3 TP/SL = 4 total
        self.assertEqual(ex.create_order.call_count, 4)

    async def test_no_stream_returns_submitted_order(self):
        """order_stream=None → submitted order treated as filled immediately."""
        mgr = _make_manager(order_stream=None)
        order = {"symbol": "BTCUSDT", "side": "buy", "type": "market",
                 "quantity": 0.01}
        result = await mgr.submit_and_confirm(order)
        self.assertIn("id", result)


# ---------------------------------------------------------------------------
# register_tp_sl
# ---------------------------------------------------------------------------

class TestRegisterTpSl(unittest.TestCase):

    def test_creates_three_orders(self):
        conn = _make_conn()
        call_count = [0]

        def fake_create(symbol, order_type, side, qty, price=None, params=None):
            call_count[0] += 1
            return {
                "id": 20000 + call_count[0],
                "type": order_type, "side": side, "price": price,
                "amount": qty, "filled": 0.0, "average": None,
                "status": "open",
                "fee": {"cost": 0.0, "currency": "USDT"},
            }

        ex = MagicMock()
        ex.create_order.side_effect = fake_create
        mgr = _make_manager(conn=conn, exchange=ex)
        mgr.register_tp_sl("BTCUSDT", "buy", 0.1, tp1=52_000.0, tp2=54_000.0, sl=48_500.0)
        self.assertEqual(ex.create_order.call_count, 3)

    def test_half_quantity_for_tp1_tp2(self):
        """TP1 and TP2 each get quantity / 2."""
        quantities = []

        def fake_create(symbol, order_type, side, qty, price=None, params=None):
            quantities.append((order_type, qty))
            return {
                "id": len(quantities),
                "type": order_type, "side": side, "price": price,
                "amount": qty, "filled": 0.0, "average": None,
                "status": "open",
                "fee": {"cost": 0.0, "currency": "USDT"},
            }

        ex = MagicMock()
        ex.create_order.side_effect = fake_create
        mgr = _make_manager(exchange=ex)
        mgr.register_tp_sl("BTCUSDT", "buy", 0.2, tp1=52_000.0, tp2=54_000.0, sl=48_500.0)

        limit_qtys = [qty for ot, qty in quantities if ot == "limit"]
        stop_qtys = [qty for ot, qty in quantities if ot == "STOP_MARKET"]
        self.assertEqual(limit_qtys, [0.1, 0.1])   # 50 % each
        self.assertEqual(stop_qtys, [0.2])          # full qty

    def test_close_side_is_opposite_of_entry(self):
        """Long entry (buy) → close side is sell; short (sell) → close is buy."""
        close_sides = []

        def fake_create(symbol, order_type, side, qty, price=None, params=None):
            close_sides.append(side)
            return {
                "id": len(close_sides),
                "type": order_type, "side": side, "price": price,
                "amount": qty, "filled": 0.0, "average": None,
                "status": "open",
                "fee": {"cost": 0.0, "currency": "USDT"},
            }

        ex = MagicMock()
        ex.create_order.side_effect = fake_create

        # Long entry → close side = sell
        mgr = _make_manager(exchange=ex)
        mgr.register_tp_sl("BTCUSDT", "buy", 0.1, 52_000.0, 54_000.0, 48_500.0)
        self.assertTrue(all(s == "sell" for s in close_sides))

        close_sides.clear()

        # Short entry → close side = buy
        mgr.register_tp_sl("BTCUSDT", "sell", 0.1, 48_000.0, 46_000.0, 51_500.0)
        self.assertTrue(all(s == "buy" for s in close_sides))

    def test_stop_market_uses_sl_price(self):
        """SL price is passed via params['stopPrice'] for STOP_MARKET orders."""
        stop_prices = []

        def fake_create(symbol, order_type, side, qty, price=None, params=None):
            if order_type == "STOP_MARKET":
                stop_prices.append((params or {}).get("stopPrice"))
            return {
                "id": len(stop_prices) + 1,
                "type": order_type, "side": side, "price": price,
                "amount": qty, "filled": 0.0, "average": None,
                "status": "open",
                "fee": {"cost": 0.0, "currency": "USDT"},
            }

        ex = MagicMock()
        ex.create_order.side_effect = fake_create
        mgr = _make_manager(exchange=ex)
        mgr.register_tp_sl("BTCUSDT", "buy", 0.1, 52_000.0, 54_000.0, 48_500.0)

        self.assertEqual(len(stop_prices), 1)
        self.assertEqual(stop_prices[0], 48_500.0)

    def test_tp_sl_db_rows_saved(self):
        conn = _make_conn()
        call_count = [0]

        def fake_create(symbol, order_type, side, qty, price=None, params=None):
            call_count[0] += 1
            return {
                "id": 30000 + call_count[0],
                "type": order_type, "side": side, "price": price,
                "amount": qty, "filled": 0.0, "average": None,
                "status": "open",
                "fee": {"cost": 0.0, "currency": "USDT"},
            }

        ex = MagicMock()
        ex.create_order.side_effect = fake_create
        mgr = _make_manager(conn=conn, exchange=ex)
        mgr.register_tp_sl("BTCUSDT", "buy", 0.1, 52_000.0, 54_000.0, 48_500.0)
        rows = _order_rows(conn)
        self.assertEqual(len(rows), 3)


# ---------------------------------------------------------------------------
# check_slippage
# ---------------------------------------------------------------------------

class TestCheckSlippage(unittest.TestCase):

    def test_buy_within_limit_returns_true(self):
        expected = 50_000.0
        actual = expected * (1 + MAX_SLIPPAGE_PCT * 0.5)  # half of limit
        self.assertTrue(OrderManager.check_slippage(expected, actual, "buy"))

    def test_buy_exactly_at_limit_returns_true(self):
        expected = 50_000.0
        actual = expected * (1 + MAX_SLIPPAGE_PCT)
        self.assertTrue(OrderManager.check_slippage(expected, actual, "buy"))

    def test_buy_exceeds_limit_returns_false(self):
        expected = 50_000.0
        actual = expected * (1 + MAX_SLIPPAGE_PCT + 0.001)
        self.assertFalse(OrderManager.check_slippage(expected, actual, "buy"))

    def test_sell_within_limit_returns_true(self):
        expected = 50_000.0
        actual = expected * (1 - MAX_SLIPPAGE_PCT * 0.5)
        self.assertTrue(OrderManager.check_slippage(expected, actual, "sell"))

    def test_sell_exactly_at_limit_returns_true(self):
        expected = 50_000.0
        actual = expected * (1 - MAX_SLIPPAGE_PCT)
        self.assertTrue(OrderManager.check_slippage(expected, actual, "sell"))

    def test_sell_exceeds_limit_returns_false(self):
        expected = 50_000.0
        actual = expected * (1 - MAX_SLIPPAGE_PCT - 0.001)
        self.assertFalse(OrderManager.check_slippage(expected, actual, "sell"))

    def test_buy_favourable_price_returns_true(self):
        """Fill below expected → always good for buy."""
        self.assertTrue(OrderManager.check_slippage(50_000.0, 49_900.0, "buy"))

    def test_sell_favourable_price_returns_true(self):
        """Fill above expected → always good for sell."""
        self.assertTrue(OrderManager.check_slippage(50_000.0, 50_200.0, "sell"))


if __name__ == "__main__":
    unittest.main()
