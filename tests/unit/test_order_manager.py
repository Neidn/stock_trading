"""Unit tests for KRX OrderManager.

KIS API calls are replaced with AsyncMock.
DB uses in-memory SQLite.
Async tests use IsolatedAsyncioTestCase.
"""

from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.execution.order_manager import MAX_SLIPPAGE_PCT, OrderManager, OrderTimeoutError, round_to_tick
from src.utils.config import TradingMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ORDERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    broker_order_id TEXT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL CHECK (side IN ('buy','sell')),
    position_side   TEXT NOT NULL DEFAULT 'both',
    order_type      TEXT NOT NULL,
    price           TEXT,
    quantity        TEXT NOT NULL,
    filled_qty      TEXT NOT NULL DEFAULT '0',
    avg_fill_price  TEXT,
    status          TEXT NOT NULL DEFAULT 'open',
    fee             TEXT NOT NULL DEFAULT '0',
    fee_asset       TEXT,
    trading_mode    TEXT NOT NULL DEFAULT 'paper',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    position_id       TEXT PRIMARY KEY,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL DEFAULT 'long',
    leverage          INTEGER NOT NULL DEFAULT 1,
    entry_price       TEXT NOT NULL DEFAULT '0',
    exit_price        TEXT,
    quantity          TEXT NOT NULL DEFAULT '0',
    liquidation_price TEXT NOT NULL DEFAULT '0',
    stop_loss         TEXT NOT NULL DEFAULT '0',
    take_profit_1     TEXT,
    take_profit_2     TEXT,
    initial_stop_loss TEXT NOT NULL DEFAULT '0',
    trailing_activated INTEGER DEFAULT 0,
    realized_pnl      TEXT DEFAULT '0',
    unrealized_pnl    TEXT DEFAULT '0',
    status            TEXT NOT NULL DEFAULT 'open',
    close_reason      TEXT,
    trading_mode      TEXT NOT NULL DEFAULT 'paper',
    strategy_name     TEXT,
    opened_at         TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at         TEXT
);
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_ORDERS_SCHEMA)
    conn.commit()
    return conn


def _make_kis(broker_id: str = "KIS12345") -> MagicMock:
    kis = MagicMock()
    kis.place_buy_order = AsyncMock(
        return_value={"odno": broker_id, "KRX_FWDG_ORD_ORGNO": "ORG001"}
    )
    kis.place_sell_order = AsyncMock(
        return_value={"odno": broker_id, "KRX_FWDG_ORD_ORGNO": "ORG001"}
    )
    kis.fetch_unfilled_orders = AsyncMock(return_value=[])
    kis.cancel_order = AsyncMock(return_value={})
    return kis


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.trading_mode = TradingMode.PAPER
    cfg.risk_per_trade = 0.01
    return cfg


def _make_manager(conn=None, kis=None, telegram_bot=None) -> OrderManager:
    return OrderManager(
        conn=conn or _make_conn(),
        kis=kis or _make_kis(),
        telegram_bot=telegram_bot,
        config=_make_config(),
    )


# ---------------------------------------------------------------------------
# round_to_tick
# ---------------------------------------------------------------------------

class TestRoundToTick(unittest.TestCase):

    def test_below_1000(self):
        self.assertEqual(round_to_tick(999), 999)

    def test_1000_to_4999(self):
        self.assertEqual(round_to_tick(1234), 1230)

    def test_5000_to_9999(self):
        self.assertEqual(round_to_tick(7777), 7770)

    def test_10000_to_49999(self):
        self.assertEqual(round_to_tick(12345), 12300)

    def test_50000_to_99999(self):
        self.assertEqual(round_to_tick(80050), 80000)

    def test_100000_to_499999(self):
        self.assertEqual(round_to_tick(123456), 123000)

    def test_500000_plus(self):
        self.assertEqual(round_to_tick(750500), 750000)


# ---------------------------------------------------------------------------
# create_order
# ---------------------------------------------------------------------------

class TestCreateOrder(unittest.IsolatedAsyncioTestCase):

    async def test_saves_row_to_db(self):
        conn = _make_conn()
        kis = _make_kis("KIS001")
        mgr = _make_manager(conn=conn, kis=kis)
        await mgr.create_order("005930", "buy", 10, price=80000)
        row = conn.execute("SELECT * FROM orders").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["symbol"], "005930")
        self.assertEqual(row["side"], "buy")
        self.assertEqual(row["quantity"], "10")
        self.assertEqual(row["status"], "open")
        self.assertEqual(row["trading_mode"], "paper")

    async def test_broker_order_id_stored(self):
        conn = _make_conn()
        kis = _make_kis("KISORD99")
        mgr = _make_manager(conn=conn, kis=kis)
        await mgr.create_order("005930", "buy", 5, price=70000)
        row = conn.execute("SELECT broker_order_id FROM orders").fetchone()
        self.assertEqual(row["broker_order_id"], "KISORD99")

    async def test_result_has_internal_order_id(self):
        mgr = _make_manager()
        result = await mgr.create_order("005930", "buy", 10, price=80000)
        self.assertIn("internal_order_id", result)

    async def test_market_order_when_no_price(self):
        conn = _make_conn()
        mgr = _make_manager(conn=conn)
        await mgr.create_order("005930", "sell", 5, price=None)
        row = conn.execute("SELECT order_type FROM orders").fetchone()
        self.assertEqual(row["order_type"], "market")

    async def test_limit_price_rounded_to_tick(self):
        kis = _make_kis()
        mgr = _make_manager(kis=kis)
        # 80050 → tick-rounded to 80000 (100원 tick in 50000-99999 range)
        await mgr.create_order("005930", "buy", 10, price=80050)
        price_arg = kis.place_buy_order.call_args[0][2]
        self.assertEqual(price_arg, 80000)

    async def test_sell_calls_place_sell_order(self):
        kis = _make_kis()
        mgr = _make_manager(kis=kis)
        await mgr.create_order("005930", "sell", 5, price=79000)
        kis.place_sell_order.assert_called_once()
        kis.place_buy_order.assert_not_called()


# ---------------------------------------------------------------------------
# cancel_symbol_orders
# ---------------------------------------------------------------------------

class TestCancelSymbolOrders(unittest.IsolatedAsyncioTestCase):

    async def test_cancels_pending_order_for_symbol(self):
        kis = _make_kis()
        pending = [{
            "symbol": "005930", "order_no": "K001", "krx_orgno": "ORG1",
            "qty": 10, "filled_qty": 0, "ord_dvsn": "00",
        }]
        kis.fetch_unfilled_orders = AsyncMock(return_value=pending)
        mgr = _make_manager(kis=kis)
        await mgr.cancel_symbol_orders("005930")
        kis.cancel_order.assert_called_once()

    async def test_skips_other_symbols(self):
        kis = _make_kis()
        pending = [{
            "symbol": "000660", "order_no": "K002", "krx_orgno": "ORG1",
            "qty": 5, "filled_qty": 0, "ord_dvsn": "00",
        }]
        kis.fetch_unfilled_orders = AsyncMock(return_value=pending)
        mgr = _make_manager(kis=kis)
        await mgr.cancel_symbol_orders("005930")  # different symbol
        kis.cancel_order.assert_not_called()

    async def test_swallows_fetch_error(self):
        kis = _make_kis()
        kis.fetch_unfilled_orders = AsyncMock(side_effect=Exception("network error"))
        mgr = _make_manager(kis=kis)
        await mgr.cancel_symbol_orders("005930")  # must not raise

    async def test_skip_fully_filled_orders(self):
        kis = _make_kis()
        pending = [{
            "symbol": "005930", "order_no": "K003", "krx_orgno": "ORG1",
            "qty": 10, "filled_qty": 10, "ord_dvsn": "00",
        }]
        kis.fetch_unfilled_orders = AsyncMock(return_value=pending)
        mgr = _make_manager(kis=kis)
        await mgr.cancel_symbol_orders("005930")
        kis.cancel_order.assert_not_called()


# ---------------------------------------------------------------------------
# market_close
# ---------------------------------------------------------------------------

class TestMarketClose(unittest.IsolatedAsyncioTestCase):

    async def test_submits_market_sell(self):
        conn = _make_conn()
        kis = _make_kis()
        mgr = _make_manager(conn=conn, kis=kis)
        with patch("asyncio.sleep", new=AsyncMock()):
            await mgr.market_close("005930", 10)
        kis.place_sell_order.assert_called_once()

    async def test_cancels_pending_before_close(self):
        conn = _make_conn()
        kis = _make_kis()
        pending = [{
            "symbol": "005930", "order_no": "K001", "krx_orgno": "ORG1",
            "qty": 5, "filled_qty": 0, "ord_dvsn": "00",
        }]
        kis.fetch_unfilled_orders = AsyncMock(return_value=pending)
        mgr = _make_manager(conn=conn, kis=kis)
        with patch("asyncio.sleep", new=AsyncMock()):
            await mgr.market_close("005930", 10)
        kis.cancel_order.assert_called_once()


# ---------------------------------------------------------------------------
# submit_and_confirm (market order path)
# ---------------------------------------------------------------------------

class TestSubmitAndConfirm(unittest.IsolatedAsyncioTestCase):

    async def test_market_order_confirmed(self):
        conn = _make_conn()
        kis = _make_kis()
        mgr = _make_manager(conn=conn, kis=kis)
        order = {"symbol": "005930", "side": "buy", "quantity": 10}
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await mgr.submit_and_confirm(order)
        self.assertTrue(result.get("confirmed"))

    async def test_telegram_notified_on_fill(self):
        conn = _make_conn()
        kis = _make_kis()
        bot = MagicMock()
        mgr = OrderManager(conn=conn, kis=kis, telegram_bot=bot, config=_make_config())
        order = {"symbol": "005930", "side": "buy", "quantity": 10}
        with patch("asyncio.sleep", new=AsyncMock()):
            await mgr.submit_and_confirm(order)
        bot.send_alert.assert_called_once()


# ---------------------------------------------------------------------------
# check_slippage
# ---------------------------------------------------------------------------

class TestCheckSlippage(unittest.TestCase):

    def test_buy_within_limit_returns_true(self):
        expected = 50_000.0
        actual = expected * (1 + MAX_SLIPPAGE_PCT * 0.5)
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
        self.assertTrue(OrderManager.check_slippage(50_000.0, 49_900.0, "buy"))

    def test_sell_favourable_price_returns_true(self):
        self.assertTrue(OrderManager.check_slippage(50_000.0, 50_200.0, "sell"))


if __name__ == "__main__":
    unittest.main()
