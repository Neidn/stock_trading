"""Order execution layer — wraps KISRestClient for KRX spot trading.

Responsibilities:
  - Submit buy/sell orders via KISRestClient (async)
  - Persist every order to the ``orders`` table
  - Confirm fills (market orders: 2s delay; limit orders: poll up to timeout)
  - Cancel pending limit orders on timeout
  - KRX tick-size rounding for limit prices
  - No native TP/SL on KRX — price monitoring is handled by safety_monitor
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.utils.config import load_config

if TYPE_CHECKING:
    import sqlite3
    from src.ingest.kis_rest import KISRestClient

logger = logging.getLogger(__name__)

MAX_SLIPPAGE_PCT: float = 0.003   # 0.3 %
MARKET_FILL_WAIT_SEC: int = 3     # wait after market order before assuming fill
LIMIT_POLL_INTERVAL_SEC: int = 3  # seconds between fill-status polls


# ---------------------------------------------------------------------------
# KRX tick-size rules (호가단위)
# ---------------------------------------------------------------------------

def round_to_tick(price: float) -> int:
    """Round *price* down to the nearest valid KRX 호가단위 (tick boundary).

    KRX tick sizes by price range (원):
        < 1,000          →    1원
        1,000 – 4,999    →    5원
        5,000 – 9,999    →   10원
        10,000 – 49,999  →   50원
        50,000 – 99,999  →  100원
        100,000 – 499,999 → 500원
        ≥ 500,000        → 1,000원
    """
    p = int(price)
    if p < 1_000:
        return p
    if p < 5_000:
        return (p // 5) * 5
    if p < 10_000:
        return (p // 10) * 10
    if p < 50_000:
        return (p // 50) * 50
    if p < 100_000:
        return (p // 100) * 100
    if p < 500_000:
        return (p // 500) * 500
    return (p // 1_000) * 1_000


class OrderTimeoutError(Exception):
    """Limit order not filled within the allowed timeout window."""


class OrderManager:
    """Manages buy/sell order submission and confirmation for KRX spot trading.

    Args:
        conn: Open SQLite connection (WAL mode recommended).
        kis: Async KISRestClient instance (must be open).
        telegram_bot: Optional bot for trade notifications.
        config: Optional Config snapshot. Loaded via :func:`load_config` if omitted.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        kis: "KISRestClient",
        telegram_bot=None,
        config=None,
    ) -> None:
        self._conn = conn
        self._kis = kis
        self._telegram = telegram_bot
        self._config = config or load_config()

    # ------------------------------------------------------------------
    # Core order operations
    # ------------------------------------------------------------------

    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: int | None = None,
        order_type: str = "limit",
    ) -> dict:
        """Submit an order to KIS and persist to DB.

        Args:
            symbol: KRX ticker, e.g. ``'005930'``.
            side: ``'buy'`` or ``'sell'``.
            quantity: Integer share count.
            price: Limit price (원), rounded to tick. None → market order.
            order_type: ``'limit'`` or ``'market'``. Derived from price if not set.

        Returns:
            Dict with ``order_id``, ``broker_order_id`` (KIS odno), and KIS response.
        """
        if price is not None:
            price = round_to_tick(price)
            order_type = "limit"
        else:
            order_type = "market"

        if side == "buy":
            result = await self._kis.place_buy_order(symbol, quantity, price)
        else:
            result = await self._kis.place_sell_order(symbol, quantity, price)

        broker_order_id = result.get("odno", "")
        krx_orgno = result.get("KRX_FWDG_ORD_ORGNO") or result.get("krx_fwdg_ord_orgno", "")
        internal_id = str(uuid.uuid4())

        self._persist_order(
            internal_id=internal_id,
            broker_order_id=broker_order_id,
            krx_orgno=krx_orgno,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        logger.info(
            "order.created symbol=%s side=%s type=%s qty=%d price=%s broker_id=%s",
            symbol, side, order_type, quantity, price, broker_order_id,
        )
        result["internal_order_id"] = internal_id
        result["broker_order_id"] = broker_order_id
        result["krx_orgno"] = krx_orgno
        return result

    async def cancel_symbol_orders(self, symbol: str) -> None:
        """Cancel all pending orders for *symbol*.

        Errors are logged but never raised — close must proceed regardless.
        """
        try:
            pending = await self._kis.fetch_unfilled_orders()
            for order in pending:
                if order["symbol"] != symbol:
                    continue
                if not order["order_no"] or not order["krx_orgno"]:
                    continue
                remaining = order["qty"] - order["filled_qty"]
                if remaining <= 0:
                    continue
                try:
                    await self._kis.cancel_order(
                        order_no=order["order_no"],
                        krx_orgno=order["krx_orgno"],
                        qty=remaining,
                        ord_dvsn=order.get("ord_dvsn", "00"),
                    )
                    self._conn.execute(
                        "UPDATE orders SET status='canceled', updated_at=? WHERE broker_order_id=?",
                        (datetime.now(timezone.utc).isoformat(), order["order_no"]),
                    )
                    self._conn.commit()
                    logger.info("order.canceled symbol=%s order_no=%s", symbol, order["order_no"])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("cancel failed symbol=%s order_no=%s: %s",
                                   symbol, order["order_no"], exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cancel_symbol_orders failed symbol=%s: %s", symbol, exc)

    async def market_close(self, symbol: str, quantity: int) -> dict:
        """Close a position immediately via a market sell order.

        Cancels any pending limit orders first.
        """
        logger.warning("market_close symbol=%s qty=%d", symbol, quantity)
        await self.cancel_symbol_orders(symbol)
        result = await self.create_order(symbol, "sell", quantity, price=None)
        await asyncio.sleep(MARKET_FILL_WAIT_SEC)
        self._update_order_fill(result["broker_order_id"], None, quantity)
        return result

    # ------------------------------------------------------------------
    # Submit and confirm (async flow)
    # ------------------------------------------------------------------

    async def submit_and_confirm(
        self,
        order: dict,
        timeout_sec: int = 30,
        _retry: bool = False,
    ) -> dict:
        """Submit an order and wait for fill confirmation.

        Flow (market orders):
            1. Submit via :meth:`create_order`.
            2. Wait ``MARKET_FILL_WAIT_SEC`` seconds.
            3. Mark as filled (KRX market orders fill near-instantly).

        Flow (limit orders):
            1. Submit via :meth:`create_order`.
            2. Poll pending order list every ``LIMIT_POLL_INTERVAL_SEC`` s.
            3. Timeout → cancel, then retry once as market.

        Args:
            order: Keys: ``symbol``, ``side``, ``quantity``; optional
                ``price``, ``tp1``, ``tp2``, ``sl``.
            _retry: Internal flag preventing infinite recursion on market retry.

        Raises:
            OrderTimeoutError: Market retry also timed out.
        """
        symbol = order["symbol"]
        side = order["side"]
        quantity = order["quantity"]
        price = order.get("price")

        submitted = await self.create_order(symbol, side, quantity, price)
        broker_id = submitted["broker_order_id"]

        if price is None:
            # Market order — fill assumed after short delay
            await asyncio.sleep(MARKET_FILL_WAIT_SEC)
            fill_price = price
            self._update_order_fill(broker_id, fill_price, quantity)
        else:
            # Limit order — poll until filled or timeout
            fill_price = await self._wait_for_fill(broker_id, timeout_sec)
            if fill_price is None:
                # Timeout: cancel and optionally retry as market
                await self.cancel_symbol_orders(symbol)
                if not _retry:
                    logger.warning(
                        "limit order timeout — retrying as market: %s %s qty=%d",
                        symbol, side, quantity,
                    )
                    market_order = {**order, "price": None}
                    return await self.submit_and_confirm(
                        market_order, timeout_sec=10, _retry=True
                    )
                raise OrderTimeoutError(
                    f"Order not filled within {timeout_sec}s: {broker_id}"
                )

        self._notify(
            f"[FILL] {symbol} {side.upper()} qty={quantity}"
            + (f" @ {fill_price}" if fill_price else " (market)")
        )
        submitted["confirmed"] = True
        return submitted

    # ------------------------------------------------------------------
    # Slippage validation
    # ------------------------------------------------------------------

    @staticmethod
    def check_slippage(
        expected_price: float,
        actual_fill_price: float,
        side: str,
    ) -> bool:
        """Return True if fill is within MAX_SLIPPAGE_PCT (0.3 %), else False."""
        if side == "buy":
            return actual_fill_price <= expected_price * (1.0 + MAX_SLIPPAGE_PCT)
        return actual_fill_price >= expected_price * (1.0 - MAX_SLIPPAGE_PCT)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_order(
        self,
        internal_id: str,
        broker_order_id: str,
        krx_orgno: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        price: int | None,
    ) -> None:
        trading_mode = self._config.trading_mode.value
        self._conn.execute(
            """
            INSERT OR IGNORE INTO orders (
                order_id, broker_order_id, symbol, side, position_side,
                order_type, price, quantity, status, trading_mode
            ) VALUES (?, ?, ?, ?, 'both', ?, ?, ?, 'open', ?)
            """,
            (
                internal_id,
                broker_order_id,
                symbol,
                side,
                order_type,
                str(price) if price is not None else None,
                str(quantity),
                trading_mode,
            ),
        )
        self._conn.commit()

    def _update_order_fill(
        self,
        broker_order_id: str,
        fill_price: float | None,
        fill_qty: int,
    ) -> None:
        self._conn.execute(
            """
            UPDATE orders
            SET filled_qty=?, avg_fill_price=?, status='filled', updated_at=?
            WHERE broker_order_id=?
            """,
            (
                str(fill_qty),
                str(fill_price) if fill_price is not None else None,
                datetime.now(timezone.utc).isoformat(),
                broker_order_id,
            ),
        )
        self._conn.commit()

    async def _wait_for_fill(
        self,
        broker_order_id: str,
        timeout_sec: int,
    ) -> float | None:
        """Poll KIS pending orders until *broker_order_id* disappears (= filled).

        Returns the approximate fill price if filled, None if timed out.
        """
        elapsed = 0
        while elapsed < timeout_sec:
            await asyncio.sleep(LIMIT_POLL_INTERVAL_SEC)
            elapsed += LIMIT_POLL_INTERVAL_SEC
            try:
                pending = await self._kis.fetch_unfilled_orders()
                pending_ids = {o["order_no"] for o in pending}
                if broker_order_id not in pending_ids:
                    # No longer in pending list → filled (or cancelled externally)
                    return None  # fill price unknown; caller uses None = market-approx
            except Exception as exc:  # noqa: BLE001
                logger.warning("fill poll failed: %s", exc)
        return None  # timeout

    def _notify(self, message: str) -> None:
        if self._telegram is None:
            return
        try:
            self._telegram.send_alert(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram notify failed: %s", exc)
