"""US stock OrderManager — wraps KISRestClient overseas API.

Differences from KR OrderManager:
  - USD float prices (no round_to_tick; US has penny increments)
  - place_buy_order_us / place_sell_order_us with excd per symbol
  - fetch_unfilled_orders_us for fill polling
  - cancel_order_us for cancellation

excd (exchange code) is resolved per-symbol from the symbols table:
  NAS → NASDAQ, NYS → NYSE, AMS → AMEX
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.db.models import upsert_position
from src.execution.order_manager import OrderManager, MARKET_FILL_WAIT_SEC, LIMIT_POLL_INTERVAL_SEC, OrderTimeoutError
from src.utils.config import load_config

if TYPE_CHECKING:
    import sqlite3
    from src.ingest.kis_rest import KISRestClient

logger = logging.getLogger(__name__)

_DEFAULT_EXCD = "NAS"


class OrderManagerUS(OrderManager):
    """KIS overseas stock OrderManager.

    Inherits the same interface as OrderManager but routes all orders through
    KIS overseas-stock endpoints (TTTT/VTTT TR IDs) with float USD prices.
    """

    def _get_excd(self, symbol: str) -> str:
        """Look up the exchange code for *symbol* from DB. Defaults to NAS."""
        try:
            row = self._conn.execute(
                "SELECT excd FROM symbols WHERE symbol=? LIMIT 1", (symbol,)
            ).fetchone()
            excd = (row["excd"] if hasattr(row, "keys") else row[0]) if row else None
            return excd.strip().upper() if excd else _DEFAULT_EXCD
        except Exception:  # noqa: BLE001
            return _DEFAULT_EXCD

    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float | None = None,
        order_type: str = "limit",
    ) -> dict:
        """Submit a US stock order via KIS overseas API.

        Args:
            symbol: US ticker, e.g. ``'AAPL'``.
            side: ``'buy'`` or ``'sell'``.
            quantity: Integer share count.
            price: Float USD limit price. None → market order (price=0).
            order_type: Informational; derived from price.

        Returns:
            Dict with ``order_id``, ``broker_order_id``, and KIS response.
        """
        order_type = "limit" if price is not None else "market"
        excd = self._get_excd(symbol)

        if side == "buy":
            result = await self._kis.place_buy_order_us(symbol, excd, quantity, price)
        else:
            result = await self._kis.place_sell_order_us(symbol, excd, quantity, price)

        broker_order_id = result.get("odno", "")
        internal_id = str(uuid.uuid4())

        self._persist_order(
            internal_id=internal_id,
            broker_order_id=broker_order_id,
            krx_orgno="",        # no KRX_FWDG for US
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=int(price * 100) if price is not None else None,  # store as cents for compat
        )
        logger.info(
            "order.us.created symbol=%s excd=%s side=%s type=%s qty=%d price=%s broker_id=%s",
            symbol, excd, side, order_type, quantity, price, broker_order_id,
        )
        result["internal_order_id"] = internal_id
        result["broker_order_id"] = broker_order_id
        result["krx_orgno"] = ""
        return result

    async def cancel_symbol_orders(self, symbol: str) -> None:
        """Cancel all pending US orders for *symbol*."""
        excd = self._get_excd(symbol)
        ovrs_excd = {"NAS": "NASD", "NYS": "NYSE", "AMS": "AMEX"}.get(excd, excd)
        try:
            pending = await self._kis.fetch_unfilled_orders_us(excd=ovrs_excd)
            for order in pending:
                if order["symbol"] != symbol:
                    continue
                remaining = order["qty"] - order["filled_qty"]
                if remaining <= 0:
                    continue
                try:
                    await self._kis.cancel_order_us(
                        order_no=order["order_no"],
                        excd=excd,
                        qty=remaining,
                        ord_dvsn=order.get("ord_dvsn", "00"),
                    )
                    self._conn.execute(
                        "UPDATE orders SET status='canceled', updated_at=? WHERE broker_order_id=?",
                        (datetime.now(timezone.utc).isoformat(), order["order_no"]),
                    )
                    self._conn.commit()
                    logger.info("order.us.canceled symbol=%s order_no=%s", symbol, order["order_no"])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("us.cancel failed symbol=%s: %s", symbol, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cancel_symbol_orders_us failed symbol=%s: %s", symbol, exc)

    async def market_close(self, symbol: str, quantity: int) -> dict:
        """Close a US position via market sell."""
        logger.warning("market_close.us symbol=%s qty=%d", symbol, quantity)
        await self.cancel_symbol_orders(symbol)
        result = await self.create_order(symbol, "sell", quantity, price=None)
        await asyncio.sleep(MARKET_FILL_WAIT_SEC)
        self._update_order_fill(result["broker_order_id"], None, quantity)
        return result

    async def _wait_for_fill(
        self,
        broker_order_id: str,
        timeout_sec: int,
    ) -> float | None:
        """Poll KIS US pending orders until broker_order_id disappears (filled)."""
        elapsed = 0
        while elapsed < timeout_sec:
            await asyncio.sleep(LIMIT_POLL_INTERVAL_SEC)
            elapsed += LIMIT_POLL_INTERVAL_SEC
            try:
                pending = await self._kis.fetch_unfilled_orders_us()
                pending_ids = {o["order_no"] for o in pending}
                if broker_order_id not in pending_ids:
                    return None  # filled (price unknown; caller uses approximation)
            except Exception as exc:  # noqa: BLE001
                logger.warning("us.fill poll failed: %s", exc)
        return None  # timeout

    def register_position(
        self,
        symbol: str,
        quantity: int,
        fill_price: float,
        sl: float,
        tp1: float | None = None,
        tp2: float | None = None,
        signal_id: str | None = None,
        strategy_name: str | None = None,
    ) -> str:
        """Create a US position record after a buy fill (USD float prices)."""
        position_id = str(uuid.uuid4())
        upsert_position(self._conn, {
            "position_id":    position_id,
            "symbol":         symbol,
            "side":           "long",
            "entry_price":    str(fill_price),
            "quantity":       str(quantity),
            "stop_loss":      str(sl),
            "initial_stop_loss": str(sl),
            "take_profit_1":  str(tp1) if tp1 else None,
            "take_profit_2":  str(tp2) if tp2 else None,
            "fill_price":     str(fill_price),
            "trading_mode":   self._config.trading_mode.value,
            "strategy_name":  strategy_name,
            "currency":       "USD",
            "market":         "US",
        })
        self._conn.commit()
        logger.info(
            "position.us.registered id=%s symbol=%s entry=%.4f sl=%.4f",
            position_id, symbol, fill_price, sl,
        )
        return position_id
