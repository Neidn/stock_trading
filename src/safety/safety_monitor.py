"""KRX Safety Monitor — independent price-watching watchdog.

Runs every CHECK_INTERVAL_SEC seconds (default 2s) for KRX scalping speed.

Per open position:
  1. Get current price from KIS WS (real-time) or REST fallback.
  2. SL breach (price ≤ stop_loss) → immediate market sell (full qty).
  3. TP1 hit (price ≥ take_profit_1, not yet done) → limit sell 50%, move SL to breakeven.
  4. TP2 hit (price ≥ take_profit_2, not yet done) → limit sell remaining qty.
  5. 15:20 KST (FORCE_CLOSE_TIME) → market sell all open positions.

SL and TP levels are set at order creation time and stored in the positions table.
KIS has no native conditional orders — this monitor is the only SL/TP enforcement.

Design constraints:
  - run_forever() NEVER exits on exception; logs + Telegram only.
  - Each check method is independent; one failure must not block others.
  - TP state (tp1_done, tp2_done) is in-memory; survives as long as process lives.
    On restart, re-check is safe: we just try to re-sell an already-gone qty.
"""

from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.risk.market_hours import is_closing_soon

if TYPE_CHECKING:
    import sqlite3
    from src.execution.order_manager import OrderManager
    from src.ingest.kis_rest import KISRestClient
    from src.ingest.kis_ws import KISWSManager

logger = logging.getLogger(__name__)

_DEFAULT_POD_URLS: dict[str, str] = {
    "data-ingest":   "http://data-ingest-svc:8080/health",
    "signal-engine": "http://signal-engine-svc:8080/health",
}

FORCE_CLOSE_BUFFER_MIN: int = 10   # force-close N minutes before market end (15:20 KST)


class SafetyMonitor:
    """KRX spot position watchdog — manual SL/TP enforcement via price monitoring.

    Args:
        conn: SQLite connection (read positions, update on close).
        order_manager: :class:`~src.execution.order_manager.OrderManager` instance.
        ws_manager: Optional KIS WS manager for real-time prices (fastest path).
        kis: Optional KIS REST client for price fallback.
        telegram_bot: Optional; must expose ``send_critical`` / ``send_warning``.
        pod_urls: Override default K8s health URLs.
        check_interval: Poll interval in seconds (default 2 for scalping).
    """

    CHECK_INTERVAL_SEC: int = 2

    def __init__(
        self,
        conn: sqlite3.Connection,
        order_manager: "OrderManager",
        ws_manager: "KISWSManager | None" = None,
        kis: "KISRestClient | None" = None,
        telegram_bot=None,
        pod_urls: dict[str, str] | None = None,
        check_interval: int | None = None,
    ) -> None:
        self._conn = conn
        self._om = order_manager
        self._ws = ws_manager
        self._kis = kis
        self._telegram = telegram_bot
        self._pod_urls = pod_urls if pod_urls is not None else _DEFAULT_POD_URLS
        self._interval = check_interval if check_interval is not None else self.CHECK_INTERVAL_SEC

        # In-memory TP state — survives as long as process runs
        self._tp1_done: set[str] = set()   # position_ids that already triggered TP1 sell
        self._tp2_done: set[str] = set()
        self._force_closed: bool = False   # set True after EOD close sweep

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """Main watchdog loop — never exits except on CancelledError."""
        logger.info("safety_monitor.start interval=%ds", self._interval)
        while True:
            try:
                if is_closing_soon(buffer_min=FORCE_CLOSE_BUFFER_MIN):
                    await self._force_close_all()
                else:
                    self._force_closed = False  # reset for next trading day
                    await self._check_all_positions()

                await self._check_pod_health()
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                logger.info("safety_monitor.cancelled")
                raise
            except Exception as exc:  # noqa: BLE001
                msg = f"Safety Monitor 에러: {html.escape(str(exc))}"
                logger.error(msg, exc_info=True)
                self._notify_critical(msg)
                await asyncio.sleep(5.0)

    # ------------------------------------------------------------------
    # Position checks
    # ------------------------------------------------------------------

    async def _check_all_positions(self) -> None:
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE status='open'"
        ).fetchall()

        for row in rows:
            pos = dict(row) if hasattr(row, "keys") else {
                k: row[i] for i, k in enumerate(row.description)
            } if hasattr(row, "description") else _row_to_dict(row)

            symbol = pos.get("symbol", "")
            if not symbol:
                continue

            price = await self._get_price(symbol)
            if price is None or price <= 0:
                logger.debug("safety.no_price symbol=%s", symbol)
                continue

            # SL check first — highest priority
            sl_closed = await self._check_sl(pos, price)
            if sl_closed:
                continue

            # TP checks
            await self._check_tp(pos, price)

    async def _check_sl(self, pos: dict, price: int) -> bool:
        """Close position immediately if price ≤ stop_loss.

        Returns True if SL was triggered.
        """
        try:
            sl = float(pos.get("stop_loss") or 0)
        except (TypeError, ValueError):
            return False
        if sl <= 0:
            return False

        if price <= sl:
            symbol = pos["symbol"]
            qty = self._current_qty(pos)
            msg = f"🛑 SL 돌파: {symbol} 현재가={price:,} ≤ SL={sl:,.0f} → 시장가 청산"
            logger.critical("sl.breach symbol=%s price=%d sl=%.0f qty=%d", symbol, price, sl, qty)
            self._notify_critical(msg)

            try:
                await self._om.market_close(symbol, qty)
                self._mark_position_closed(pos, price, "sl_hit")
            except Exception as exc:  # noqa: BLE001
                logger.error("sl.close_failed symbol=%s: %s", symbol, exc)
                self._notify_critical(f"❗ SL 청산 실패 {symbol}: {html.escape(str(exc))}")
            return True
        return False

    async def _check_tp(self, pos: dict, price: int) -> None:
        """Execute TP1/TP2 limit sells when price reaches targets."""
        position_id = pos.get("position_id", "")
        symbol = pos.get("symbol", "")

        try:
            tp1 = float(pos.get("take_profit_1") or 0)
            tp2 = float(pos.get("take_profit_2") or 0)
        except (TypeError, ValueError):
            return

        total_qty = self._current_qty(pos)

        # TP1 — sell 50%
        if tp1 > 0 and price >= tp1 and position_id not in self._tp1_done:
            half_qty = total_qty // 2
            if half_qty > 0:
                msg = f"🎯 TP1 도달: {symbol} 현재가={price:,} ≥ TP1={tp1:,.0f} → {half_qty}주 지정가 매도"
                logger.info("tp1.hit symbol=%s price=%d tp1=%.0f qty=%d", symbol, price, tp1, half_qty)
                self._notify_info(msg)
                try:
                    await self._om.create_order(symbol, "sell", half_qty, int(tp1))
                    self._tp1_done.add(position_id)
                    # Move SL to breakeven only when entry_price is known
                    entry_price_str = pos.get("entry_price") or "0"
                    if float(entry_price_str) > 0:
                        self._move_sl_to_breakeven(position_id, entry_price_str)
                    else:
                        logger.warning(
                            "tp1.breakeven_skip symbol=%s: entry_price=0, keeping initial SL",
                            symbol,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.error("tp1.sell_failed symbol=%s: %s", symbol, exc)

        # TP2 — sell remaining qty
        if tp2 > 0 and price >= tp2 and position_id not in self._tp2_done:
            remaining = (total_qty - total_qty // 2) if position_id in self._tp1_done else total_qty
            if remaining > 0:
                msg = f"🎯 TP2 도달: {symbol} 현재가={price:,} ≥ TP2={tp2:,.0f} → {remaining}주 지정가 매도"
                logger.info("tp2.hit symbol=%s price=%d tp2=%.0f qty=%d", symbol, price, tp2, remaining)
                self._notify_info(msg)
                try:
                    await self._om.create_order(symbol, "sell", remaining, int(tp2))
                    self._tp2_done.add(position_id)
                    self._mark_position_closed(pos, price, "tp2_hit")
                except Exception as exc:  # noqa: BLE001
                    logger.error("tp2.sell_failed symbol=%s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Force close at market end
    # ------------------------------------------------------------------

    async def _force_close_all(self) -> None:
        """Market-sell all open positions at FORCE_CLOSE_TIME."""
        if self._force_closed:
            return

        rows = self._conn.execute(
            "SELECT * FROM positions WHERE status='open'"
        ).fetchall()

        if not rows:
            self._force_closed = True
            return

        logger.warning("safety.force_close_eod count=%d", len(rows))
        self._notify_critical(f"⏰ 장마감 강제 청산: {len(rows)}개 포지션")

        for row in rows:
            pos = _row_to_dict(row)
            symbol = pos.get("symbol", "")
            qty = self._current_qty(pos)
            if not symbol or qty <= 0:
                continue
            try:
                await self._om.market_close(symbol, qty)
                self._mark_position_closed(pos, None, "force_close_eod")
            except Exception as exc:  # noqa: BLE001
                logger.error("force_close.failed symbol=%s: %s", symbol, exc)
                self._notify_critical(f"❗ 강제 청산 실패 {symbol}: {html.escape(str(exc))}")

        self._force_closed = True

    # ------------------------------------------------------------------
    # Pod health checks
    # ------------------------------------------------------------------

    async def _check_pod_health(self) -> None:
        import urllib.error
        import urllib.request

        for pod_name, url in self._pod_urls.items():
            try:
                def _get(u=url):
                    with urllib.request.urlopen(u, timeout=5) as r:
                        return r.status
                status = await asyncio.to_thread(_get)
                if status != 200:
                    self._notify_warning(f"{pod_name} 헬스체크 실패 (HTTP {status})")
            except Exception as exc:  # noqa: BLE001
                self._notify_warning(f"{pod_name} 응답 없음: {html.escape(str(exc))}")

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    async def _get_price(self, symbol: str) -> int | None:
        # 1. Real-time WS price (fastest, ~1s lag)
        if self._ws is not None:
            price = self._ws.get_last_price(symbol)
            if price:
                return price

        # 2. REST fallback (adds 50–200ms latency)
        if self._kis is not None:
            try:
                detail = await self._kis.fetch_current_price(symbol)
                raw = detail.get("price", "0")
                return int(float(raw)) if raw else None
            except Exception as exc:  # noqa: BLE001
                logger.debug("safety.price_rest_failed symbol=%s: %s", symbol, exc)
        return None

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _current_qty(self, pos: dict) -> int:
        try:
            return int(float(pos.get("quantity") or 0))
        except (TypeError, ValueError):
            return 0

    def _mark_position_closed(self, pos: dict, price: int | None, reason: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """UPDATE positions
               SET status='closed', close_reason=?, closed_at=?, exit_price=?
               WHERE position_id=? AND status='open'""",
            (reason, now, str(price) if price else None, pos.get("position_id")),
        )
        self._conn.commit()

    def _move_sl_to_breakeven(self, position_id: str, entry_price: str) -> None:
        """Raise stop_loss to entry_price after TP1 is hit (breakeven stop)."""
        self._conn.execute(
            "UPDATE positions SET stop_loss=? WHERE position_id=? AND status='open'",
            (entry_price, position_id),
        )
        self._conn.commit()
        logger.info("sl.moved_to_breakeven position_id=%s entry=%s", position_id, entry_price)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _notify_critical(self, message: str) -> None:
        if self._telegram is None:
            return
        try:
            self._telegram.send_critical(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram.critical_failed: %s", exc)

    def _notify_warning(self, message: str) -> None:
        if self._telegram is None:
            return
        try:
            self._telegram.send_warning(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram.warning_failed: %s", exc)

    def _notify_info(self, message: str) -> None:
        if self._telegram is None:
            return
        try:
            self._telegram.send_info(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram.info_failed: %s", exc)


# ---------------------------------------------------------------------------
# Utility: sqlite3.Row → dict without row.description dependency
# ---------------------------------------------------------------------------

def _row_to_dict(row) -> dict:
    if hasattr(row, "keys"):
        return dict(row)
    # Plain tuple fallback (shouldn't happen with conn.row_factory = sqlite3.Row)
    return {}
