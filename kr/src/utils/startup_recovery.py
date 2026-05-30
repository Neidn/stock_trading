"""Pod startup recovery — reconciles DB state with KIS API ground truth.

Must be the first thing called in each pod's entry point, before any
trading logic or WebSocket connections are opened.

Usage::

    from src.utils.startup_recovery import StartupRecovery

    recovery = StartupRecovery(conn, kis_client)
    await recovery.run()
    # normal logic starts here
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from src.db.models import log_system_event, upsert_position
from src.monitoring.logger import get_logger

logger = get_logger("startup_recovery")

# In-process balance cache — written by _sync_balance, read by risk/execution
_balance_cache: dict[str, str] = {}


def get_cached_balance() -> dict[str, str]:
    """Return the most recently synced balance snapshot."""
    return dict(_balance_cache)


class StartupRecovery:
    """Reconcile local DB with live KIS API on pod startup.

    Args:
        conn: Open SQLite connection (already initialised via init_db).
        rest_client: Open KISRestClient instance.
        telegram: Optional callable ``async (msg: str) -> None``.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        rest_client: Any,
        telegram: Any | None = None,
    ) -> None:
        self._conn = conn
        self._rest = rest_client
        self._telegram = telegram

        self._report: dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "positions": {},
            "orders": {},
            "balance": {},
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Execute the full recovery sequence."""
        logger.info("=== StartupRecovery BEGIN ===")
        start = time.monotonic()

        await self._recover_open_positions()
        await self._recover_pending_orders()
        await self._sync_balance()
        self._log_recovery_event(elapsed=time.monotonic() - start)

        logger.info("=== StartupRecovery DONE in %.1fs ===", time.monotonic() - start)

    # ------------------------------------------------------------------
    # Step 1 — positions
    # ------------------------------------------------------------------

    async def _recover_open_positions(self) -> None:
        logger.info("Recovering open positions …")
        report = self._report["positions"]

        db_rows = self._conn.execute(
            "SELECT * FROM positions WHERE status = 'open'"
        ).fetchall()
        db_positions: dict[str, Any] = {row["symbol"]: dict(row) for row in db_rows}

        try:
            api_list = await self._rest.fetch_positions()
        except Exception as exc:
            logger.error("fetch_positions failed: %s — skipping reconcile", exc)
            report["error"] = str(exc)
            return

        api_positions: dict[str, Any] = {p["symbol"]: p for p in api_list}

        added, removed, updated = [], [], []

        for sym, api_pos in api_positions.items():
            if sym not in db_positions:
                logger.warning("UNKNOWN position on KIS: %s — adding to DB", sym)
                self._upsert_from_api(sym, api_pos)
                added.append(sym)
                await self._alert(
                    f"⚠️ StartupRecovery: 미추적 포지션 {sym} "
                    f"(long qty={api_pos['positionAmt']}) — DB에 추가됨"
                )
            else:
                self._upsert_from_api(sym, api_pos, existing=db_positions[sym])
                updated.append(sym)

        for sym, db_pos in db_positions.items():
            if sym not in api_positions:
                logger.warning("Position %s in DB but not on KIS — marking force_closed", sym)
                self._conn.execute(
                    """UPDATE positions
                       SET status = 'force_closed',
                           close_reason = 'external_close_on_restart',
                           closed_at = ?
                       WHERE position_id = ? AND status = 'open'""",
                    (datetime.now(timezone.utc).isoformat(), db_pos["position_id"]),
                )
                removed.append(sym)
                await self._alert(f"⚠️ StartupRecovery: {sym} DB에 있었으나 KIS에 없음 → force_closed")

        self._conn.commit()
        report.update({"added": added, "removed": removed, "updated": updated})
        logger.info("Positions: %d added, %d removed, %d updated", len(added), len(removed), len(updated))

    def _upsert_from_api(self, symbol: str, api: dict, existing: dict | None = None) -> None:
        entry = float(api.get("entryPrice") or 0)
        qty   = int(float(api.get("positionAmt") or 0))
        # KRX long-only: fallback SL 3% below entry when no existing SL
        fallback_sl = str(round(entry * 0.97)) if entry > 0 else "0"
        position = {
            "position_id":        existing["position_id"] if existing else str(uuid.uuid4()),
            "symbol":             symbol,
            "side":               "long",
            "leverage":           1,
            "entry_price":        str(entry),
            "exit_price":         existing.get("exit_price") if existing else None,
            "quantity":           str(qty),
            "liquidation_price":  "0",
            "stop_loss":          existing.get("stop_loss", fallback_sl) if existing else fallback_sl,
            "take_profit_1":      existing.get("take_profit_1") if existing else None,
            "take_profit_2":      existing.get("take_profit_2") if existing else None,
            "initial_stop_loss":  existing.get("initial_stop_loss", fallback_sl) if existing else fallback_sl,
            "trailing_activated": existing.get("trailing_activated", 0) if existing else 0,
            "realized_pnl":       "0",
            "unrealized_pnl":     api.get("unrealizedProfit", "0"),
            "status":             "open",
            "close_reason":       None,
            "trading_mode":       "paper",
            "opened_at":          existing.get("opened_at") if existing else datetime.now(timezone.utc).isoformat(),
            "closed_at":          None,
        }
        upsert_position(self._conn, position)

    # ------------------------------------------------------------------
    # Step 2 — pending orders
    # ------------------------------------------------------------------

    async def _recover_pending_orders(self) -> None:
        logger.info("Recovering pending orders …")
        report = self._report["orders"]

        pending_rows = self._conn.execute(
            "SELECT * FROM orders WHERE status IN ('new', 'pending', 'partially_filled')"
        ).fetchall()

        if not pending_rows:
            logger.info("No pending orders to recover.")
            report["synced"] = 0
            return

        try:
            live_orders = await self._rest.fetch_unfilled_orders()
        except Exception as exc:
            logger.error("fetch_unfilled_orders failed: %s — skipping order recovery", exc)
            report["error"] = str(exc)
            return

        live_order_nos: set[str] = {o["order_no"] for o in live_orders if o.get("order_no")}

        canceled = 0
        now = datetime.now(timezone.utc).isoformat()
        for row in pending_rows:
            broker_oid = row["broker_order_id"] if hasattr(row, "keys") else None
            oid        = row["order_id"]         if hasattr(row, "keys") else row[0]

            if not broker_oid or str(broker_oid) not in live_order_nos:
                self._conn.execute(
                    "UPDATE orders SET status='canceled', updated_at=? WHERE order_id=?",
                    (now, oid),
                )
                canceled += 1
                logger.warning("Order %s has no live KIS match — marked canceled", oid)

        self._conn.commit()
        report.update({"canceled": canceled})
        logger.info("Orders: %d marked canceled", canceled)

    # ------------------------------------------------------------------
    # Step 3 — balance
    # ------------------------------------------------------------------

    async def _sync_balance(self) -> None:
        logger.info("Syncing account balance …")
        try:
            balance = await self._rest.fetch_account_balance()
            _balance_cache.clear()
            _balance_cache.update(balance)
            self._report["balance"] = balance
            logger.info(
                "Balance synced — total: %s KRW, available: %s KRW",
                balance.get("totalWalletBalance", "?"),
                balance.get("availableBalance", "?"),
            )
        except Exception as exc:
            logger.error("Balance sync failed: %s", exc)
            self._report["balance"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # Step 4 — audit log
    # ------------------------------------------------------------------

    def _log_recovery_event(self, elapsed: float) -> None:
        self._report["elapsed_sec"] = round(elapsed, 2)
        self._report["completed_at"] = datetime.now(timezone.utc).isoformat()
        try:
            log_system_event(
                self._conn,
                module="startup_recovery",
                severity="info",
                message="Startup recovery completed",
                metadata=self._report,
                event_type="recovery",
            )
            self._conn.commit()
        except Exception as exc:
            logger.error("Failed to write recovery event: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _alert(self, message: str) -> None:
        if self._telegram is None:
            logger.warning("ALERT (no telegram): %s", message)
            return
        try:
            await self._telegram(message)
        except Exception as exc:
            logger.error("Telegram alert failed: %s", exc)
