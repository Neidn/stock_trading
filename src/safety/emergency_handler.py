"""Emergency position closure handler.

Triggered by:
  - Telegram /emergency_close command
  - LiquidationGuard CRITICAL level (automatic)
  - DrawdownGuard daily/weekly limit breach

Design constraints:
  - close_all_positions NEVER stops mid-loop on individual failure
  - Every close attempt is logged; failures alert via Telegram
  - SafeMode is always activated after close_all_positions completes
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


class EmergencyHandler:
    """Handles emergency position closures.

    Args:
        conn: Open SQLite connection.
        order_manager: :class:`~src.execution.order_manager.OrderManager` instance.
        position_tracker: :class:`~src.execution.position_tracker.PositionTracker` class or
            instance — only static methods are used.
        safe_mode: :class:`~src.safety.safe_mode.SafeMode` instance.
        telegram_bot: Optional bot; must expose ``send_alert(message: str)``.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        order_manager,
        position_tracker,
        safe_mode,
        telegram_bot=None,
    ) -> None:
        self._conn = conn
        self._om = order_manager
        self._pt = position_tracker
        self._safe_mode = safe_mode
        self._telegram = telegram_bot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def close_all_positions(
        self,
        reason: str,
        exchange_positions: list[dict] | None = None,
    ) -> dict:
        """Close every open position immediately via market orders.

        Individual failures do not abort the loop — all positions are
        attempted regardless.  SafeMode is activated unconditionally after
        the loop finishes.

        Args:
            reason: Human-readable trigger description.
            exchange_positions: Optional list of live position dicts fetched by the
                caller (SafetyMonitor).  Used as fallback when the local DB has no
                open rows — handles the case where a position exists on the exchange
                but was never persisted (or already removed from) the DB.

        Returns:
            Summary dict::

                {
                    "total": int,
                    "closed": int,
                    "failed": int,
                    "results": [{"symbol": str, "status": "closed"|"failed",
                                 "error": str | None}, ...]
                }
        """
        self._notify(f"[EMERGENCY] 긴급 전량 청산 시작 — 이유: {reason}")
        logger.critical("EmergencyHandler.close_all_positions: %s", reason)

        rows = self._conn.execute(
            "SELECT * FROM positions WHERE status='open'"
        ).fetchall()

        # Build work list from DB rows
        work: list[dict] = [
            {
                "symbol": row["symbol"],
                "side": row["side"],
                "qty": float(row["quantity"]),
                "pid": row["position_id"],
                "source": "db",
            }
            for row in rows
        ]

        # Supplement with live exchange positions not already covered by DB
        if exchange_positions:
            db_symbols = {w["symbol"] for w in work}
            for pos in exchange_positions:
                sym = pos.get("symbol", "")
                contracts = float(pos.get("contracts") or pos.get("contractSize") or 0)
                side = (pos.get("side") or "").lower()
                if sym and sym not in db_symbols and contracts > 0 and side in ("long", "short"):
                    work.append({
                        "symbol": sym,
                        "side": side,
                        "qty": contracts,
                        "pid": None,
                        "source": "exchange",
                    })
                    logger.warning(
                        "Emergency close: %s found on exchange but not in DB — closing directly",
                        sym,
                    )

        results = []
        for item in work:
            symbol = item["symbol"]
            side   = item["side"]
            qty    = item["qty"]
            pid    = item["pid"]

            close_side = "sell" if side == "long" else "buy"

            try:
                self._om.market_close(symbol, close_side, qty, position_side=side)
                if pid:
                    self._pt.close_position(self._conn, pid, self._last_fill_price(symbol), reason)
                results.append({"symbol": symbol, "status": "closed", "error": None})
                logger.info("Emergency closed: %s (source=%s)", symbol, item["source"])
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                # -2022: position already liquidated on exchange — sync DB and treat as closed
                if "-2022" in error_msg:
                    logger.warning(
                        "Emergency close %s: already liquidated (-2022), syncing DB", symbol,
                    )
                    if pid:
                        self._pt.close_position(self._conn, pid,
                                                self._last_fill_price(symbol), "already_liquidated")
                    results.append({"symbol": symbol, "status": "closed", "error": None})
                else:
                    results.append({"symbol": symbol, "status": "failed", "error": error_msg})
                    logger.error("Emergency close FAILED for %s: %s", symbol, error_msg)
                    self._notify(f"[EMERGENCY] 청산 실패: {symbol} — {error_msg}")

        closed = sum(1 for r in results if r["status"] == "closed")
        failed = sum(1 for r in results if r["status"] == "failed")

        summary_lines = [
            f"[EMERGENCY] 긴급 청산 완료",
            f"성공: {closed}건 / 실패: {failed}건",
        ]
        if failed:
            failed_symbols = [r["symbol"] for r in results if r["status"] == "failed"]
            summary_lines.append(f"실패 심볼: {failed_symbols}")
            summary_lines.append("⚠️ 실패한 포지션은 수동 확인 필요")
        self._notify("\n".join(summary_lines))

        # Always activate SafeMode after emergency close regardless of failures
        if not self._safe_mode.is_active():
            self._safe_mode.activate(reason=f"긴급 청산 후: {reason}")

        return {
            "total": len(results),
            "closed": closed,
            "failed": failed,
            "results": results,
        }

    def close_position(self, symbol: str, reason: str) -> dict:
        """Close a single open position for *symbol* immediately.

        Args:
            symbol: Trading pair to close, e.g. ``'BTCUSDT'``.
            reason: Human-readable trigger description.

        Returns:
            Result dict::

                {"symbol": str, "status": "closed"|"failed"|"not_found",
                 "error": str | None}
        """
        row = self._conn.execute(
            "SELECT * FROM positions WHERE symbol=? AND status='open'",
            (symbol,),
        ).fetchone()

        if row is None:
            logger.warning("close_position: no open position for %s", symbol)
            return {"symbol": symbol, "status": "not_found", "error": None}

        close_side = "sell" if row["side"] == "long" else "buy"
        qty = float(row["quantity"])

        try:
            self._om.market_close(symbol, close_side, qty, position_side=row["side"])
            self._pt.close_position(
                self._conn, row["position_id"], self._last_fill_price(symbol), reason
            )
            self._notify(f"[CLOSE] {symbol} 청산 완료 — {reason}")
            logger.info("Position closed: %s reason=%s", symbol, reason)
            return {"symbol": symbol, "status": "closed", "error": None}
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            if "-2022" in error_msg:
                logger.warning("close_position %s: already liquidated (-2022), syncing DB", symbol)
                self._pt.close_position(
                    self._conn, row["position_id"], self._last_fill_price(symbol),
                    "already_liquidated"
                )
                self._notify(f"[CLOSE] {symbol} 이미 청산됨 (DB 동기화 완료)")
                return {"symbol": symbol, "status": "closed", "error": None}
            self._notify(f"[CLOSE FAILED] {symbol} 청산 실패 — {error_msg}")
            logger.error("Position close FAILED: %s — %s", symbol, error_msg)
            return {"symbol": symbol, "status": "failed", "error": error_msg}

    def partial_close(self, position_id: str, pct: float, reason: str) -> dict:
        """Partially close a position by *pct* percent of its quantity.

        Used by LiquidationGuard WARNING level for preemptive 50 % reduction.

        Args:
            position_id: Primary key of the position to partially close.
            pct: Fraction to close, 0 < pct <= 1.0  (e.g. ``0.5`` = 50 %).
            reason: Human-readable trigger description.

        Returns:
            Result dict::

                {"position_id": str, "symbol": str, "status": "closed"|"failed",
                 "closed_qty": float, "error": str | None}
        """
        if not (0 < pct <= 1.0):
            raise ValueError(f"pct must be in (0, 1]: got {pct}")

        row = self._conn.execute(
            "SELECT * FROM positions WHERE position_id=? AND status='open'",
            (position_id,),
        ).fetchone()

        if row is None:
            logger.warning("partial_close: position not found or not open: %s", position_id)
            return {
                "position_id": position_id,
                "symbol": "unknown",
                "status": "not_found",
                "closed_qty": 0.0,
                "error": None,
            }

        symbol = row["symbol"]
        side = row["side"]
        full_qty = float(row["quantity"])
        close_qty = round(full_qty * pct, 8)
        close_side = "sell" if side == "long" else "buy"

        try:
            self._om.market_close(symbol, close_side, close_qty, position_side=side)
            remaining_qty = full_qty - close_qty

            if remaining_qty <= 0:
                # Full close via partial (pct=1.0)
                self._pt.close_position(
                    self._conn, position_id, self._last_fill_price(symbol), reason
                )
            else:
                # Reduce quantity in DB — position stays open with reduced size
                self._conn.execute(
                    "UPDATE positions SET quantity=? WHERE position_id=?",
                    (str(remaining_qty), position_id),
                )
                self._conn.commit()

            self._notify(
                f"[PARTIAL CLOSE] {symbol} {pct*100:.0f}% 부분 청산 — {reason} "
                f"(qty={close_qty:.6f})"
            )
            logger.info(
                "Partial close: %s pct=%.0f%% qty=%.6f reason=%s",
                symbol, pct * 100, close_qty, reason,
            )
            return {
                "position_id": position_id,
                "symbol": symbol,
                "status": "closed",
                "closed_qty": close_qty,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            if "-2022" in error_msg:
                logger.warning("partial_close %s: already liquidated (-2022), syncing DB", symbol)
                self._pt.close_position(
                    self._conn, position_id, self._last_fill_price(symbol), "already_liquidated"
                )
                self._notify(f"[PARTIAL CLOSE] {symbol} 이미 청산됨 (DB 동기화 완료)")
                return {
                    "position_id": position_id,
                    "symbol": symbol,
                    "status": "closed",
                    "closed_qty": close_qty,
                    "error": None,
                }
            self._notify(f"[PARTIAL CLOSE FAILED] {symbol} — {error_msg}")
            logger.error("Partial close FAILED: %s — %s", symbol, error_msg)
            return {
                "position_id": position_id,
                "symbol": symbol,
                "status": "failed",
                "closed_qty": 0.0,
                "error": error_msg,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _last_fill_price(self, symbol: str) -> float:
        """Return the most recent avg_fill_price for *symbol* from the orders table.

        Falls back to the position's entry_price when no fill record exists
        (e.g. in tests or if the market order was not yet persisted).
        """
        row = self._conn.execute(
            """
            SELECT avg_fill_price FROM orders
            WHERE symbol=? AND status IN ('filled', 'closed') AND avg_fill_price IS NOT NULL
            ORDER BY created_at DESC LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if row and row["avg_fill_price"]:
            return float(row["avg_fill_price"])

        # Fallback: use entry_price of the open position
        pos = self._conn.execute(
            "SELECT entry_price FROM positions WHERE symbol=? AND status='open'",
            (symbol,),
        ).fetchone()
        return float(pos["entry_price"]) if pos else 0.0

    def _notify(self, message: str) -> None:
        if self._telegram is None:
            return
        try:
            self._telegram.send_alert(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram notify failed: %s", exc)
