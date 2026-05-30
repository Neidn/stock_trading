"""Position sync CronJob — runs every 30 minutes.

Compares live KIS API holdings with DB positions:
  missing_in_db  : Holding on KIS but not in DB → insert with fallback SL
  ghost_position : Position in DB but no KIS holding → mark closed
  quantity_mismatch: Quantity differs → update DB to match API

KRX long-only spot: no leverage, no liquidation price.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PositionSyncJob:
    """Reconcile DB positions with live KIS holdings.

    Args:
        kis: KISRestClient instance.
        conn: SQLite connection.
        telegram_bot: Optional; send_warning / send_critical used.
    """

    def __init__(self, kis, conn: sqlite3.Connection, telegram_bot=None) -> None:
        self._kis = kis
        self._conn = conn
        self._telegram = telegram_bot

    def run(self) -> dict:
        return asyncio.run(self.async_run())

    async def async_run(self) -> dict:
        """Execute sync and return result summary."""
        try:
            api_positions = await self._fetch_api_positions()
            db_positions = self._fetch_db_positions()

            discrepancies = self._find_discrepancies(api_positions, db_positions)
            for disc in discrepancies:
                self._resolve(disc, api_positions)

            if discrepancies:
                self._notify_warning(f"포지션 불일치 {len(discrepancies)}건 자동 수정 완료")

            self._log_sync_event(success=True, discrepancies=len(discrepancies))
            logger.info("PositionSyncJob done: %d discrepancies resolved", len(discrepancies))
            return {"success": True, "discrepancies": len(discrepancies), "error": None}

        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            logger.error("PositionSyncJob failed: %s", error_msg, exc_info=True)
            self._notify_critical(f"포지션 동기화 실패: {error_msg}")
            self._log_sync_event(success=False, discrepancies=0, error=error_msg)
            return {"success": False, "discrepancies": 0, "error": error_msg}

    # ------------------------------------------------------------------
    # Data fetch
    # ------------------------------------------------------------------

    async def _fetch_api_positions(self) -> dict[str, dict]:
        """Return active KIS holdings keyed by symbol."""
        holdings = await self._kis.fetch_positions()
        return {
            h["symbol"]: {
                "symbol":     h["symbol"],
                "side":       "long",
                "quantity":   float(h["positionAmt"]),
                "entry_price": float(h["entryPrice"] or 0),
            }
            for h in holdings
            if float(h.get("positionAmt") or 0) > 0
        }

    def _fetch_db_positions(self) -> dict[str, dict]:
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE status='open'"
        ).fetchall()
        return {r["symbol"]: dict(r) for r in rows}

    # ------------------------------------------------------------------
    # Discrepancy detection
    # ------------------------------------------------------------------

    def _find_discrepancies(
        self, api: dict[str, dict], db: dict[str, dict]
    ) -> list[dict]:
        result = []

        for symbol, api_pos in api.items():
            if symbol not in db:
                result.append({"type": "missing_in_db", "symbol": symbol, "api": api_pos})
                continue
            db_pos = db[symbol]
            qty_diff = abs(api_pos["quantity"] - float(db_pos.get("quantity") or 0))
            if qty_diff > 0.5:
                result.append({
                    "type": "quantity_mismatch",
                    "symbol": symbol,
                    "api": api_pos,
                    "db": db_pos,
                })

        for symbol in db:
            if symbol not in api:
                result.append({"type": "ghost_position", "symbol": symbol, "db": db[symbol]})

        return result

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve(self, disc: dict, api_positions: dict[str, dict]) -> None:
        disc_type = disc["type"]
        symbol    = disc["symbol"]
        now       = datetime.now(timezone.utc).isoformat()

        if disc_type == "missing_in_db":
            api_pos = disc["api"]
            entry   = api_pos["entry_price"]
            # Fallback SL: 3% below entry (KRX long-only)
            fallback_sl = str(round(entry * 0.97))
            self._conn.execute(
                """INSERT OR IGNORE INTO positions
                   (position_id, symbol, side, leverage, entry_price, quantity,
                    liquidation_price, stop_loss, initial_stop_loss, status, close_reason,
                    trading_mode, opened_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()), symbol, "long", 1,
                    str(entry), str(int(api_pos["quantity"])),
                    "0",           # no liquidation for KRX spot
                    fallback_sl, fallback_sl,
                    "open", None, "paper", now,
                ),
            )
            self._conn.commit()
            logger.warning("Inserted missing position from KIS: %s", symbol)
            self._notify_warning(f"⚠️ 미추적 포지션 발견 후 복원: {symbol}")

        elif disc_type == "quantity_mismatch":
            api_pos = disc["api"]
            self._conn.execute(
                "UPDATE positions SET quantity=? WHERE symbol=? AND status='open'",
                (str(int(api_pos["quantity"])), symbol),
            )
            self._conn.commit()
            logger.warning("Corrected qty mismatch for %s", symbol)

        elif disc_type == "ghost_position":
            db_pos = disc["db"]
            self._conn.execute(
                """UPDATE positions
                   SET status='closed', close_reason='external_close',
                       closed_at=?
                   WHERE symbol=? AND status='open'""",
                (now, symbol),
            )
            self._conn.commit()
            logger.warning("Marked ghost position as closed: %s", symbol)
            self._notify_warning(f"⚠️ 외부 청산 감지: {symbol}")

    # ------------------------------------------------------------------
    # DB logging
    # ------------------------------------------------------------------

    def _log_sync_event(
        self, success: bool, discrepancies: int, error: str | None = None
    ) -> None:
        try:
            self._conn.execute(
                """INSERT INTO sync_events (event_id, success, discrepancies, error_message)
                   VALUES (?,?,?,?)""",
                (str(uuid.uuid4()), 1 if success else 0, discrepancies, error),
            )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to log sync_event: %s", exc)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _notify_warning(self, message: str) -> None:
        if self._telegram is None:
            return
        try:
            self._telegram.send_warning(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram notify failed: %s", exc)

    def _notify_critical(self, message: str) -> None:
        if self._telegram is None:
            return
        try:
            self._telegram.send_critical(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram notify failed: %s", exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    import asyncio

    from src.db.connection import get_connection
    from src.db.models import init_db
    from src.ingest.kis_rest import KISRestClient
    from src.monitoring.telegram_bot import get_telegram_bot
    from src.utils.config import load_config

    config = load_config()
    init_db(config.sqlite_db_path)
    conn = get_connection(config.sqlite_db_path)
    telegram = get_telegram_bot(conn=conn)

    async def _run() -> None:
        async with KISRestClient() as kis:
            await PositionSyncJob(kis=kis, conn=conn, telegram_bot=telegram).async_run()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
