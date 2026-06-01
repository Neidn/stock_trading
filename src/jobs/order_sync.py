"""One-time (and periodic) order sync — marks stale DB orders as canceled.

Fetches all open orders from KIS API, compares with DB orders where
status='open'.  Any DB order not found on KIS is marked 'canceled'.
Never modifies KIS state.

Run manually:
    kubectl exec -n trading-stock-kr <signal-engine-pod> -- python -m src.jobs.order_sync
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class OrderSyncJob:
    """Sync orders table against live KIS open orders.

    Args:
        kis: KISRestClient instance.
        conn: SQLite connection.
    """

    def __init__(self, kis, conn: sqlite3.Connection) -> None:
        self._kis = kis
        self._conn = conn

    def run(self) -> dict:
        return asyncio.run(self.async_run())

    async def async_run(self) -> dict:
        """Mark stale DB orders as canceled.

        Returns:
            {"canceled": int, "kept": int, "error": str | None}
        """
        try:
            kis_orders = await self._kis.fetch_unfilled_orders()
            kis_order_nos: set[str] = {
                o["order_no"] for o in kis_orders if o.get("order_no")
            }

            db_orders = self._conn.execute(
                "SELECT order_id, broker_order_id, symbol FROM orders WHERE status='open'"
            ).fetchall()

            canceled = 0
            kept = 0
            now = datetime.now(timezone.utc).isoformat()

            for row in db_orders:
                broker_oid = row["broker_order_id"] if hasattr(row, "keys") else row[1]
                sym        = row["symbol"]           if hasattr(row, "keys") else row[2]
                oid        = row["order_id"]         if hasattr(row, "keys") else row[0]

                if broker_oid is None or str(broker_oid) not in kis_order_nos:
                    self._conn.execute(
                        "UPDATE orders SET status='canceled', updated_at=? WHERE order_id=?",
                        (now, oid),
                    )
                    canceled += 1
                    logger.info(
                        "Marked canceled (not on KIS): broker_id=%s symbol=%s",
                        broker_oid, sym,
                    )
                else:
                    kept += 1

            self._conn.commit()
            logger.info("OrderSyncJob done: %d canceled, %d kept", canceled, kept)
            return {"canceled": canceled, "kept": kept, "error": None}

        except Exception as exc:  # noqa: BLE001
            logger.error("OrderSyncJob failed: %s", exc, exc_info=True)
            return {"canceled": 0, "kept": 0, "error": str(exc)}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    import asyncio

    from src.db.connection import get_connection
    from src.db.models import init_db
    from src.ingest.kis_rest import KISRestClient
    from src.utils.config import load_config

    config = load_config()
    init_db(config.sqlite_db_path)
    conn = get_connection(config.sqlite_db_path)

    async def _run() -> None:
        async with KISRestClient() as kis:
            result = await OrderSyncJob(kis=kis, conn=conn).async_run()
            print(
                f"Done — canceled={result['canceled']} "
                f"kept={result['kept']} error={result['error']}"
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
