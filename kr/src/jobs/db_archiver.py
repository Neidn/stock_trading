"""DB archiver CronJob — runs every Sunday at 00:00 UTC.

Keeps last KEEP_CANDLES rows per (symbol, interval_type) and runs VACUUM to
reclaim disk space.  Screener needs 230 candles (SMA-200 + warmup); 500 gives
2× safety margin.
"""

from __future__ import annotations

import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

KEEP_CANDLES: int = 500


class DbArchiverJob:
    """Prune klines to a rolling window and compact the SQLite file.

    Args:
        conn: SQLite connection.
        keep_candles: Rows to retain per (symbol, interval_type) (default 500).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        keep_candles: int = KEEP_CANDLES,
    ) -> None:
        self._conn = conn
        self._keep = keep_candles

    def run(self) -> dict:
        """Prune all intervals and VACUUM.

        Returns:
            ``{"deleted_rows": int, "vacuumed": bool}``
        """
        combos = self._conn.execute(
            "SELECT DISTINCT symbol, interval_type FROM klines"
        ).fetchall()

        total_deleted = 0
        for row in combos:
            symbol, interval = row[0], row[1]
            cutoff_row = self._conn.execute(
                """SELECT open_time FROM klines
                   WHERE symbol=? AND interval_type=?
                   ORDER BY open_time DESC
                   LIMIT 1 OFFSET ?""",
                (symbol, interval, self._keep),
            ).fetchone()

            if cutoff_row is None:
                continue

            cutoff = cutoff_row[0]
            cursor = self._conn.execute(
                """DELETE FROM klines
                   WHERE symbol=? AND interval_type=? AND open_time <= ?""",
                (symbol, interval, cutoff),
            )
            deleted = cursor.rowcount
            total_deleted += deleted
            if deleted:
                logger.info(
                    "Pruned %d rows [%s %s] (cutoff open_time=%s)",
                    deleted, symbol, interval, cutoff,
                )

        self._conn.commit()
        logger.info("Total deleted: %d rows", total_deleted)

        self._conn.execute("VACUUM")
        logger.info("VACUUM complete")

        result = {"deleted_rows": total_deleted, "vacuumed": True}
        logger.info("DbArchiverJob done: %s", result)
        return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    db_path = os.environ.get("SQLITE_DB_PATH", "/data/trading.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    DbArchiverJob(conn=conn).run()


if __name__ == "__main__":
    main()
