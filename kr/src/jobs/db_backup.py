"""Daily SQLite backup — hot backup via sqlite3.Connection.backup(), keeps last 7."""
import logging
import os
import sqlite3
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("SQLITE_DB_PATH", "/data/trading.db"))
KEEP = 7


def run() -> None:
    if not DB_PATH.exists():
        logger.error("DB not found: %s", DB_PATH)
        raise FileNotFoundError(DB_PATH)

    backup_dir = DB_PATH.parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    dst = backup_dir / f"trading_{date.today().isoformat()}.db"

    src_conn = sqlite3.connect(DB_PATH)
    dst_conn = sqlite3.connect(dst)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    logger.info("Backup written: %s (%.1f MB)", dst.name, dst.stat().st_size / 1e6)

    old = sorted(backup_dir.glob("trading_*.db"))[:-KEEP]
    for f in old:
        f.unlink()
        logger.info("Pruned: %s", f.name)

    remaining = sorted(backup_dir.glob("trading_*.db"))
    logger.info("Kept %d backup(s): %s … %s", len(remaining), remaining[0].name, remaining[-1].name)


if __name__ == "__main__":
    run()
