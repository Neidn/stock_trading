"""US stock data ingest pod entry point (KIS overseas REST API).

On startup:
  1. Init DB (apply migrations).
  2. Sleep until US market opens (09:30 ET) if pre-market.
  3. Backfill 250 daily candles for all active US symbols.
  4. Refresh candles every 60 minutes during market hours.
  5. Exit at market close (16:00 ET) or SIGTERM/SIGINT.

Active US symbols: rows in symbols table where excd IS NOT NULL AND is_active=1.
"""

from __future__ import annotations

import asyncio
import os
import signal

from src.db.connection import get_connection
from src.db.models import init_db, insert_kline
from src.ingest.kis_rest import KISRestClient
from src.monitoring.health import start_health_server
from src.monitoring.logger import get_logger
from src.monitoring.telegram_bot import get_telegram_bot
from src.risk.market_hours import is_us_market_open, seconds_until_us_open, is_us_closing_soon

logger = get_logger("ingest.us.main")

_BACKFILL_LIMIT = 250
_REFRESH_INTERVAL_SEC = 3600  # re-fetch latest candles every 60 min during market hours


def _get_us_symbols(conn) -> list[tuple[str, str]]:
    """Return [(symbol, excd), ...] for active US symbols."""
    env_symbols = os.getenv("US_SYMBOLS", "").strip()
    if env_symbols:
        # Format: "AAPL:NAS,TSLA:NAS,NVDA:NAS"
        result = []
        for part in env_symbols.split(","):
            part = part.strip()
            if ":" in part:
                sym, excd = part.split(":", 1)
                result.append((sym.strip().upper(), excd.strip().upper()))
            elif part:
                result.append((part.upper(), "NAS"))
        return result

    rows = conn.execute(
        "SELECT symbol, excd FROM symbols WHERE excd IS NOT NULL AND is_active=1"
    ).fetchall()
    if rows:
        return [(r["symbol"], r["excd"] or "NAS") for r in rows]

    logger.warning("No active US symbols in DB and US_SYMBOLS env not set")
    return []


async def _backfill_klines(
    conn,
    kis: KISRestClient,
    symbols: list[tuple[str, str]],
    limit: int = _BACKFILL_LIMIT,
) -> None:
    for sym, excd in symbols:
        try:
            rows = await kis.fetch_klines_us(sym, excd=excd, limit=limit)
            inserted = 0
            for row in rows:
                try:
                    insert_kline(conn, row)
                    inserted += 1
                except Exception:  # noqa: BLE001
                    pass
            conn.commit()
            logger.info("us.klines.backfilled symbol=%s excd=%s rows=%d", sym, excd, inserted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("us.klines.backfill_failed symbol=%s: %s", sym, exc)


async def _refresh_latest(
    conn,
    kis: KISRestClient,
    symbols: list[tuple[str, str]],
) -> None:
    """Fetch the most recent 5 candles to keep today's bar current."""
    for sym, excd in symbols:
        try:
            rows = await kis.fetch_klines_us(sym, excd=excd, limit=5)
            for row in rows:
                try:
                    insert_kline(conn, row)
                except Exception:  # noqa: BLE001
                    pass
            conn.commit()
            logger.debug("us.klines.refreshed symbol=%s", sym)
        except Exception as exc:  # noqa: BLE001
            logger.warning("us.klines.refresh_failed symbol=%s: %s", sym, exc)


async def main() -> None:
    db_path = os.getenv("SQLITE_DB_PATH", "/data/trading.db")
    conn = get_connection(db_path)
    init_db(db_path)

    symbols = _get_us_symbols(conn)
    logger.info("ingest.us.symbols count=%d symbols=%s", len(symbols), [s for s, _ in symbols])

    get_telegram_bot(conn=conn)
    health = start_health_server("data-ingest-us", db_conn=conn)

    if not is_us_market_open():
        wait_sec = seconds_until_us_open()
        if wait_sec > 0:
            logger.info("ingest.us.pre_market sleeping=%.0fs until 09:30 ET", wait_sec)
            await asyncio.sleep(wait_sec)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    async with KISRestClient() as kis:
        await _backfill_klines(conn, kis, symbols)

        while not stop_event.is_set():
            if is_us_closing_soon(buffer_min=1):
                logger.info("ingest.us.market_closing — shutting down")
                break
            await asyncio.sleep(_REFRESH_INTERVAL_SEC)
            if not stop_event.is_set() and is_us_market_open():
                await _refresh_latest(conn, kis, symbols)

        health.stop()

    logger.info("ingest.us.shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
