"""Data Ingest pod entry point for KRX spot trading.

On startup:
  1. Init DB (apply migrations).
  2. Gate on KRX market hours — if pre-market, sleep until 09:00 KST.
  3. Fetch historical OHLCV klines for all active symbols (backfill).
  4. Start KISWSManager for real-time price ticks (H0STCNT0).
  5. Run until market close (15:30 KST) or SIGTERM/SIGINT.

Symbols sourced from SYMBOLS env var (comma-separated) or DB active symbols.
"""

from __future__ import annotations

import asyncio
import os
import signal

from src.db.connection import get_connection
from src.db.models import init_db, insert_kline
from src.ingest.kis_rest import KISRestClient
from src.ingest.kis_ws import KISWSManager
from src.monitoring.health import start_health_server
from src.monitoring.logger import get_logger
from src.monitoring.telegram_bot import get_telegram_bot
from src.risk.market_hours import is_market_open, seconds_until_open, is_closing_soon

logger = get_logger("ingest.main")

_DEFAULT_SYMBOLS = ["005930", "000660"]   # 삼성전자, SK하이닉스


def _get_symbols(conn) -> list[str]:
    env_symbols = os.getenv("SYMBOLS", "").strip()
    if env_symbols:
        return [s.strip() for s in env_symbols.split(",") if s.strip()]
    rows = conn.execute(
        "SELECT symbol FROM symbols WHERE is_active = 1"
    ).fetchall()
    if rows:
        return [r["symbol"] for r in rows]
    logger.warning("No active symbols in DB and SYMBOLS env not set — using defaults")
    return _DEFAULT_SYMBOLS


async def _backfill_klines(conn, kis: KISRestClient, symbols: list[str]) -> None:
    """Fetch daily OHLCV for each symbol and insert into DB."""
    for sym in symbols:
        try:
            rows = await kis.fetch_klines(sym, interval="D", limit=100)
            for row in rows:
                try:
                    insert_kline(conn, row)
                except Exception:  # noqa: BLE001
                    pass
            conn.commit()
            logger.info("klines.backfilled symbol=%s rows=%d", sym, len(rows))
        except Exception as exc:  # noqa: BLE001
            logger.warning("klines.backfill_failed symbol=%s: %s", sym, exc)


async def _price_callback(symbol: str, price: int) -> None:
    """Called by KISWSManager on each real-time price tick."""
    logger.debug("tick symbol=%s price=%d", symbol, price)


async def main() -> None:
    db_path = os.getenv("SQLITE_DB_PATH", "/data/trading.db")
    conn = get_connection(db_path)
    init_db(db_path)

    symbols = _get_symbols(conn)
    logger.info("ingest.symbols count=%d symbols=%s", len(symbols), symbols)

    bot = get_telegram_bot(conn=conn)
    health = start_health_server("data-ingest", db_conn=conn)

    # Gate: sleep until KRX market opens (09:00 KST)
    if not is_market_open(buffer_open_sec=0):
        wait_sec = seconds_until_open()
        if wait_sec > 0:
            logger.info("ingest.pre_market sleeping=%.0fs until open", wait_sec)
            await asyncio.sleep(wait_sec)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    async with KISRestClient() as kis:
        # Backfill historical klines for strategy indicators
        await _backfill_klines(conn, kis, symbols)

        # Start real-time price WebSocket
        async with KISWSManager(
            price_callback=_price_callback,
            paper=(os.getenv("TRADING_MODE", "paper") != "live"),
            telegram_bot=bot,
        ) as ws:
            for sym in symbols:
                await ws.subscribe(sym)

            logger.info("ingest.running")

            # Run until market close or stop signal
            while not stop_event.is_set():
                if is_closing_soon(buffer_min=1):
                    logger.info("ingest.market_closing — shutting down")
                    break
                await asyncio.sleep(30)

            health.stop()

    logger.info("ingest.shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
