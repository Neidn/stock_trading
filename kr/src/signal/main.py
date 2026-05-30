"""Signal Engine + Safety Monitor pod entry point.

Wires:
  KISRestClient → OrderManager → SignalEngine (signal gen + order placement)
  KISWSManager  → SafetyMonitor (2s SL/TP price monitoring)

All components share the same sqlite3 connection and run as concurrent async tasks.
Health server on port 8080 for K8s liveness/readiness.
"""

from __future__ import annotations

import asyncio
import os
import signal

from src.db.connection import get_connection
from src.db.models import init_db
from src.execution.order_manager import OrderManager
from src.ingest.kis_rest import KISRestClient
from src.ingest.kis_ws import KISWSManager
from src.monitoring.health import start_health_server
from src.monitoring.logger import get_logger
from src.monitoring.telegram_bot import get_telegram_bot
from src.risk.market_hours import is_market_open, seconds_until_open
from src.safety.safety_monitor import SafetyMonitor
from src.signal.signal_engine import SignalEngine
from src.signal.strategy_runner import StrategyRunner

logger = get_logger("signal.main")


def _get_symbols(conn) -> list[str]:
    rows = conn.execute("SELECT symbol FROM symbols WHERE is_active=1").fetchall()
    return [r["symbol"] for r in rows] if rows else []


async def _price_noop(symbol: str, price: int) -> None:
    """Placeholder price callback — SafetyMonitor reads last_prices directly."""


async def main() -> None:
    db_path = os.getenv("SQLITE_DB_PATH", "/data/trading.db")
    conn = get_connection(db_path)
    init_db(db_path)

    telegram = get_telegram_bot(conn=conn)

    # Gate: sleep until market opens
    if not is_market_open(buffer_open_sec=0):
        wait_sec = seconds_until_open()
        if wait_sec > 0:
            logger.info("signal.pre_market sleep=%.0fs", wait_sec)
            await asyncio.sleep(wait_sec)

    async with KISRestClient() as kis:
        order_manager = OrderManager(conn=conn, kis=kis, telegram_bot=telegram)
        strategy_runner = StrategyRunner(conn=conn)
        engine = SignalEngine(
            conn=conn,
            strategy_runner=strategy_runner,
            order_manager=order_manager,
            kis=kis,
        )

        async with KISWSManager(
            price_callback=_price_noop,
            paper=(os.getenv("TRADING_MODE", "paper") != "live"),
            telegram_bot=telegram,
        ) as ws:
            # Subscribe WS to all active symbols
            for sym in _get_symbols(conn):
                await ws.subscribe(sym)

            safety_monitor = SafetyMonitor(
                conn=conn,
                order_manager=order_manager,
                ws_manager=ws,
                kis=kis,
                telegram_bot=telegram,
            )

            health = start_health_server("signal-engine", db_conn=conn)

            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop_event.set)

            logger.info("signal.main.running strategy=%s",
                        strategy_runner.get_active_strategy_name())

            # Run signal engine + safety monitor concurrently
            engine_task   = asyncio.create_task(engine._async_run_forever(), name="signal_engine")
            monitor_task  = asyncio.create_task(safety_monitor.run_forever(), name="safety_monitor")

            await stop_event.wait()
            engine_task.cancel()
            monitor_task.cancel()
            await asyncio.gather(engine_task, monitor_task, return_exceptions=True)
            health.stop()

    logger.info("signal.main.shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
