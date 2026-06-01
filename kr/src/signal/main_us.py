"""US signal engine pod entry point.

Wires:
  KISRestClient → OrderManagerUS → SignalEngineUS (signal gen + order placement)

US market hours gate: sleeps until 09:30 ET, shuts down at 16:00 ET.
Health server on port 8080 for K8s liveness/readiness.

Note: No WebSocket / SafetyMonitor in V1 — real-time SL/TP enforcement
      via price polling is planned for V2.
"""

from __future__ import annotations

import asyncio
import os
import signal

from src.db.connection import get_connection
from src.db.models import init_db
from src.execution.order_manager_us import OrderManagerUS
from src.ingest.kis_rest import KISRestClient
from src.monitoring.health import start_health_server
from src.monitoring.logger import get_logger
from src.monitoring.telegram_bot import get_telegram_bot
from src.risk.market_hours import is_us_market_open, seconds_until_us_open
from src.signal.signal_engine_us import SignalEngineUS
from src.signal.strategy_runner import StrategyRunner

logger = get_logger("signal.us.main")

_CYCLE_INTERVAL_SEC = 3600   # run strategy every 60 min (daily candles)


def _get_us_symbols(conn) -> list[str]:
    rows = conn.execute(
        "SELECT symbol FROM symbols WHERE excd IS NOT NULL AND is_active=1"
    ).fetchall()
    return [r["symbol"] for r in rows] if rows else []


async def main() -> None:
    db_path = os.getenv("SQLITE_DB_PATH", "/data/trading.db")
    conn = get_connection(db_path)
    init_db(db_path)

    telegram = get_telegram_bot(conn=conn)
    health = start_health_server("signal-engine-us", db_conn=conn)

    # Gate: sleep until US market opens (09:30 ET)
    if not is_us_market_open():
        wait_sec = seconds_until_us_open()
        if wait_sec > 0:
            logger.info("signal.us.pre_market sleep=%.0fs until 09:30 ET", wait_sec)
            await asyncio.sleep(wait_sec)

    async with KISRestClient() as kis:
        order_manager = OrderManagerUS(conn=conn, kis=kis, telegram_bot=telegram)
        strategy_runner = StrategyRunner(conn=conn)
        engine = SignalEngineUS(
            conn=conn,
            strategy_runner=strategy_runner,
            order_manager=order_manager,
            kis=kis,
        )

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        logger.info(
            "signal.us.main.running strategy=%s",
            strategy_runner.get_active_strategy_name(),
        )

        while not stop_event.is_set():
            try:
                reload_fn = getattr(strategy_runner, "reload_if_changed", None)
                if callable(reload_fn):
                    await asyncio.to_thread(reload_fn)

                results = await engine.process_all_symbols()
                actionable = sum(1 for r in results if r.is_actionable())
                logger.info(
                    "us.cycle complete — %d symbols, %d actionable",
                    len(results), actionable,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Unhandled error in US signal loop: %s", exc, exc_info=True)

            if stop_event.is_set():
                break
            await asyncio.sleep(_CYCLE_INTERVAL_SEC)

        health.stop()

    logger.info("signal.us.main.shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
