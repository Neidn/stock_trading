from __future__ import annotations

import os
import sys

import structlog

from screener.config import Settings
from screener.kis_client import KISClient
from screener.notifier import TelegramNotifier
from screener.screener import StockScreener

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)

log = structlog.get_logger()


def main() -> int:
    settings = Settings()
    run_mode = os.getenv("RUN_MODE", "pre_market")  # pre_market | final
    apply_gap = run_mode == "final"

    log.info("screener.start", run_mode=run_mode, market=settings.market)

    client = KISClient(settings)
    screener = StockScreener(client, settings)
    notifier = TelegramNotifier(settings)

    candidates = screener.run(apply_gap_filter=apply_gap)

    if not candidates:
        log.warning("screener.no_candidates")
        notifier.send([], run_mode=run_mode)
        return 0

    notifier.send(candidates, run_mode=run_mode)
    log.info("screener.complete", count=len(candidates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
