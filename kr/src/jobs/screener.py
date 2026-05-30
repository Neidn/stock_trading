"""KRX stock screener CronJob — runs at 08:00 and 08:50 KST.

Two-stage filter pipeline:
  Stage 1 (API-level): Fetch volume + fluctuation rankings from KIS.
      Pre-filter: change_pct ≥ MIN_CHANGE_PCT, trade_amount ≥ MIN_TRADE_AMT
  Stage 2 (enrichment): Fetch individual price data for top candidates.
      Post-filter: market_cap ≥ MIN_MARKET_CAP

Scoring (0–100):
  40 pts : change_pct (normalized to 0–15%)
  30 pts : trade_amount (normalized to 0–100억)
  20 pts : volume rank position bonus
  10 pts : appears in both volume AND fluctuation rankings

Top N by total score written to ``symbols`` table.
Symbols with open positions are never deactivated.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingest.kis_rest import KISRestClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filter thresholds
# ---------------------------------------------------------------------------
MIN_CHANGE_PCT: float = 2.5      # minimum positive gap (%)
MIN_TRADE_AMT_WON: int = 500_000_000   # 5억원 최소 거래대금
MIN_MARKET_CAP_100M: int = 300   # 300억원 최소 시가총액
TOP_N: int = 10
RANKING_FETCH_N: int = 30        # items to fetch per market per ranking type


class ScreenerJob:
    """Morning screener for high-volume gapping KRX stocks.

    Args:
        kis: Open KISRestClient instance.
        conn: SQLite connection.
        telegram_bot: Optional; ``send_info(str)`` used.
        top_n: Max symbols to activate.
    """

    def __init__(
        self,
        kis: "KISRestClient",
        conn: sqlite3.Connection,
        telegram_bot=None,
        top_n: int = TOP_N,
    ) -> None:
        self._kis = kis
        self._conn = conn
        self._telegram = telegram_bot
        self._top_n = top_n

    async def run(self) -> dict:
        """Execute screening and update DB.

        Returns:
            ``{"total_screened": int, "added": list[str], "removed": list[str]}``
        """
        logger.info("screener.start")

        # Stage 1: fetch rankings for KOSPI and KOSDAQ
        vol_j, vol_q, flu_j, flu_q = await asyncio.gather(
            self._safe_fetch_volume("J"),
            self._safe_fetch_volume("Q"),
            self._safe_fetch_fluctuation("J"),
            self._safe_fetch_fluctuation("Q"),
        )

        vol_all = vol_j + vol_q
        flu_all = flu_j + flu_q

        flu_set: set[str] = {s["symbol"] for s in flu_all}

        # Deduplicate by symbol, keep highest change_pct entry
        by_symbol: dict[str, dict] = {}
        for item in vol_all + flu_all:
            sym = item["symbol"]
            existing = by_symbol.get(sym)
            if existing is None or _parse_float(item["change_pct"]) > _parse_float(existing["change_pct"]):
                by_symbol[sym] = item

        # Stage 1 filter
        candidates = [
            s for s in by_symbol.values()
            if _parse_float(s["change_pct"]) >= MIN_CHANGE_PCT
            and _parse_int(s["trade_amount"]) >= MIN_TRADE_AMT_WON
        ]

        if not candidates:
            logger.warning("screener.no_candidates after stage1 filter")
            return {"total_screened": 0, "added": [], "removed": []}

        # Stage 2: enrich top 3×N candidates with individual price data
        candidates.sort(key=lambda s: _parse_float(s["change_pct"]), reverse=True)
        enrich_pool = candidates[: self._top_n * 3]

        enriched = await self._enrich(enrich_pool)

        # Stage 2 filter: market_cap
        filtered = [
            s for s in enriched
            if _parse_int(s.get("market_cap", "0")) >= MIN_MARKET_CAP_100M
        ]

        if not filtered:
            logger.warning("screener.no_candidates after stage2 filter")
            return {"total_screened": 0, "added": [], "removed": []}

        # Score
        scored = sorted(filtered, key=lambda s: _score(s, flu_set), reverse=True)
        selected = scored[: self._top_n]

        new_syms = [s["symbol"] for s in selected]
        old_syms = set(self._get_active_symbols())
        new_set = set(new_syms)

        added = sorted(new_set - old_syms)
        removed = [
            sym for sym in sorted(old_syms - new_set)
            if not self._conn.execute(
                "SELECT 1 FROM positions WHERE symbol=? AND status='open'", (sym,)
            ).fetchone()
        ]

        self._update_symbols_table(selected, removed)

        logger.info(
            "screener.done selected=%d added=%d removed=%d",
            len(selected), len(added), len(removed),
        )
        self._notify(selected, added, removed)
        return {"total_screened": len(selected), "added": added, "removed": removed}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _safe_fetch_volume(self, market: str) -> list[dict]:
        try:
            return await self._kis.fetch_volume_ranking(market=market, top_n=RANKING_FETCH_N)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_volume_ranking market=%s failed: %s", market, exc)
            return []

    async def _safe_fetch_fluctuation(self, market: str) -> list[dict]:
        try:
            return await self._kis.fetch_fluctuation_ranking(
                market=market, top_n=RANKING_FETCH_N,
                min_change=MIN_CHANGE_PCT, max_change=29.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_fluctuation_ranking market=%s failed: %s", market, exc)
            return []

    async def _enrich(self, candidates: list[dict]) -> list[dict]:
        """Fetch individual price data (market_cap) for each candidate."""
        results = []
        for item in candidates:
            try:
                detail = await self._kis.fetch_current_price(item["symbol"])
                enriched = {**item, "market_cap": detail.get("market_cap", "0")}
                results.append(enriched)
            except Exception as exc:  # noqa: BLE001
                logger.warning("enrich failed symbol=%s: %s", item["symbol"], exc)
                results.append({**item, "market_cap": "0"})
        return results

    def _get_active_symbols(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT symbol FROM symbols WHERE is_active=1"
        ).fetchall()
        return [r[0] if not hasattr(r, "keys") else r["symbol"] for r in rows]

    def _update_symbols_table(self, selected: list[dict], removed: list[str]) -> None:
        for sym in removed:
            self._conn.execute("UPDATE symbols SET is_active=0 WHERE symbol=?", (sym,))

        for item in selected:
            sym = item["symbol"]
            name = item.get("name", sym)
            market = "KOSPI" if item.get("market_code") == "J" else "KOSDAQ"
            market_cap = item.get("market_cap", "0")
            self._conn.execute(
                """
                INSERT INTO symbols (symbol, base_asset, quote_asset, is_active, market, market_cap)
                VALUES (?, ?, 'KRW', 1, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    is_active=1, market=excluded.market, market_cap=excluded.market_cap
                """,
                (sym, name, market, market_cap),
            )
        self._conn.commit()

    def _notify(self, selected: list[dict], added: list[str], removed: list[str]) -> None:
        if self._telegram is None:
            return
        lines = [f"📡 스크리너 완료: {len(selected)}개 선별"]
        for i, s in enumerate(selected, 1):
            change = _parse_float(s.get("change_pct", "0"))
            amt_100m = _parse_int(s.get("trade_amount", "0")) // 100_000_000
            lines.append(
                f"{i}. {s.get('name', s['symbol'])} ({s['symbol']}) "
                f"+{change:.1f}% 거래대금 {amt_100m}억"
            )
        if added:
            lines.append(f"신규: {', '.join(added)}")
        if removed:
            lines.append(f"제외: {', '.join(removed)}")
        try:
            self._telegram.send_info("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram notify failed: %s", exc)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(item: dict, flu_set: set[str]) -> float:
    """Score 0–100 for a screener candidate."""
    change = min(_parse_float(item.get("change_pct", "0")), 15.0)
    amt = _parse_int(item.get("trade_amount", "0"))

    score  = (change / 15.0) * 40.0                     # 40pts: change %
    score += min(amt / 10_000_000_000, 1.0) * 30.0      # 30pts: trade amount (100억 cap)
    score += 20.0 if item["symbol"] in flu_set else 0.0  # 20pts: in fluctuation ranking
    score += min(_parse_int(item.get("market_cap", "0")) / 1000, 1.0) * 10.0  # 10pts: cap
    return score


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_float(s) -> float:
    try:
        return float(s or 0)
    except (ValueError, TypeError):
        return 0.0


def _parse_int(s) -> int:
    try:
        return int(float(s or 0))
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _run() -> None:
    from src.db.connection import get_connection
    from src.db.models import init_db
    from src.ingest.kis_rest import KISRestClient
    from src.monitoring.telegram_bot import get_telegram_bot

    db_path = os.getenv("SQLITE_DB_PATH", "/data/trading.db")
    conn = get_connection(db_path)
    init_db(db_path)
    telegram = get_telegram_bot(conn=conn)

    async with KISRestClient() as kis:
        result = await ScreenerJob(kis=kis, conn=conn, telegram_bot=telegram).run()
    logger.info("screener.result %s", result)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
