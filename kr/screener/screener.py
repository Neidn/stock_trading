from __future__ import annotations

from dataclasses import dataclass

import structlog

from screener.config import Settings
from screener.kis_client import KISClient, StockData

log = structlog.get_logger()


@dataclass
class ScreenedStock:
    data: StockData
    score: float

    @property
    def ticker(self) -> str:
        return self.data.ticker

    @property
    def name(self) -> str:
        return self.data.name


class StockScreener:
    """
    Two-stage filter pipeline:

    Stage 1 — from ranking API responses (cheap, no extra calls):
      - change_pct range  [applies to all candidates]
      - trade_amount      [applies to volume rank candidates only; missing from fluctuation rank]

    Stage 2 — after enrich_stock_data() per remaining candidate:
      - market_cap
      - volume_ratio
      - gap_pct           [apply_gap_filter=True only]
      - halt / admin / liquidation

    API exclusions (fid_trgt_exls_cls_code) already remove 관리종목, 정리매매,
    거래정지, ETF, ETN, SPAC at the ranking query level. Stage 2 checks double-confirm.
    """

    def __init__(self, client: KISClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def run(self, apply_gap_filter: bool = False) -> list[ScreenedStock]:
        s = self._settings

        # ── Stage 1: collect from ranking APIs ────────────────────────
        candidates = self._collect_candidates()
        log.info("screener.collected", count=len(candidates))
        candidates = self._dedupe(candidates)
        log.info("screener.deduped", count=len(candidates))

        # Broad pre-filter: change_pct (available from both ranking APIs)
        candidates = [c for c in candidates if self._pass_change_pct(c, s)]
        log.info("screener.after_change_pct", count=len(candidates))

        # trade_amount pre-filter for stocks where it's already populated
        # (volume rank has it; fluctuation rank does not → keep those through)
        candidates = [
            c for c in candidates
            if c.trade_amount == 0.0 or c.trade_amount >= s.min_trade_amount
        ]
        log.info("screener.after_trade_amount_prefilter", count=len(candidates))

        # ── Stage 2: enrich then full filter ──────────────────────────
        candidates = [self._client.enrich_stock_data(c) for c in candidates]
        log.info("screener.enriched", count=len(candidates))

        candidates = [c for c in candidates if not c.is_halted]
        candidates = [c for c in candidates if not c.is_admin]
        candidates = [c for c in candidates if not c.is_liquidation]
        log.info("screener.after_status_exclusions", count=len(candidates))

        candidates = [c for c in candidates if self._pass_trade_amount(c, s)]
        log.info("screener.after_trade_amount", count=len(candidates))

        candidates = [c for c in candidates if self._pass_market_cap(c, s)]
        log.info("screener.after_market_cap", count=len(candidates))

        candidates = [c for c in candidates if self._pass_volume_ratio(c, s)]
        log.info("screener.after_volume_ratio", count=len(candidates))

        if apply_gap_filter:
            candidates = [c for c in candidates if self._pass_gap(c, s)]
            log.info("screener.after_gap", count=len(candidates))

        # ── Score and rank ─────────────────────────────────────────────
        scored = sorted(
            [ScreenedStock(data=c, score=_calculate_score(c, apply_gap_filter)) for c in candidates],
            key=lambda x: x.score,
            reverse=True,
        )
        result = scored[: s.top_n]
        log.info("screener.done", top_n=len(result))
        return result

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _collect_candidates(self) -> list[StockData]:
        stocks: list[StockData] = []
        markets = self._target_markets()

        for market_code, market_label in markets:
            try:
                stocks.extend(self._client.get_volume_ranking(market_code))
            except Exception:
                log.exception("screener.volume_rank.failed", market=market_code)
            try:
                stocks.extend(self._client.get_fluctuation_ranking(market_code))
            except Exception:
                log.exception("screener.fluctuation_rank.failed", market=market_code)

        return stocks

    def _target_markets(self) -> list[tuple[str, str]]:
        """Returns list of (api_code, label) pairs."""
        m = self._settings.market.upper()
        # Volume rank uses "J" (KRX = KOSPI+KOSDAQ combined)
        # Fluctuation rank separates "J" (KOSPI) and "Q" (KOSDAQ)
        if m == "ALL":
            return [("J", "KRX"), ("Q", "KOSDAQ")]
        if m == "KOSPI":
            return [("J", "KOSPI")]
        if m == "KOSDAQ":
            return [("Q", "KOSDAQ")]
        log.warning("screener.unknown_market", market=m, fallback="ALL")
        return [("J", "KRX"), ("Q", "KOSDAQ")]

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe(stocks: list[StockData]) -> list[StockData]:
        """Keep highest-volume entry per ticker."""
        seen: dict[str, StockData] = {}
        for s in stocks:
            if s.ticker not in seen or s.volume > seen[s.ticker].volume:
                seen[s.ticker] = s
        return list(seen.values())

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    @staticmethod
    def _pass_change_pct(s: StockData, cfg: Settings) -> bool:
        return cfg.min_change_pct <= s.change_pct_abs <= cfg.max_change_pct

    @staticmethod
    def _pass_trade_amount(s: StockData, cfg: Settings) -> bool:
        return s.trade_amount >= cfg.min_trade_amount

    @staticmethod
    def _pass_market_cap(s: StockData, cfg: Settings) -> bool:
        # Skip if market_cap unavailable (enrich failed)
        if s.market_cap == 0.0:
            return True
        return s.market_cap >= cfg.min_market_cap

    @staticmethod
    def _pass_volume_ratio(s: StockData, cfg: Settings) -> bool:
        return s.volume_ratio >= cfg.min_volume_ratio

    @staticmethod
    def _pass_gap(s: StockData, cfg: Settings) -> bool:
        return abs(s.gap_pct) >= cfg.min_gap_pct


# ------------------------------------------------------------------
# Scoring (0–100)
# ------------------------------------------------------------------

def _calculate_score(s: StockData, include_gap: bool = True) -> float:
    """
    가중치:
      거래량 급증 (volume_ratio): 40%
      변동률 절댓값 (change_pct_abs): 30%
      거래대금 (trade_amount, 정규화): 20%
      갭 크기 (gap_pct): 10%
    """
    gap_component = min(abs(s.gap_pct) / 3, 1.0) * 10 if include_gap else 0.0
    score = (
        min(s.volume_ratio / 10, 1.0) * 40
        + min(s.change_pct_abs / 10, 1.0) * 30
        + min(s.trade_amount / 100_000_000_000, 1.0) * 20
        + gap_component
    )
    return round(score, 2)
