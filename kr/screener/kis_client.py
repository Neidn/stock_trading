from __future__ import annotations

import dataclasses
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import requests
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from screener.config import Settings

log = structlog.get_logger()

# fid_trgt_exls_cls_code: 10-digit flag string
# Positions: 투자위험/경고/주의(1) 관리종목(2) 정리매매(3) 불성실공시(4) 우선주(5)
#            거래정지(6) ETF(7) ETN(8) 신용주문불가(9) SPAC(10)
# Exclude: 관리종목(2), 정리매매(3), 거래정지(6), ETF(7), ETN(8), SPAC(10)
_EXLS_CODE = "0110011101"


@dataclass
class StockData:
    ticker: str
    name: str
    market: str
    current_price: float
    change_pct: float        # 전일 대비 등락률 (signed %)
    volume: int              # 당일 누적 거래량
    prev_volume: int         # 전일 거래량 (0 if not from ranking API)
    trade_amount: float      # 당일 거래대금 (원; 0 if not from ranking API)
    # Populated by enrich_stock_data():
    market_cap: float = 0.0          # 시가총액 (원)
    open_price: float = 0.0          # 당일 시가
    prev_close: float = 0.0          # 전일 종가 (기준가)
    is_halted: bool = False          # 임시 거래정지
    is_admin: bool = False           # 관리종목
    is_liquidation: bool = False     # 정리매매

    @property
    def volume_ratio(self) -> float:
        if self.prev_volume == 0:
            return 0.0
        return self.volume / self.prev_volume

    @property
    def change_pct_abs(self) -> float:
        return abs(self.change_pct)

    @property
    def gap_pct(self) -> float:
        if self.prev_close == 0:
            return 0.0
        return (self.open_price - self.prev_close) / self.prev_close * 100


class _RateLimiter:
    """Sliding-window rate limiter: max_calls per period seconds."""

    def __init__(self, max_calls: int, period: float = 1.0) -> None:
        self._max = max_calls
        self._period = period
        self._lock = threading.Lock()
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self._period:
                self._calls.popleft()
            if len(self._calls) >= self._max:
                sleep_for = self._period - (now - self._calls[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self._calls.append(time.monotonic())


class KISClient:
    """
    한국투자증권 Open API 클라이언트.

    - OAuth2 token auto-fetch, refresh 5 min before expiry
    - Rate limit: 18 req/sec (safely under API limit of 20)
    - Retry: 3 attempts, exponential backoff
    - live / paper domain via Settings.kis_mode
    """

    _TOKEN_BUFFER = 300  # seconds before expiry to refresh

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.kis_base_url
        self._app_key = settings.kis_app_key
        self._app_secret = settings.kis_app_secret
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._rate = _RateLimiter(max_calls=18, period=1.0)
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - self._TOKEN_BUFFER:
            return self._token
        self._fetch_token()
        assert self._token is not None
        return self._token

    def _fetch_token(self) -> None:
        log.info("kis.token.fetch")
        resp = self._session.post(
            f"{self._base_url}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self._app_key,
                "appsecret": self._app_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 86400)
        log.info("kis.token.ok", expires_in=data.get("expires_in"))

    # ------------------------------------------------------------------
    # Internal request
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def _get(self, path: str, tr_id: str, params: dict[str, str]) -> dict[str, Any]:
        self._rate.acquire()
        token = self._ensure_token()
        resp = self._session.get(
            f"{self._base_url}{path}",
            headers={
                "content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {token}",
                "appkey": self._app_key,
                "appsecret": self._app_secret,
                "tr_id": tr_id,
            },
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("rt_cd", "-1") != "0":
            raise requests.RequestException(
                f"KIS error rt_cd={body.get('rt_cd')} msg={body.get('msg1', '')}"
            )
        return body

    # ------------------------------------------------------------------
    # Public ranking APIs
    # ------------------------------------------------------------------

    def get_volume_ranking(self, market: str = "J") -> list[StockData]:
        """
        거래량순위 조회.
        market: "J" (KRX, covers KOSPI+KOSDAQ), "UN" (통합)
        Ref: v1_국내주식-047, tr_id FHPST01710000
        """
        body = self._get(
            "/uapi/domestic-stock/v1/quotations/volume-rank",
            tr_id="FHPST01710000",
            params={
                "FID_COND_MRKT_DIV_CODE": market,
                "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": "0000",
                "FID_DIV_CLS_CODE": "0",
                "FID_BLNG_CLS_CODE": "0",     # 0=평균거래량
                "FID_TRGT_CLS_CODE": "111111111",
                "FID_TRGT_EXLS_CLS_CODE": _EXLS_CODE,
                "FID_INPUT_PRICE_1": "",
                "FID_INPUT_PRICE_2": "",
                "FID_VOL_CNT": "",
                "FID_INPUT_DATE_1": "",
            },
        )
        return [
            self._parse_volume_row(row, market)
            for row in body.get("output", [])
            if row.get("mksc_shrn_iscd", "").strip()
        ]

    def get_fluctuation_ranking(self, market: str = "J") -> list[StockData]:
        """
        등락률순위 조회.
        market: "J" (KOSPI/KRX), "Q" (KOSDAQ)
        Ref: v1_국내주식-088, tr_id FHPST01700000
        Note: response lacks prdy_vol and acml_tr_pbmn — enrich before filtering.
        """
        body = self._get(
            "/uapi/domestic-stock/v1/ranking/fluctuation",
            tr_id="FHPST01700000",
            params={
                "fid_cond_mrkt_div_code": market,
                "fid_cond_scr_div_code": "20170",
                "fid_input_iscd": "0000",
                "fid_rank_sort_cls_code": "0",
                "fid_input_cnt_1": "0",
                "fid_prc_cls_code": "0",
                "fid_input_price_1": "",
                "fid_input_price_2": "",
                "fid_vol_cnt": "",
                "fid_trgt_cls_code": "0",
                "fid_trgt_exls_cls_code": _EXLS_CODE,
                "fid_div_cls_code": "0",
                "fid_rsfl_rate1": "",
                "fid_rsfl_rate2": "",
            },
        )
        mkt_label = "KOSPI" if market == "J" else "KOSDAQ"
        return [
            self._parse_fluctuation_row(row, mkt_label)
            for row in body.get("output", [])
            if row.get("stck_shrn_iscd", "").strip()
        ]

    def get_current_price(self, ticker: str) -> dict[str, Any]:
        """주식현재가 시세 (단일 종목). Ref: v1_국내주식-008, tr_id FHKST01010100."""
        body = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": ticker,
            },
        )
        return body.get("output", {})

    def enrich_stock_data(self, stock: StockData) -> StockData:
        """
        Fetch open_price, prev_close, market_cap, halt/admin flags via inquire_price.
        Falls back to original data on API failure.
        """
        try:
            raw = self.get_current_price(stock.ticker)
        except Exception:
            log.warning("kis.enrich.failed", ticker=stock.ticker)
            return stock

        hts_avls = _f(raw, "hts_avls")
        open_price = _f(raw, "stck_oprc")
        prev_close = _f(raw, "stck_sdpr")
        # hts_avls is in 억원; convert to 원
        market_cap = hts_avls * 100_000_000 if hts_avls else 0.0

        # If volume/trade_amount weren't populated from a ranking response, fill them
        volume = _i(raw, "acml_vol") or stock.volume
        trade_amount = _f(raw, "acml_tr_pbmn") or stock.trade_amount

        return dataclasses.replace(
            stock,
            open_price=open_price,
            prev_close=prev_close,
            market_cap=market_cap,
            volume=volume,
            trade_amount=trade_amount,
            is_halted=raw.get("temp_stop_yn", "N") == "Y",
            is_admin=False,  # ranking API already excludes 관리종목 via fid_trgt_exls_cls_code
            is_liquidation=raw.get("sltr_yn", "N") == "Y",
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_volume_row(row: dict[str, Any], market: str) -> StockData:
        """
        Volume rank response fields (confirmed):
        mksc_shrn_iscd, hts_kor_isnm, stck_prpr, prdy_vrss_sign, prdy_vrss,
        prdy_ctrt, acml_vol, prdy_vol, acml_tr_pbmn, data_rank, ...
        NOT present: stck_oprc, stck_sdpr, hts_avls
        """
        mkt_label = "KOSPI" if market == "J" else market
        return StockData(
            ticker=row.get("mksc_shrn_iscd", "").strip(),
            name=row.get("hts_kor_isnm", "").strip(),
            market=mkt_label,
            current_price=_f(row, "stck_prpr"),
            change_pct=_f(row, "prdy_ctrt"),
            volume=_i(row, "acml_vol"),
            prev_volume=_i(row, "prdy_vol"),
            trade_amount=_f(row, "acml_tr_pbmn"),
        )

    @staticmethod
    def _parse_fluctuation_row(row: dict[str, Any], market: str) -> StockData:
        """
        Fluctuation rank response fields (confirmed):
        stck_shrn_iscd, hts_kor_isnm, stck_prpr, prdy_vrss_sign, prdy_vrss,
        prdy_ctrt, acml_vol, stck_hgpr, stck_lwpr, oprc_vrss_prpr_rate, ...
        NOT present: prdy_vol, acml_tr_pbmn, stck_oprc, stck_sdpr, hts_avls
        Ticker field: stck_shrn_iscd (different from volume rank!)
        """
        return StockData(
            ticker=row.get("stck_shrn_iscd", "").strip(),
            name=row.get("hts_kor_isnm", "").strip(),
            market=market,
            current_price=_f(row, "stck_prpr"),
            change_pct=_f(row, "prdy_ctrt"),
            volume=_i(row, "acml_vol"),
            prev_volume=0,      # not in fluctuation rank; filled by enrich
            trade_amount=0.0,   # not in fluctuation rank; filled by enrich
        )


# ------------------------------------------------------------------
# Field extraction helpers
# ------------------------------------------------------------------

def _f(row: dict[str, Any], key: str) -> float:
    return float(row.get(key) or 0)


def _i(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)
