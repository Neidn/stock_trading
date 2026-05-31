"""KIS (한국투자증권) Open API async REST client.

Drop-in replacement for BinanceRestClient — same async context-manager
lifecycle, same method signatures where possible.

Endpoints used:
  POST /oauth2/tokenP                                  — access token
  GET  /uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice  — OHLCV
  GET  /uapi/domestic-stock/v1/quotations/inquire-price                 — current price
  GET  /uapi/domestic-stock/v1/trading/inquire-balance                  — holdings + cash
  POST /uapi/domestic-stock/v1/trading/order-cash                       — buy / sell

tr_id mapping (paper vs live):
  Paper buy  : VTTC0802U   Live buy  : TTTC0802U
  Paper sell : VTTC0801U   Live sell : TTTC0801U
  Paper bal  : VTTC8434R   Live bal  : TTTC8434R

Usage::

    async with KISRestClient() as client:
        ohlcv  = await client.fetch_klines("005930", "D", limit=200)
        bal    = await client.fetch_account_balance()
        pos    = await client.fetch_positions()
        price  = await client.fetch_current_price("005930")
        result = await client.place_buy_order("005930", qty=10)
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from src.monitoring.logger import get_logger
from src.utils.api_guard import NetworkError, RateLimitError, with_retry

logger = get_logger("kis_rest")

_PAPER_BASE = "https://openapivts.koreainvestment.com:29443"
_LIVE_BASE = "https://openapi.koreainvestment.com:9443"

_TOKEN_BUFFER_SEC = 300   # refresh 5 min before expiry
_RATE_MAX = 18            # requests per second (KIS limit is 20; stay under)
_RATE_PERIOD = 1.0
_MIN_DELAY = 0.05         # 50 ms floor between requests


class _RateLimiter:
    def __init__(self, max_calls: int = _RATE_MAX, period: float = _RATE_PERIOD) -> None:
        self._max = max_calls
        self._period = period
        self._calls: deque[float] = deque()

    async def acquire(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > self._period:
            self._calls.popleft()
        if len(self._calls) >= self._max:
            sleep_for = self._period - (now - self._calls[0])
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        self._calls.append(time.monotonic())


class KISRestClient:
    """Async KIS Open API REST client.

    Credentials and trading mode are resolved from environment variables::

        KIS_APP_KEY           — order app key (paper or live)
        KIS_APP_SECRET        — order app secret
        KIS_DATA_APP_KEY      — data app key (실전투자; falls back to KIS_APP_KEY)
        KIS_DATA_APP_SECRET   — data app secret (실전투자; falls back to KIS_APP_SECRET)
        KIS_ACCOUNT_NO        — 계좌번호 (format: "XXXXXXXXXX-XX")
        TRADING_MODE          — "paper" (모의투자) | "live"

    GET (market data) always targets openapi.koreainvestment.com (live data server)
    using the data credentials.  POST (orders) targets the server determined by
    TRADING_MODE using the order credentials.
    """

    def __init__(self) -> None:
        mode = os.getenv("TRADING_MODE", "paper").strip().lower()
        self._paper = mode != "live"
        self._base = _PAPER_BASE if self._paper else _LIVE_BASE

        # Order credentials (paper or live)
        self._app_key = os.getenv("KIS_APP_KEY", "")
        self._app_secret = os.getenv("KIS_APP_SECRET", "")

        # Data credentials — fall back to order credentials if not set.
        # Market data APIs (rankings, quotes) only exist on the live server;
        # the paper server (openapivts) returns 404 for these endpoints.
        self._data_app_key = os.getenv("KIS_DATA_APP_KEY", "").strip() or self._app_key
        self._data_app_secret = os.getenv("KIS_DATA_APP_SECRET", "").strip() or self._app_secret

        raw_acct = os.getenv("KIS_ACCOUNT_NO", "")
        # Accept both "XXXXXXXXXX-XX" and "XXXXXXXXXXXX" formats
        if "-" in raw_acct:
            parts = raw_acct.split("-")
            self._acct_no = parts[0]       # 계좌번호 앞자리 10자리
            self._acct_prod = parts[1]     # 상품코드 2자리
        else:
            self._acct_no = raw_acct[:10]
            self._acct_prod = raw_acct[10:] or "01"

        # Order token (paper or live server)
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._token_lock: asyncio.Lock = asyncio.Lock()

        # Data token (always live server)
        self._data_token: str | None = None
        self._data_token_expires_at: float = 0.0
        self._data_token_lock: asyncio.Lock = asyncio.Lock()

        self._rate = _RateLimiter()
        self._session: aiohttp.ClientSession | None = None
        self._last_req_at: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "KISRestClient":
        await self.open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - _TOKEN_BUFFER_SEC:
            return self._token
        async with self._token_lock:
            if self._token and time.time() < self._token_expires_at - _TOKEN_BUFFER_SEC:
                return self._token
            await self._fetch_token()
        assert self._token
        return self._token

    async def _fetch_token(self) -> None:
        logger.info("kis.token.fetch")
        assert self._session
        async with self._session.post(
            f"{self._base}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self._app_key,
                "appsecret": self._app_secret,
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise NetworkError(f"KIS token fetch failed {resp.status}: {text}")
            data = await resp.json(content_type=None)

        self._token = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 86400)
        logger.info("kis.token.ok expires_in=%s", data.get("expires_in"))

    async def _ensure_data_token(self) -> str:
        if self._data_token and time.time() < self._data_token_expires_at - _TOKEN_BUFFER_SEC:
            return self._data_token
        async with self._data_token_lock:
            if self._data_token and time.time() < self._data_token_expires_at - _TOKEN_BUFFER_SEC:
                return self._data_token
            await self._fetch_data_token()
        assert self._data_token
        return self._data_token

    async def _fetch_data_token(self) -> None:
        logger.info("kis.data_token.fetch")
        assert self._session
        async with self._session.post(
            f"{_LIVE_BASE}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self._data_app_key,
                "appsecret": self._data_app_secret,
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise NetworkError(f"KIS data token fetch failed {resp.status}: {text}")
            data = await resp.json(content_type=None)
        self._data_token = data["access_token"]
        self._data_token_expires_at = time.time() + data.get("expires_in", 86400)
        logger.info("kis.data_token.ok expires_in=%s", data.get("expires_in"))

    # ------------------------------------------------------------------
    # Internal request
    # ------------------------------------------------------------------

    def _headers(self, tr_id: str, token: str) -> dict[str, str]:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def _data_headers(self, tr_id: str, token: str) -> dict[str, str]:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self._data_app_key,
            "appsecret": self._data_app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    async def _throttle(self) -> None:
        await self._rate.acquire()
        elapsed = time.monotonic() - self._last_req_at
        if elapsed < _MIN_DELAY:
            await asyncio.sleep(_MIN_DELAY - elapsed)
        self._last_req_at = time.monotonic()

    @with_retry(max_retries=3, delay=1.0)
    async def _get(self, path: str, tr_id: str, params: dict[str, str]) -> dict:
        assert self._session
        await self._throttle()
        token = await self._ensure_data_token()
        async with self._session.get(
            f"{_LIVE_BASE}{path}",
            headers=self._data_headers(tr_id, token),
            params=params,
        ) as resp:
            if resp.status == 429:
                raise RateLimitError(f"KIS rate limit {resp.url}")
            data = await resp.json(content_type=None)
        if not isinstance(data, dict):
            raise NetworkError(f"KIS non-dict response (http={resp.status}): {data!r}")
        if data.get("rt_cd", "-1") != "0":
            raise NetworkError(
                f"KIS error rt_cd={data.get('rt_cd')} msg={data.get('msg1', '')}"
            )
        return data

    @with_retry(max_retries=3, delay=1.0)
    async def _post(self, path: str, tr_id: str, body: dict) -> dict:
        assert self._session
        await self._throttle()
        token = await self._ensure_token()
        async with self._session.post(
            f"{self._base}{path}",
            headers=self._headers(tr_id, token),
            json=body,
        ) as resp:
            if resp.status == 429:
                raise RateLimitError(f"KIS rate limit {resp.url}")
            data = await resp.json(content_type=None)
        if data.get("rt_cd", "-1") != "0":
            raise NetworkError(
                f"KIS order error rt_cd={data.get('rt_cd')} msg={data.get('msg1', '')}"
            )
        return data

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @with_retry(max_retries=3, delay=1.0)
    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "D",
        limit: int = 100,
        since: int | None = None,
    ) -> list[dict]:
        """Fetch OHLCV daily candles for *symbol*.

        Args:
            symbol: KRX ticker, e.g. ``'005930'``.
            interval: ``'D'`` (일봉) only; weekly/monthly not yet supported.
            limit: Number of candles (max 100 per KIS page).
            since: Ignored (KIS returns most-recent *limit* candles).

        Returns:
            List of dicts with keys: symbol, interval_type, open_time, open,
            high, low, close, volume, close_time.
        """
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=limit * 2)).strftime("%Y%m%d")

        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            tr_id="FHKST03010100",
            params={
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": symbol,
                "fid_input_date_1": start_date,
                "fid_input_date_2": end_date,
                "fid_period_div_code": "D",
                "fid_org_adj_prc": "0",
            },
        )

        rows = data.get("output2", [])
        result = []
        for row in rows:
            dt_str = row.get("stck_bsop_date", "")
            if not dt_str:
                continue
            # open_time as ms epoch (KIS returns YYYYMMDD)
            try:
                dt = datetime.strptime(dt_str, "%Y%m%d").replace(
                    hour=9, tzinfo=timezone.utc
                )
                open_ts = int(dt.timestamp() * 1000)
                close_ts = int(dt.replace(hour=15, minute=30).timestamp() * 1000)
            except ValueError:
                continue
            result.append({
                "symbol":        symbol,
                "interval_type": "1d",
                "open_time":     str(open_ts),
                "open":          row.get("stck_oprc", "0"),
                "high":          row.get("stck_hgpr", "0"),
                "low":           row.get("stck_lwpr", "0"),
                "close":         row.get("stck_clpr", "0"),
                "volume":        row.get("acml_vol", "0"),
                "close_time":    str(close_ts),
            })

        # Oldest first (matches Binance convention)
        result.sort(key=lambda r: r["open_time"])
        return result[-limit:]

    @with_retry(max_retries=3, delay=1.0)
    async def fetch_current_price(self, symbol: str) -> dict:
        """Fetch current price snapshot for *symbol*.

        Returns:
            Dict with keys: symbol, price, open, high, low, volume,
            change_pct, market_cap.
        """
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": symbol,
            },
        )
        out = data.get("output", {})
        return {
            "symbol":     symbol,
            "price":      out.get("stck_prpr", "0"),
            "open":       out.get("stck_oprc", "0"),
            "high":       out.get("stck_hgpr", "0"),
            "low":        out.get("stck_lwpr", "0"),
            "volume":     out.get("acml_vol", "0"),
            "change_pct": out.get("prdy_ctrt", "0"),   # 전일 대비 등락률
            "market_cap": out.get("hts_avls", "0"),    # 시가총액 (억원)
        }

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    @with_retry(max_retries=3, delay=1.0)
    async def fetch_account_balance(self) -> dict:
        """Fetch KRW account balance.

        Returns:
            Dict with keys: totalWalletBalance, availableBalance
            (both as string KRW amounts).
        """
        tr_id = "VTTC8434R" if self._paper else "TTTC8434R"
        data = await self._get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id=tr_id,
            params={
                "CANO": self._acct_no,
                "ACNT_PRDT_CD": self._acct_prod,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )
        summary = data.get("output2", [{}])
        s = summary[0] if summary else {}
        return {
            "totalWalletBalance": s.get("tot_evlu_amt", "0"),     # 총평가금액
            "availableBalance":   s.get("prvs_rcdl_excc_amt", "0"),  # 가용현금
        }

    @with_retry(max_retries=3, delay=1.0)
    async def fetch_positions(self) -> list[dict]:
        """Fetch current stock holdings.

        Returns:
            List of dicts with keys: symbol, positionAmt, entryPrice,
            unrealizedProfit, market.
        """
        tr_id = "VTTC8434R" if self._paper else "TTTC8434R"
        data = await self._get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id=tr_id,
            params={
                "CANO": self._acct_no,
                "ACNT_PRDT_CD": self._acct_prod,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )
        holdings = []
        for item in data.get("output1", []):
            qty = int(item.get("hldg_qty", "0") or "0")
            if qty == 0:
                continue
            holdings.append({
                "symbol":           item.get("pdno", ""),      # 종목코드
                "name":             item.get("prdt_name", ""), # 종목명
                "positionAmt":      str(qty),
                "entryPrice":       item.get("pchs_avg_pric", "0"),  # 매입평균가
                "unrealizedProfit": item.get("evlu_pfls_amt", "0"),  # 평가손익
                "market":           "KOSPI",  # KIS balance API doesn't return market; enrich separately
            })
        return holdings

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def place_buy_order(
        self,
        symbol: str,
        qty: int,
        price: int | None = None,
    ) -> dict:
        """Submit a buy order.

        Args:
            symbol: KRX ticker.
            qty: Integer share count.
            price: Limit price (원). None → market order.

        Returns:
            KIS order response output dict.
        """
        tr_id = "VTTC0802U" if self._paper else "TTTC0802U"
        order_type = "00" if price is None else "00"  # 00=시장가, 01=지정가
        # KIS: 00=지정가, 01=시장가... actually:
        # ORD_DVSN 00=지정가, 01=시장가
        if price is None:
            ord_dvsn = "01"  # 시장가
            ord_unpr = "0"
        else:
            ord_dvsn = "00"  # 지정가
            ord_unpr = str(price)

        data = await self._post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id=tr_id,
            body={
                "CANO": self._acct_no,
                "ACNT_PRDT_CD": self._acct_prod,
                "PDNO": symbol,
                "ORD_DVSN": ord_dvsn,
                "ORD_QTY": str(qty),
                "ORD_UNPR": ord_unpr,
            },
        )
        logger.info(
            "kis.buy.placed symbol=%s qty=%d price=%s mode=%s",
            symbol, qty, ord_unpr, "paper" if self._paper else "live",
        )
        return data.get("output", {})

    async def place_sell_order(
        self,
        symbol: str,
        qty: int,
        price: int | None = None,
    ) -> dict:
        """Submit a sell order.

        Args:
            symbol: KRX ticker.
            qty: Integer share count.
            price: Limit price (원). None → market order.

        Returns:
            KIS order response output dict.
        """
        tr_id = "VTTC0801U" if self._paper else "TTTC0801U"
        if price is None:
            ord_dvsn = "01"  # 시장가
            ord_unpr = "0"
        else:
            ord_dvsn = "00"  # 지정가
            ord_unpr = str(price)

        data = await self._post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id=tr_id,
            body={
                "CANO": self._acct_no,
                "ACNT_PRDT_CD": self._acct_prod,
                "PDNO": symbol,
                "ORD_DVSN": ord_dvsn,
                "ORD_QTY": str(qty),
                "ORD_UNPR": ord_unpr,
            },
        )
        logger.info(
            "kis.sell.placed symbol=%s qty=%d price=%s mode=%s",
            symbol, qty, ord_unpr, "paper" if self._paper else "live",
        )
        return data.get("output", {})

    async def cancel_order(
        self,
        order_no: str,
        krx_orgno: str,
        qty: int,
        ord_dvsn: str = "00",
    ) -> dict:
        """Cancel a pending order.

        Args:
            order_no: 주문번호 (odno from place_buy/sell_order response).
            krx_orgno: KRX 전송 조직 번호 (KRX_FWDG_ORD_ORGNO from order response).
            qty: Quantity to cancel (use 0 + QTY_ALL_ORD_YN="Y" for full cancel).
            ord_dvsn: Order type code from original order.
        """
        tr_id = "VTTC0803U" if self._paper else "TTTC0803U"
        data = await self._post(
            "/uapi/domestic-stock/v1/trading/order-rvsecncl",
            tr_id=tr_id,
            body={
                "CANO": self._acct_no,
                "ACNT_PRDT_CD": self._acct_prod,
                "KRX_FWDG_ORD_ORGNO": krx_orgno,
                "ORGN_ODNO": order_no,
                "ORD_DVSN": ord_dvsn,
                "RVSE_CNCL_DVSN_CD": "02",  # 취소
                "ORD_QTY": str(qty),
                "QTY_ALL_ORD_YN": "Y",       # 전량취소
                "ORD_UNPR": "0",
            },
        )
        logger.info("kis.cancel.placed order_no=%s", order_no)
        return data.get("output", {})

    async def fetch_unfilled_orders(self) -> list[dict]:
        """Fetch all pending (unfilled/cancellable) orders for today.

        Returns:
            List of dicts with keys: order_no, krx_orgno, symbol, qty,
            filled_qty, ord_dvsn, status.
        """
        tr_id = "VTTC8036R" if self._paper else "TTTC8036R"
        data = await self._get(
            "/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl",
            tr_id=tr_id,
            params={
                "CANO": self._acct_no,
                "ACNT_PRDT_CD": self._acct_prod,
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
                "INQR_DVSN_1": "0",
                "INQR_DVSN_2": "0",
            },
        )
        result = []
        for item in data.get("output", []):
            result.append({
                "order_no":  item.get("odno", ""),
                "krx_orgno": item.get("krx_fwdg_ord_orgno", ""),
                "symbol":    item.get("pdno", ""),
                "qty":       int(item.get("ord_qty", "0") or "0"),
                "filled_qty": int(item.get("tot_ccld_qty", "0") or "0"),
                "ord_dvsn":  item.get("ord_dvsn", "00"),
                "price":     item.get("ord_unpr", "0"),
            })
        return result

    # ------------------------------------------------------------------
    # Screener ranking APIs
    # ------------------------------------------------------------------

    async def fetch_volume_ranking(
        self,
        market: str = "J",
        top_n: int = 30,
    ) -> list[dict]:
        """Fetch top stocks by trading volume (거래량 순위).

        Args:
            market: ``'J'`` for all KRX (KOSPI + KOSDAQ). ``'NX'`` for NXT.
            top_n: Max results to return (API max = 30).

        Returns:
            List of dicts: symbol, name, price, change_pct, volume,
            trade_amount (원), market_code.
        """
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/volume-rank",
            tr_id="FHPST01710000",
            params={
                "FID_COND_MRKT_DIV_CODE": market,
                "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": "0000",
                "FID_DIV_CLS_CODE": "0",
                "FID_BLNG_CLS_CODE": "0",
                "FID_TRGT_CLS_CODE": "111111111",
                "FID_TRGT_EXLS_CLS_CODE": "0110011101",
                "FID_INPUT_PRICE_1": "",
                "FID_INPUT_PRICE_2": "",
                "FID_VOL_CNT": "",
            },
        )
        result = []
        for item in (data.get("output", []) or [])[:top_n]:
            symbol = item.get("mksc_shrn_iscd", "").strip()
            if not symbol:
                continue
            result.append({
                "symbol":       symbol,
                "name":         item.get("hts_kor_isnm", "").strip(),
                "price":        item.get("stck_prpr", "0"),
                "change_pct":   item.get("prdy_ctrt", "0"),
                "volume":       item.get("acml_vol", "0"),
                "trade_amount": item.get("acml_tr_pbmn", "0"),
                "market_code":  market,
            })
        return result

    async def fetch_fluctuation_ranking(
        self,
        market: str = "J",
        top_n: int = 30,
        min_change: float = 2.0,
        max_change: float = 30.0,
    ) -> list[dict]:
        """Fetch top stocks by price change rate (등락률 순위, 상승 only).

        Args:
            market: ``'J'`` for all KRX. ``'NX'`` for NXT.
            top_n: Max results (API max = 30).
            min_change: Minimum positive change % (e.g. 2.0 = +2%).
            max_change: Maximum change % cap (filters out limit-up noise).

        Returns:
            List of dicts: symbol, name, price, change_pct, volume, trade_amount.
        """
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/fluctuation-rank",
            tr_id="FHPST01700000",
            params={
                "FID_COND_MRKT_DIV_CODE": market,
                "FID_COND_SCR_DIV_CODE": "20170",
                "FID_INPUT_ISCD": "0000",
                "FID_RANK_SORT_CLS_CODE": "0",
                "FID_INPUT_CNT_1": "0",
                "FID_PRC_CLS_CODE": "0",
                "FID_INPUT_PRICE_1": "",
                "FID_INPUT_PRICE_2": "",
                "FID_VOL_CNT": "",
                "FID_TRGT_CLS_CODE": "111111111",
                "FID_TRGT_EXLS_CLS_CODE": "0110011101",
                "FID_DIV_CLS_CODE": "0",
                "FID_RSFL_RATE1": str(min_change),
                "FID_RSFL_RATE2": str(max_change),
            },
        )
        result = []
        for item in (data.get("output", []) or [])[:top_n]:
            symbol = item.get("mksc_shrn_iscd", "").strip()
            if not symbol:
                continue
            result.append({
                "symbol":       symbol,
                "name":         item.get("hts_kor_isnm", "").strip(),
                "price":        item.get("stck_prpr", "0"),
                "change_pct":   item.get("prdy_ctrt", "0"),
                "volume":       item.get("acml_vol", "0"),
                "trade_amount": item.get("acml_tr_pbmn", "0"),
                "market_code":  market,
            })
        return result
