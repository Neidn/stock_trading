"""KIS (한국투자증권) real-time WebSocket client.

Single multiplexed connection — all KRX tickers share one socket.
KIS limits WS connections per app key; opening one per ticker causes
immediate connection resets.

Architecture::

    KISWSManager
    ├── subscribe(ticker)   → adds to _subscribed; sends SUB msg if WS open
    ├── unsubscribe(ticker) → sends UNSUB msg; removes from _subscribed
    ├── _conn_loop()        → ONE WS connection, exponential back-off on failure
    │   └── _handle_messages(ws) → routes each trade msg by ticker from data
    ├── _watchdog()         → reconnects if connection silent > _SILENCE_TIMEOUT
    └── get_last_price(ticker) → last known price or None

Usage::

    async with KISWSManager(price_callback=on_price) as ws:
        await ws.subscribe("005930")
        await ws.subscribe("000660")
        await asyncio.sleep(3600)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Callable

import aiohttp

from src.monitoring.logger import get_logger

logger = get_logger("kis_ws")

_PAPER_WS   = "ws://ops.koreainvestment.com:31000"
_LIVE_WS    = "ws://ops.koreainvestment.com:21000"
_PAPER_REST = "https://openapivts.koreainvestment.com:29443"
_LIVE_REST  = "https://openapi.koreainvestment.com:9443"

_APPROVAL_PATH   = "/oauth2/Approval"
_WATCHDOG_INTERVAL = 5    # seconds between watchdog checks
_SILENCE_TIMEOUT   = 300  # seconds without any message → reconnect
_MAX_BACKOFF       = 300  # max reconnect delay (seconds)

_TR_ID_TRADE = "H0STCNT0"

# Field indices in pipe-delimited KIS trade response
# Format: {enc}|{tr_id}|{count}|{field0}^{field1}^{field2}^...
_IDX_TICKER = 0
_IDX_PRICE  = 2   # 현재가


class KISWSManager:
    """Single multiplexed KIS WebSocket for real-time KRX price monitoring.

    Args:
        price_callback: Called with (ticker: str, price: int) on each trade.
        paper: Use 모의투자 endpoints if True. Defaults to TRADING_MODE env.
        telegram_bot: Optional; used for persistent-disconnect alerts.
    """

    def __init__(
        self,
        price_callback: Callable[[str, int], None] | None = None,
        paper: bool | None = None,
        telegram_bot=None,
    ) -> None:
        if paper is None:
            paper = os.getenv("TRADING_MODE", "paper").strip().lower() != "live"
        self._ws_base   = _PAPER_WS   if paper else _LIVE_WS
        self._rest_base = _PAPER_REST  if paper else _LIVE_REST
        self._app_key    = os.getenv("KIS_APP_KEY", "")
        self._app_secret = os.getenv("KIS_APP_SECRET", "")

        self._price_callback = price_callback
        self._telegram       = telegram_bot

        self._approval_key: str | None = None
        self._subscribed: set[str] = set()
        self._last_prices: dict[str, int] = {}
        self._last_msg_time: float = time.monotonic()

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._conn_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )

    async def close(self) -> None:
        tasks = [t for t in (self._conn_task, self._watchdog_task) if t]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "KISWSManager":
        await self._ensure_session()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Approval key
    # ------------------------------------------------------------------

    async def _get_approval_key(self) -> str:
        if self._approval_key:
            return self._approval_key
        await self._ensure_session()
        assert self._session
        async with self._session.post(
            f"{self._rest_base}{_APPROVAL_PATH}",
            json={
                "grant_type": "client_credentials",
                "appkey": self._app_key,
                "secretkey": self._app_secret,
            },
        ) as resp:
            data = await resp.json(content_type=None)
        self._approval_key = data.get("approval_key", "")
        logger.info("KIS WS approval key obtained")
        return self._approval_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def subscribe(self, ticker: str) -> None:
        """Register ticker and send subscription msg if WS is already open."""
        self._subscribed.add(ticker)
        logger.info("KIS WS subscribing %s", ticker)

        if self._ws is not None and not self._ws.closed:
            await self._send_sub(self._ws, ticker, subscribe=True)

        # Start connection loop if not running
        await self._ensure_session()
        if self._conn_task is None or self._conn_task.done():
            self._conn_task = asyncio.create_task(self._conn_loop(), name="kisws_conn")
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog(), name="kisws_watchdog")

    async def unsubscribe(self, ticker: str) -> None:
        self._subscribed.discard(ticker)
        if self._ws is not None and not self._ws.closed:
            await self._send_sub(self._ws, ticker, subscribe=False)
        logger.info("KIS WS unsubscribed %s", ticker)

    def get_last_price(self, ticker: str) -> int | None:
        return self._last_prices.get(ticker)

    # ------------------------------------------------------------------
    # Single connection loop
    # ------------------------------------------------------------------

    async def _conn_loop(self) -> None:
        delay   = 1
        attempt = 0
        alerted = False

        while True:
            try:
                approval_key = await self._get_approval_key()
                attempt += 1
                logger.info("KIS WS connect attempt #%d tickers=%s", attempt, sorted(self._subscribed))
                assert self._session

                async with self._session.ws_connect(
                    self._ws_base,
                    heartbeat=30,
                    receive_timeout=_SILENCE_TIMEOUT + 30,
                ) as ws:
                    connected_at = time.monotonic()
                    self._ws = ws
                    self._last_msg_time = time.monotonic()
                    logger.info("KIS WS connected")

                    if alerted and self._telegram:
                        self._telegram.send_info("✅ KIS WS 재연결 성공")
                        alerted = False

                    # Subscribe all registered tickers on this connection
                    for ticker in list(self._subscribed):
                        await self._send_sub(ws, ticker, subscribe=True)

                    await self._handle_messages(ws)

                if time.monotonic() - connected_at >= 10:
                    delay = 1

            except asyncio.CancelledError:
                return
            except Exception as exc:
                safe_exc = str(exc).replace("<", "(").replace(">", ")")
                logger.warning("KIS WS disconnected (%s), reconnect in %ds", safe_exc, delay)
                if delay >= 60 and self._telegram and not alerted:
                    self._telegram.send_warning(f"⚠️ KIS WS 지속 끊김 — {safe_exc}")
                    alerted = True

            self._ws = None
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            delay = min(delay * 2, _MAX_BACKOFF)

    async def _send_sub(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        ticker: str,
        subscribe: bool,
    ) -> None:
        approval_key = await self._get_approval_key()
        await ws.send_str(json.dumps({
            "header": {
                "approval_key": approval_key,
                "custtype":     "P",
                "tr_type":      "1" if subscribe else "2",
                "content-type": "utf-8",
            },
            "body": {
                "input": {
                    "tr_id":  _TR_ID_TRADE,
                    "tr_key": ticker,
                }
            },
        }))

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_messages(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self._last_msg_time = time.monotonic()
                try:
                    self._dispatch(msg.data)
                except Exception as exc:
                    logger.warning("KIS WS parse error: %s", exc)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                logger.warning("KIS WS closed/error: %s", msg.data)
                return

    def _dispatch(self, raw: str) -> None:
        if raw.startswith("{"):
            data = json.loads(raw)
            tr = data.get("header", {}).get("tr_id", "")
            if tr == _TR_ID_TRADE:
                logger.debug("KIS WS subscription confirmed")
            return

        parts = raw.split("|")
        if len(parts) < 4:
            return
        if parts[1] == _TR_ID_TRADE:
            self._on_trade(parts[3])

    def _on_trade(self, data_str: str) -> None:
        fields = data_str.split("^")
        if len(fields) <= max(_IDX_TICKER, _IDX_PRICE):
            return
        ticker = fields[_IDX_TICKER].strip()
        try:
            price = int(fields[_IDX_PRICE])
        except (ValueError, IndexError):
            return

        self._last_prices[ticker] = price

        if self._price_callback:
            try:
                self._price_callback(ticker, price)
            except Exception as exc:
                logger.warning("price_callback error %s: %s", ticker, exc)

    # ------------------------------------------------------------------
    # Watchdog — reconnect if connection goes silent
    # ------------------------------------------------------------------

    async def _watchdog(self) -> None:
        while True:
            try:
                await asyncio.sleep(_WATCHDOG_INTERVAL)
            except asyncio.CancelledError:
                return
            silence = time.monotonic() - self._last_msg_time
            if silence > _SILENCE_TIMEOUT:
                logger.warning("KIS WS: no message for %.0fs — forcing reconnect", silence)
                if self._ws and not self._ws.closed:
                    await self._ws.close()
                self._ws = None
                self._last_msg_time = time.monotonic()
