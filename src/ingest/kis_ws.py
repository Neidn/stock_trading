"""KIS (한국투자증권) real-time WebSocket client.

Subscribes to 실시간체결(H0STCNT0) stream per ticker.  Used by the
safety monitor for manual stop-loss price monitoring.

Architecture::

    KISWSManager
    ├── subscribe(ticker) → spawns connect_with_retry task per ticker
    ├── connect_with_retry(url, ticker) → exponential back-off loop
    │   └── _handle_messages(ws, ticker) → routes each message
    │       └── 체결 → _on_trade → calls price_callback(ticker, price)
    ├── _watchdog() → restarts dead connections after silence timeout
    └── unsubscribe(ticker) → cancels task + closes ws

Usage::

    def on_price(ticker: str, price: int) -> None:
        print(f"{ticker}: {price:,}원")

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

_PAPER_WS = "ws://ops.koreainvestment.com:31000"
_LIVE_WS = "ws://ops.koreainvestment.com:21000"

_WATCHDOG_INTERVAL = 5     # seconds between watchdog checks
_SILENCE_TIMEOUT = 60      # seconds without message → reconnect
_MAX_BACKOFF = 300          # max reconnect delay (seconds)

_APPROVAL_PATH = "/oauth2/Approval"
_PAPER_REST = "https://openapivts.koreainvestment.com:29443"
_LIVE_REST = "https://openapi.koreainvestment.com:9443"

# Real-time 체결 stream identifier
_TR_ID_TRADE = "H0STCNT0"

# Field indices in the pipe-delimited KIS trade response
# Format: {enc}|{tr_id}|{count}|{data1}^{data2}^...
# data fields for H0STCNT0 (체결):
_IDX_TICKER = 0
_IDX_PRICE = 2   # 현재가


class KISWSManager:
    """Manage KIS real-time WebSocket subscriptions for price monitoring.

    Args:
        price_callback: Called with (ticker: str, price: int) on each trade.
        paper: If True, use 모의투자 endpoints. Default reads TRADING_MODE env.
        telegram_bot: Optional; used for disconnect/reconnect alerts.
    """

    def __init__(
        self,
        price_callback: Callable[[str, int], None] | None = None,
        paper: bool | None = None,
        telegram_bot=None,
    ) -> None:
        if paper is None:
            paper = os.getenv("TRADING_MODE", "paper").strip().lower() != "live"
        self._paper = paper
        self._ws_base = _PAPER_WS if paper else _LIVE_WS
        self._rest_base = _PAPER_REST if paper else _LIVE_REST
        self._app_key = os.getenv("KIS_APP_KEY", "")
        self._app_secret = os.getenv("KIS_APP_SECRET", "")

        self._price_callback = price_callback
        self._telegram = telegram_bot

        self._approval_key: str | None = None
        self._tasks: dict[str, asyncio.Task] = {}
        self._ws_connections: dict[str, aiohttp.ClientWebSocketResponse] = {}
        self._last_msg_time: dict[str, float] = {}
        self._session: aiohttp.ClientSession | None = None
        self._watchdog_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )

    async def close(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        if self._watchdog_task:
            self._watchdog_task.cancel()
        all_tasks = list(self._tasks.values())
        if self._watchdog_task:
            all_tasks.append(self._watchdog_task)
        await asyncio.gather(*all_tasks, return_exceptions=True)
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "KISWSManager":
        await self._ensure_session()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Approval key (required before WS subscription)
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
        """Start streaming 체결 data for *ticker*."""
        await self._ensure_session()
        approval_key = await self._get_approval_key()

        logger.info("KIS WS subscribing %s", ticker)
        self._last_msg_time[ticker] = time.monotonic()

        task = asyncio.create_task(
            self.connect_with_retry(ticker, approval_key),
            name=f"kisws_{ticker}",
        )
        self._tasks[ticker] = task

        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(
                self._watchdog(), name="kisws_watchdog"
            )

    async def unsubscribe(self, ticker: str) -> None:
        task = self._tasks.pop(ticker, None)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        ws = self._ws_connections.pop(ticker, None)
        if ws and not ws.closed:
            await ws.close()
        self._last_msg_time.pop(ticker, None)
        logger.info("KIS WS unsubscribed %s", ticker)

    def get_last_price(self, ticker: str) -> int | None:
        """Return last known price for *ticker*, or None if not yet received."""
        return self._last_prices.get(ticker) if hasattr(self, "_last_prices") else None

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def connect_with_retry(self, ticker: str, approval_key: str) -> None:
        delay = 1
        attempt = 0
        alerted = False   # True once a persistent-disconnect Telegram was sent
        if not hasattr(self, "_last_prices"):
            self._last_prices: dict[str, int] = {}

        while True:
            try:
                attempt += 1
                logger.info("KIS WS %s attempt #%d", ticker, attempt)
                assert self._session
                async with self._session.ws_connect(
                    self._ws_base,
                    heartbeat=30,
                    receive_timeout=_SILENCE_TIMEOUT + 5,
                ) as ws:
                    connected_at = time.monotonic()
                    self._ws_connections[ticker] = ws
                    logger.info("KIS WS %s connected", ticker)

                    if alerted and self._telegram:
                        self._telegram.send_info(f"✅ KIS WS {ticker} 재연결 성공")
                        alerted = False

                    # Send subscription message
                    await ws.send_str(json.dumps({
                        "header": {
                            "approval_key": approval_key,
                            "custtype": "P",
                            "tr_type": "1",
                            "content-type": "utf-8",
                        },
                        "body": {
                            "input": {
                                "tr_id": _TR_ID_TRADE,
                                "tr_key": ticker,
                            }
                        },
                    }))

                    await self._handle_messages(ws, ticker)

                # Only reset backoff when connection was stable ≥10s
                if time.monotonic() - connected_at >= 10:
                    delay = 1

            except asyncio.CancelledError:
                return
            except Exception as exc:
                safe_exc = str(exc).replace("<", "(").replace(">", ")")
                logger.warning(
                    "KIS WS %s disconnected (%s), reconnect in %ds",
                    ticker, safe_exc, delay,
                )
                # Only alert after backoff reaches 60s (persistent failure)
                if delay >= 60 and self._telegram and not alerted:
                    self._telegram.send_warning(
                        f"⚠️ KIS WS {ticker} 지속 끊김 — {safe_exc}"
                    )
                    alerted = True

            self._ws_connections.pop(ticker, None)
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            delay = min(delay * 2, _MAX_BACKOFF)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_messages(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        ticker: str,
    ) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self._last_msg_time[ticker] = time.monotonic()
                try:
                    self._dispatch(msg.data, ticker)
                except Exception as exc:
                    logger.warning("KIS WS parse error %s: %s", ticker, exc)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                logger.warning("KIS WS %s closed/error: %s", ticker, msg.data)
                return

    def _dispatch(self, raw: str, ticker: str) -> None:
        """Route raw KIS message to appropriate handler."""
        # KIS sends JSON for system messages (ping/subscribe confirm)
        if raw.startswith("{"):
            data = json.loads(raw)
            if data.get("header", {}).get("tr_id") == _TR_ID_TRADE:
                logger.debug("KIS WS %s subscription confirmed", ticker)
            return

        # Real-time data: "0|H0STCNT0|001|field1^field2^..."
        parts = raw.split("|")
        if len(parts) < 4:
            return
        tr_id = parts[1]
        if tr_id == _TR_ID_TRADE:
            self._on_trade(parts[3], ticker)

    def _on_trade(self, data_str: str, ticker: str) -> None:
        """Parse 체결 data and invoke price callback."""
        fields = data_str.split("^")
        if len(fields) <= max(_IDX_TICKER, _IDX_PRICE):
            return

        recv_ticker = fields[_IDX_TICKER].strip()
        try:
            price = int(fields[_IDX_PRICE])
        except (ValueError, IndexError):
            return

        if not hasattr(self, "_last_prices"):
            self._last_prices = {}
        self._last_prices[recv_ticker] = price

        if self._price_callback:
            try:
                self._price_callback(recv_ticker, price)
            except Exception as exc:
                logger.warning("price_callback error %s: %s", recv_ticker, exc)

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    async def _watchdog(self) -> None:
        while True:
            try:
                await asyncio.sleep(_WATCHDOG_INTERVAL)
            except asyncio.CancelledError:
                return
            now = time.monotonic()
            for ticker, last in list(self._last_msg_time.items()):
                if now - last > _SILENCE_TIMEOUT:
                    silence = now - last
                    logger.warning(
                        "KIS WS %s: no message for %.0fs — forcing reconnect",
                        ticker, silence,
                    )
                    ws = self._ws_connections.pop(ticker, None)
                    if ws and not ws.closed:
                        await ws.close()
                    self._last_msg_time[ticker] = now
