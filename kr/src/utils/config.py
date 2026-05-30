"""Application configuration — loaded from environment variables.

All values sourced exclusively from environment variables (K8s ConfigMap/Secret
or local .env file).  No hardcoded credentials belong here.

Usage::

    from src.utils.config import cfg
    print(cfg.trading_mode)
    client = KISRestClient()   # reads env vars directly
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

logger = logging.getLogger(__name__)

LIVE_MAX_RISK_PER_TRADE = 0.005
LIVE_MAX_POSITIONS = 5
LIVE_MAX_DAILY_LOSS_LIMIT = 0.03


class TradingMode(str, Enum):
    PAPER = "paper"   # 모의투자 (KIS sandbox)
    LIVE = "live"     # real money


@dataclass(frozen=True)
class Config:
    trading_mode: TradingMode
    active_strategy: str
    strategy_params: dict
    risk_per_trade: float
    max_positions: int
    daily_loss_limit: float
    sqlite_db_path: str
    log_level: str

    # Market settings
    market: str          # KOSPI | KOSDAQ | BOTH
    allow_overnight: bool
    force_close_time: str  # "HH:MM" KST

    # KIS credentials (never logged)
    kis_app_key: str = field(repr=False)
    kis_app_secret: str = field(repr=False)
    kis_account_no: str = field(repr=False)

    # Telegram
    telegram_bot_token: str = field(repr=False)
    telegram_chat_id: str = field(repr=False)


def load_config() -> Config:
    raw_mode = os.getenv("TRADING_MODE", "paper").strip().lower()
    try:
        trading_mode = TradingMode(raw_mode)
    except ValueError:
        raise ValueError(
            f"TRADING_MODE='{raw_mode}' is invalid. Must be 'paper' or 'live'."
        )

    if trading_mode is TradingMode.LIVE:
        _confirm_live_mode()

    raw_params = os.getenv("STRATEGY_PARAMS", "{}").strip()
    try:
        strategy_params = json.loads(raw_params)
    except json.JSONDecodeError as exc:
        logger.warning("STRATEGY_PARAMS JSON parse failed, using {}: %s", exc)
        strategy_params = {}

    active_strategy = os.getenv("ACTIVE_STRATEGY", "rsi_macd").strip()
    risk_per_trade = _float_env("RISK_PER_TRADE", "0.01")
    max_positions = _int_env("MAX_POSITIONS", "5")
    daily_loss_limit = _float_env("DAILY_LOSS_LIMIT", "0.05")
    sqlite_db_path = os.getenv("SQLITE_DB_PATH", "/data/trading.db")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    market = os.getenv("MARKET", "BOTH").upper()
    allow_overnight = os.getenv("ALLOW_OVERNIGHT", "false").lower() == "true"
    force_close_time = os.getenv("FORCE_CLOSE_TIME", "15:20")
    kis_app_key = os.getenv("KIS_APP_KEY", "")
    kis_app_secret = os.getenv("KIS_APP_SECRET", "")
    kis_account_no = os.getenv("KIS_ACCOUNT_NO", "")
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    _validate_config(
        trading_mode=trading_mode,
        risk_per_trade=risk_per_trade,
        max_positions=max_positions,
        daily_loss_limit=daily_loss_limit,
        kis_app_key=kis_app_key,
        kis_app_secret=kis_app_secret,
    )

    return Config(
        trading_mode=trading_mode,
        active_strategy=active_strategy,
        strategy_params=strategy_params,
        risk_per_trade=risk_per_trade,
        max_positions=max_positions,
        daily_loss_limit=daily_loss_limit,
        sqlite_db_path=sqlite_db_path,
        log_level=log_level,
        market=market,
        allow_overnight=allow_overnight,
        force_close_time=force_close_time,
        kis_app_key=kis_app_key,
        kis_app_secret=kis_app_secret,
        kis_account_no=kis_account_no,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
    )


def _int_env(name: str, default: str) -> int:
    raw = os.getenv(name, default).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


def _float_env(name: str, default: str) -> float:
    raw = os.getenv(name, default).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}.") from exc


def _validate_config(
    *,
    trading_mode: TradingMode,
    risk_per_trade: float,
    max_positions: int,
    daily_loss_limit: float,
    kis_app_key: str,
    kis_app_secret: str,
) -> None:
    if max_positions < 1:
        raise ValueError("MAX_POSITIONS must be >= 1.")
    if not 0 < risk_per_trade < 1:
        raise ValueError("RISK_PER_TRADE must be between 0 and 1.")
    if not 0 < daily_loss_limit < 1:
        raise ValueError("DAILY_LOSS_LIMIT must be between 0 and 1.")

    if trading_mode is not TradingMode.LIVE:
        return

    violations: list[str] = []
    if risk_per_trade > LIVE_MAX_RISK_PER_TRADE:
        violations.append(f"RISK_PER_TRADE={risk_per_trade} > {LIVE_MAX_RISK_PER_TRADE}")
    if max_positions > LIVE_MAX_POSITIONS:
        violations.append(f"MAX_POSITIONS={max_positions} > {LIVE_MAX_POSITIONS}")
    if daily_loss_limit > LIVE_MAX_DAILY_LOSS_LIMIT:
        violations.append(f"DAILY_LOSS_LIMIT={daily_loss_limit} > {LIVE_MAX_DAILY_LOSS_LIMIT}")
    if not kis_app_key or not kis_app_secret:
        violations.append("KIS_APP_KEY and KIS_APP_SECRET are required for live mode")

    if violations:
        raise ValueError(
            "Unsafe live configuration. Violations: " + "; ".join(violations)
        )


_live_confirmed: bool = False


def _confirm_live_mode() -> None:
    """Block startup until operator types LIVE — prevents accidental live deploy."""
    global _live_confirmed
    if os.getenv("LIVE_CONFIRM") == "LIVE":
        if not _live_confirmed:
            logger.warning(
                "LIVE_CONFIRM=LIVE — skipping TTY gate. Running in live trading mode."
            )
            _live_confirmed = True
        return

    if not sys.stdin.isatty():
        logger.critical(
            "TRADING_MODE=live detected but stdin is not a TTY. "
            "Set LIVE_CONFIRM=LIVE in ConfigMap to allow non-interactive live startup."
        )
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ⚠️  LIVE TRADING MODE — 실제 자금 거래")
    print("  To confirm, type exactly:  LIVE")
    print("=" * 60)
    response = input("확인 입력: ").strip()
    if response != "LIVE":
        print("입력 불일치. 종료합니다.")
        sys.exit(1)
    print("LIVE 모드 확인 완료.\n")


cfg: Config = load_config()
