from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # KIS credentials
    kis_app_key: str = Field(..., alias="KIS_APP_KEY")
    kis_app_secret: str = Field(..., alias="KIS_APP_SECRET")
    # live = 실전, paper = 모의투자
    kis_mode: str = Field("paper", alias="MODE")

    # Telegram
    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(..., alias="TELEGRAM_CHAT_ID")

    # Screener thresholds (loaded from ConfigMap env vars)
    min_trade_amount: int = Field(5_000_000_000, alias="MIN_TRADE_AMOUNT")
    min_market_cap: int = Field(300_000_000_000, alias="MIN_MARKET_CAP")
    min_change_pct: float = Field(1.0, alias="MIN_CHANGE_PCT")
    max_change_pct: float = Field(28.0, alias="MAX_CHANGE_PCT")
    min_gap_pct: float = Field(0.5, alias="MIN_GAP_PCT")
    min_volume_ratio: float = Field(2.0, alias="MIN_VOLUME_RATIO")
    top_n: int = Field(5, alias="TOP_N")
    # ALL / KOSPI / KOSDAQ
    market: str = Field("ALL", alias="MARKET")

    @property
    def kis_base_url(self) -> str:
        if self.kis_mode == "live":
            return "https://openapi.koreainvestment.com:9443"
        return "https://openapivts.koreainvestment.com:29443"

    model_config = {"env_file": ".env", "populate_by_name": True}
