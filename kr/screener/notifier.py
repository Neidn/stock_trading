from __future__ import annotations

import structlog
import requests

from screener.config import Settings
from screener.screener import ScreenedStock

log = structlog.get_logger()

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

_RANK_EMOJI = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
_RUN_MODE_LABEL = {
    "pre_market": "08:00 사전 스크리닝",
    "final": "08:50 최종 확정",
}


class TelegramNotifier:
    def __init__(self, settings: Settings) -> None:
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id

    def send(self, stocks: list[ScreenedStock], run_mode: str = "pre_market") -> None:
        text = _format_message(stocks, run_mode)
        self._post(text)

    def _post(self, text: str) -> None:
        url = _TELEGRAM_API.format(token=self._token)
        try:
            resp = requests.post(
                url,
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            resp.raise_for_status()
            log.info("notifier.sent")
        except requests.RequestException:
            log.exception("notifier.send.failed")
            raise


def _format_message(stocks: list[ScreenedStock], run_mode: str) -> str:
    label = _RUN_MODE_LABEL.get(run_mode, run_mode)
    lines = [f"🔔 <b>[스캘핑 후보]</b> {label}\n"]

    for idx, s in enumerate(stocks):
        d = s.data
        emoji = _RANK_EMOJI[idx] if idx < len(_RANK_EMOJI) else f"{idx + 1}."
        gap_sign = "+" if d.gap_pct >= 0 else ""
        change_sign = "+" if d.change_pct >= 0 else ""
        trade_eok = d.trade_amount / 1_0000_0000

        lines.append(
            f"{emoji} <b>{d.name}</b> ({d.ticker}) ★ {s.score}점\n"
            f"   현재가: {d.current_price:,.0f} | 갭: {gap_sign}{d.gap_pct:.1f}%\n"
            f"   전일比 등락: {change_sign}{d.change_pct:.1f}% | 거래량: 전일比 {d.volume_ratio:.1f}배\n"
            f"   거래대금: {trade_eok:.0f}억 | 시장: {d.market}\n"
        )

    lines.append("⚠️ 본 정보는 참고용이며 투자 판단은 본인 책임입니다.")
    return "\n".join(lines)
