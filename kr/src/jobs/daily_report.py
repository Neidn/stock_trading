"""Daily performance report CronJob — runs at 09:00 KST (00:00 UTC).

Queries ``daily_performance`` for yesterday's aggregates, appends per-position
liquidation-distance lines, and sends the report via Telegram (plan v5 §14.2).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date, timedelta
from typing import TYPE_CHECKING

from src.risk.liquidation_guard import distance_to_liquidation_pct

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DailyReportJob:
    """Send yesterday's performance report via Telegram.

    Args:
        conn: SQLite connection.
        telegram_bot: Must expose ``send_info(str)``.
    """

    def __init__(self, conn: sqlite3.Connection, telegram_bot=None) -> None:
        self._conn = conn
        self._telegram = telegram_bot

    def run(self) -> str:
        """Build and send report; return the report string."""
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        report = self._build_report(yesterday)
        logger.info("DailyReportJob: sending report for %s", yesterday)
        self._send(report)
        return report

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(self, perf_date: str) -> str:
        perf = self._conn.execute(
            """SELECT total_trades, winning_trades, losing_trades, liquidated_trades,
                      net_pnl, gross_profit, gross_loss, total_fees, max_drawdown
               FROM daily_performance WHERE perf_date=?""",
            (perf_date,),
        ).fetchone()

        open_positions = self._conn.execute(
            """SELECT symbol, side, entry_price, quantity, liquidation_price, unrealized_pnl
               FROM positions WHERE status='open'"""
        ).fetchall()

        lines = [f"📊 <b>일일 성과 리포트 ({perf_date})</b>\n"]

        if perf is None:
            lines.append("거래 없음")
        else:
            total = perf["total_trades"] or 0
            wins = perf["winning_trades"] or 0
            win_rate = wins / total * 100 if total > 0 else 0.0
            lines += [
                f"총 거래: {total}건",
                f"승률: {win_rate:.1f}%  (승 {wins} / 패 {perf['losing_trades'] or 0})",
                f"청산 발생: {perf['liquidated_trades'] or 0}건",
                f"순손익: {float(perf['net_pnl'] or 0):+.2f} USDT",
                f"총수익: +{float(perf['gross_profit'] or 0):.2f} USDT",
                f"총손실: -{float(perf['gross_loss'] or 0):.2f} USDT",
                f"수수료: {float(perf['total_fees'] or 0):.2f} USDT",
                f"최대낙폭: {float(perf['max_drawdown'] or 0):.2f}%",
            ]

        lines.append(f"\n현재 오픈 포지션: {len(open_positions)}개")
        for pos in open_positions:
            dist = self._liq_distance(pos)
            upnl = float(pos["unrealized_pnl"] or 0)
            sign = "+" if upnl >= 0 else ""
            dist_str = f"{dist:.1f}%" if dist is not None else "N/A"
            lines.append(
                f"  • {pos['symbol']} {pos['side'].upper()} "
                f"| uPnL={sign}{upnl:.2f} | 청산까지 {dist_str}"
            )

        return "\n".join(lines)

    @staticmethod
    def _liq_distance(pos) -> float | None:
        try:
            entry = float(pos["entry_price"])
            liq = float(pos["liquidation_price"])
            side = pos["side"]
            # Use entry price as proxy for current price when no mark price available
            return distance_to_liquidation_pct(entry, liq, side)
        except Exception:  # noqa: BLE001
            return None

    def _send(self, report: str) -> None:
        if self._telegram is None:
            logger.debug("No telegram bot; report not sent")
            return
        try:
            self._telegram.send_info(report)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram send failed: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    db_path = os.environ.get("SQLITE_DB_PATH", "/data/trading.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    from src.monitoring.telegram_bot import get_telegram_bot
    telegram = get_telegram_bot()

    DailyReportJob(conn=conn, telegram_bot=telegram).run()


if __name__ == "__main__":
    main()
