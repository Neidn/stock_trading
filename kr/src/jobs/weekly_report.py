"""Weekly performance report CronJob — runs every Monday 00:00 UTC (09:00 KST).

Covers the 7-day window ending at run time. Sends via Telegram:
  - Overall PF, win rate, net PnL with week-over-week delta
  - Per-strategy breakdown (strategies with >= 2 trades)
  - Per-coin top 3 winners / losers
  - Close reason breakdown
  - Alerts: strategy PF < 1.0 or coin with 3+ consecutive losses
"""

from __future__ import annotations

import html
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_CONSECUTIVE_LOSS_ALERT = 3   # alert if a coin lost this many straight
_MIN_TRADES_FOR_STRATEGY = 2  # skip strategy breakdown for <2 trades


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def _week_positions(conn: sqlite3.Connection, end: datetime, days: int = 7) -> list[sqlite3.Row]:
    start = (end - timedelta(days=days)).isoformat()
    end_s = end.isoformat()
    return _rows(
        conn,
        """SELECT symbol, side, realized_pnl, close_reason, strategy_name, closed_at,
                  slippage_bps
           FROM positions
           WHERE status='closed'
             AND realized_pnl IS NOT NULL
             AND realized_pnl != '0'
             AND closed_at >= ? AND closed_at < ?
           ORDER BY closed_at""",
        (start, end_s),
    )


def _pf_winrate(records: list[tuple[float, str]]) -> tuple[float, float, int]:
    """Return (profit_factor, win_rate, trade_count) from (pnl, reason) pairs."""
    if not records:
        return 0.0, 0.0, 0
    wins   = [p for p, _ in records if p > 0]
    losses = [p for p, _ in records if p < 0]
    gross_p = sum(wins)
    gross_l = abs(sum(losses))
    pf = gross_p / gross_l if gross_l > 0 else float("inf")
    wr = len(wins) / len(records)
    return pf, wr, len(records)


def _format_pf(pf: float) -> str:
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def _pf_emoji(pf: float) -> str:
    if pf == float("inf") or pf >= 1.5:
        return "🟢"
    if pf >= 1.0:
        return "🟡"
    return "🔴"


class WeeklyReportJob:
    """Send weekly performance report via Telegram.

    Args:
        conn: SQLite connection.
        telegram_bot: Must expose ``send_info(str)``.
    """

    def __init__(self, conn: sqlite3.Connection, telegram_bot=None) -> None:
        self._conn = conn
        self._telegram = telegram_bot

    def run(self) -> str:
        """Build and send weekly report; return the report string."""
        now = datetime.now(timezone.utc)
        report = self._build_report(now)
        logger.info("WeeklyReportJob: sending report ending %s", now.date().isoformat())
        self._send(report)
        return report

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(self, end: datetime) -> str:
        start = end - timedelta(days=7)
        this_week = _week_positions(self._conn, end, days=7)
        prev_week = _week_positions(self._conn, start, days=7)

        lines = [
            f"📈 <b>주간 성과 리포트</b>",
            f"<b>{start.date()} ~ {(end - timedelta(seconds=1)).date()}</b>\n",
        ]

        # ----------------------------------------------------------------
        # 1. Overall summary
        # ----------------------------------------------------------------
        this_records = [(float(r["realized_pnl"]), r["close_reason"] or "") for r in this_week]
        prev_records = [(float(r["realized_pnl"]), r["close_reason"] or "") for r in prev_week]

        pf, wr, total = _pf_winrate(this_records)
        prev_pf, _, prev_total = _pf_winrate(prev_records)

        net_pnl = sum(p for p, _ in this_records)
        wins    = sum(1 for p, _ in this_records if p > 0)
        losses  = sum(1 for p, _ in this_records if p < 0)

        if total == 0:
            lines.append("이번 주 종료 포지션 없음")
            self._send("\n".join(lines))
            return "\n".join(lines)

        pf_delta = ""
        if prev_total > 0:
            delta = pf - prev_pf if pf != float("inf") and prev_pf != float("inf") else 0.0
            arrow = "▲" if delta >= 0 else "▼"
            pf_delta = f"  ({arrow}{abs(delta):.2f} vs 전주 {_format_pf(prev_pf)})"

        lines += [
            "━━━━━ 전체 요약 ━━━━━",
            f"거래: {total}건  |  승률: {wr:.1%} ({wins}승/{losses}패)",
            f"순손익: {net_pnl:+.2f} USDT",
            f"Profit Factor: {_pf_emoji(pf)} {_format_pf(pf)}{pf_delta}",
            "",
        ]

        # ----------------------------------------------------------------
        # 2. Per-strategy breakdown
        # ----------------------------------------------------------------
        by_strategy: dict[str, list[tuple[float, str]]] = {}
        by_strategy_slip: dict[str, list[float]] = {}
        for r in this_week:
            name = r["strategy_name"] or "unknown"
            by_strategy.setdefault(name, []).append(
                (float(r["realized_pnl"]), r["close_reason"] or "")
            )
            slip = r["slippage_bps"] if "slippage_bps" in r.keys() else None
            if slip is not None:
                by_strategy_slip.setdefault(name, []).append(float(slip))

        known = {k: v for k, v in by_strategy.items() if len(v) >= _MIN_TRADES_FOR_STRATEGY}
        if known:
            lines.append("━━━━━ 전략별 ━━━━━")
            for name, recs in sorted(known.items(), key=lambda x: -len(x[1])):
                s_pf, s_wr, s_total = _pf_winrate(recs)
                s_net = sum(p for p, _ in recs)
                slip_vals = by_strategy_slip.get(name, [])
                slip_str = f" | 슬리피지 {sum(slip_vals)/len(slip_vals):+.1f}bps" if slip_vals else ""
                lines.append(
                    f"{_pf_emoji(s_pf)} {html.escape(name)}: {s_total}건 | "
                    f"승률 {s_wr:.0%} | PF {_format_pf(s_pf)} | {s_net:+.2f} USDT{slip_str}"
                )
            lines.append("")

        # ----------------------------------------------------------------
        # 3. Per-coin breakdown (top 3 + / bottom 3 -)
        # ----------------------------------------------------------------
        by_coin: dict[str, list[float]] = {}
        for r in this_week:
            by_coin.setdefault(r["symbol"], []).append(float(r["realized_pnl"]))

        coin_summary = sorted(
            [(sym, sum(pnls), sum(1 for p in pnls if p > 0), sum(1 for p in pnls if p < 0))
             for sym, pnls in by_coin.items()],
            key=lambda x: -x[1],
        )

        if coin_summary:
            lines.append("━━━━━ 코인별 ━━━━━")
            top3 = coin_summary[:3]
            bot3 = [c for c in coin_summary[-3:] if c[1] < 0]
            for sym, net, w, l in top3:
                lines.append(f"▲ {html.escape(sym)}: {net:+.2f} USDT ({w}승/{l}패)")
            for sym, net, w, l in bot3:
                if (sym, net, w, l) not in top3:
                    lines.append(f"▼ {html.escape(sym)}: {net:+.2f} USDT ({w}승/{l}패)")
            lines.append("")

        # ----------------------------------------------------------------
        # 4. Close reason breakdown
        # ----------------------------------------------------------------
        reason_counts: dict[str, int] = {}
        for _, reason in this_records:
            key = reason or "unknown"
            reason_counts[key] = reason_counts.get(key, 0) + 1

        if reason_counts:
            lines.append("━━━━━ 청산 방식 ━━━━━")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {html.escape(reason)}: {count}건")
            lines.append("")

        # ----------------------------------------------------------------
        # 5. Alerts
        # ----------------------------------------------------------------
        alerts = self._compute_alerts(pf, by_strategy, by_coin, this_week)
        if alerts:
            lines.append("━━━━━ ⚠️ 경보 ━━━━━")
            lines.extend(alerts)

        return "\n".join(lines)

    def _compute_alerts(
        self,
        overall_pf: float,
        by_strategy: dict[str, list[tuple[float, str]]],
        by_coin: dict[str, list[float]],
        rows: list[sqlite3.Row],
    ) -> list[str]:
        alerts = []

        # Strategy PF < 1.0 with enough trades
        for name, recs in by_strategy.items():
            if len(recs) >= _MIN_TRADES_FOR_STRATEGY:
                s_pf, _, _ = _pf_winrate(recs)
                if s_pf != float("inf") and s_pf < 1.0:
                    alerts.append(f"🔴 {html.escape(name)} PF={s_pf:.2f} &lt; 1.0 — 파라미터 재검토 필요")

        # Coin with N consecutive losses (check chronological order)
        coin_last: dict[str, list[float]] = {}
        for r in rows:
            coin_last.setdefault(r["symbol"], []).append(float(r["realized_pnl"]))
        for sym, pnls in coin_last.items():
            tail = 0
            for p in reversed(pnls):
                if p < 0:
                    tail += 1
                else:
                    break
            if tail >= _CONSECUTIVE_LOSS_ALERT:
                alerts.append(f"⚠️ {html.escape(sym)} {tail}연패 — 점검 필요")

        return alerts

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

    from src.db.models import init_db
    init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    from src.monitoring.telegram_bot import get_telegram_bot
    telegram = get_telegram_bot()

    WeeklyReportJob(conn=conn, telegram_bot=telegram).run()


if __name__ == "__main__":
    main()
