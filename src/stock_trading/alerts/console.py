from __future__ import annotations

from stock_trading.models import ScreenerResult, Signal


def format_screener_results(results: list[ScreenerResult]) -> str:
    lines = ["Screener Results"]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        reason = "; ".join(result.reasons)
        lines.append(f"{status} {result.symbol} score={result.score:.1f} {reason}")
    return "\n".join(lines)


def format_signals(signals: list[Signal]) -> str:
    lines = ["Signals"]
    for signal in signals:
        expiry = signal.expiry.isoformat() if signal.expiry else "none"
        if signal.risk:
            risk = (
                f"entry={signal.risk.entry:.2f} stop={signal.risk.stop:.2f} "
                f"target={signal.risk.target:.2f} shares={signal.risk.shares} "
                f"risk=${signal.risk.capital_at_risk:.2f} notional=${signal.risk.notional:.2f}"
            )
        else:
            risk = "no risk plan"
        lines.append(
            f"{signal.direction.value.upper()} {signal.symbol} status={signal.status.value} "
            f"strategy={signal.strategy} confidence={signal.confidence:.3f} expiry={expiry} "
            f"{risk} reason={signal.reason}"
        )
    return "\n".join(lines)
