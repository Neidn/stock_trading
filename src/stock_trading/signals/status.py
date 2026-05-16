from __future__ import annotations

from dataclasses import replace

from stock_trading.models import Signal, SignalStatus


ALLOWED_STATUS_TRANSITIONS: dict[SignalStatus, set[SignalStatus]] = {
    SignalStatus.NEW: {
        SignalStatus.WATCHING,
        SignalStatus.ENTERED_MANUAL,
        SignalStatus.EXPIRED,
        SignalStatus.CANCELLED,
    },
    SignalStatus.WATCHING: {
        SignalStatus.ENTERED_MANUAL,
        SignalStatus.EXPIRED,
        SignalStatus.CANCELLED,
    },
    SignalStatus.ENTERED_MANUAL: {
        SignalStatus.CLOSED_MANUAL,
        SignalStatus.CANCELLED,
    },
    SignalStatus.CLOSED_MANUAL: set(),
    SignalStatus.EXPIRED: set(),
    SignalStatus.CANCELLED: set(),
}


def transition_signal_status(signal: Signal, status: SignalStatus | str) -> Signal:
    target = SignalStatus(status)
    if signal.status == target:
        return signal

    allowed = ALLOWED_STATUS_TRANSITIONS[signal.status]
    if target not in allowed:
        raise ValueError(f"cannot transition signal status from {signal.status.value} to {target.value}")

    return replace(signal, status=target)
