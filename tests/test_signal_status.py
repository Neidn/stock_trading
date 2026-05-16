from datetime import datetime

import pytest

from stock_trading.models import Signal, SignalDirection, SignalStatus
from stock_trading.signals.status import transition_signal_status


def make_signal(status: SignalStatus = SignalStatus.NEW) -> Signal:
    return Signal(
        symbol="TEST",
        strategy="unit",
        as_of=datetime(2025, 1, 1),
        direction=SignalDirection.BUY,
        confidence=0.8,
        reason="unit test",
        status=status,
    )


def test_transition_signal_status_returns_updated_signal() -> None:
    signal = make_signal()

    updated = transition_signal_status(signal, SignalStatus.WATCHING)

    assert signal.status == SignalStatus.NEW
    assert updated.status == SignalStatus.WATCHING


def test_transition_signal_status_rejects_terminal_transition() -> None:
    signal = make_signal(SignalStatus.CLOSED_MANUAL)

    with pytest.raises(ValueError, match="cannot transition"):
        transition_signal_status(signal, SignalStatus.WATCHING)
