"""Unit tests for environment configuration validation."""

from __future__ import annotations

import pytest

from src.utils.config import TradingMode, load_config


def _set_live_phase5_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("LIVE_CONFIRM", "LIVE")
    monkeypatch.setenv("ACTIVE_STRATEGY", "rsi_macd")
    monkeypatch.setenv("MAX_LEVERAGE", "3")
    monkeypatch.setenv("RISK_PER_TRADE", "0.005")
    monkeypatch.setenv("MAX_POSITIONS", "3")
    monkeypatch.setenv("DAILY_LOSS_LIMIT", "0.03")
    monkeypatch.setenv("BINANCE_API_KEY", "dummy-live-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "dummy-live-secret")


def test_live_phase5_caps_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_live_phase5_env(monkeypatch)

    config = load_config()

    assert config.trading_mode is TradingMode.LIVE
    assert config.max_leverage == 3
    assert config.risk_per_trade == 0.005
    assert config.max_positions == 3
    assert config.daily_loss_limit == 0.03


@pytest.mark.parametrize(
    ("env_name", "unsafe_value", "message"),
    [
        ("MAX_LEVERAGE", "4", "MAX_LEVERAGE=4 > 3"),
        ("RISK_PER_TRADE", "0.006", "RISK_PER_TRADE=0.006 > 0.005"),
        ("MAX_POSITIONS", "6", "MAX_POSITIONS=6 > 5"),
        ("DAILY_LOSS_LIMIT", "0.031", "DAILY_LOSS_LIMIT=0.031 > 0.03"),
    ],
)
def test_live_rejects_values_above_phase5_caps(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    unsafe_value: str,
    message: str,
) -> None:
    _set_live_phase5_env(monkeypatch)
    monkeypatch.setenv(env_name, unsafe_value)

    with pytest.raises(ValueError, match="Unsafe live configuration") as exc_info:
        load_config()

    assert message in str(exc_info.value)


def test_live_requires_live_api_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_live_phase5_env(monkeypatch)
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="BINANCE_API_KEY and BINANCE_API_SECRET"):
        load_config()


def test_testnet_allows_values_above_live_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_MODE", "testnet")
    monkeypatch.setenv("ACTIVE_STRATEGY", "rsi_macd")
    monkeypatch.setenv("MAX_LEVERAGE", "5")
    monkeypatch.setenv("RISK_PER_TRADE", "0.01")
    monkeypatch.setenv("MAX_POSITIONS", "5")
    monkeypatch.setenv("DAILY_LOSS_LIMIT", "0.05")
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

    config = load_config()

    assert config.trading_mode is TradingMode.TESTNET
    assert config.max_leverage == 5
