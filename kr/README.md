# Binance Futures Autotrading

24/7 automated crypto futures trading system for Binance USDT-M Futures. Prioritizes **liquidation avoidance and uninterrupted operation** over maximum profit.

**Status**: Live trading — first position placed 2026-05-21.

## Architecture

```
WebSocket → Data Ingest → Signal Engine → LiquidationGuard → OrderManager
                ↓                                                   ↓
           SQLite DB ←──────────────────────────────── Position Tracker
                ↓
         Safety Monitor (independent pod)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13+ |
| Exchange | ccxt (Binance USDT-M Futures) |
| Indicators | TA-Lib (C-compiled) |
| Database | SQLite WAL mode (K8s PVC `/data/trading.db`) |
| Orchestration | Kubernetes |
| Alerts/Control | Telegram bot |
| Dashboard | Flask |

## Strategies

Screener assigns strategy per symbol based on ADX + ATR regime:

| ADX Zone | Strategy | Signal Type |
|----------|----------|-------------|
| < 25 (ranging) | `bb_breakout` | Bollinger squeeze breakout |
| 25–32 (transitioning) | `rsi_supertrend` | SuperTrend flip + RSI confirm |
| 32–55 (trending) | `ema_pullback_rsi` | Multi-EMA alignment + RSI pullback |
| 55+ (strong trend) | `macd_sma200_chartart` | MACD cross + SMA200 filter |

All strategies inherit `BaseStrategy`. Hot-swap via `ACTIVE_STRATEGY` ConfigMap — no code changes needed.

## K8s Pods

| Pod | Role | Schedule |
|-----|------|----------|
| `signal-engine` | Strategy execution + order placement | Continuous (1h cycle) |
| `data-ingest` | WebSocket candle collection | Continuous |
| `safety-monitor` | Emergency stop, liquidation watch | Continuous |
| `dashboard` | Flask UI | Continuous |
| `screener` | Symbol selection + strategy assignment | Daily 00:00 UTC |
| `position-sync` | Reconcile DB with Binance | Every 30 min |
| `db-archiver` | Archive old candles | Weekly |

## Configuration

All runtime config via K8s ConfigMap (`k8s/configmaps/trading-config.yaml`):

```yaml
ACTIVE_STRATEGY: "ema_pullback_rsi"   # fallback; screener assigns per symbol
TRADING_MODE: "live"                  # testnet | live
MAX_POSITIONS: "3"
RISK_PER_TRADE: "0.005"              # 0.5% per trade
MAX_LEVERAGE: "3"
DAILY_LOSS_LIMIT: "0.03"             # 3% daily drawdown halt
```

Secrets (`k8s/secrets/`): Binance API keys, Telegram token.

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/
python -m pytest tests/test_strategy.py  # single file
```

## Design Principles

**Liquidation Avoidance** — #1 priority:
- `LiquidationGuard`: WATCH → WARNING → CRITICAL levels
- At CRITICAL: auto-reduce positions before liquidation
- `SafetyMonitor` in separate pod — survives main pod crashes
- Telegram `/emergency_close` closes all positions immediately

**Data Integrity**:
- Full state written to SQLite after every order/position change
- Never trust in-memory state; re-read DB on recovery
- `position-sync` CronJob: exchange API is ground truth

**Strategy Swap**:
- Change `ACTIVE_STRATEGY` in ConfigMap + `kubectl rollout restart` — zero code changes
- Screener auto-assigns per-symbol strategy based on live market regime

**Binance Account Requirements**:
- Position mode: **Hedge Mode** (supports simultaneous LONG/SHORT per symbol)
