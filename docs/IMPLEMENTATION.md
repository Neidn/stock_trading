# Stock Screener and Signal System Implementation

## Objective

Build a stock research system that starts with a manually maintained watchlist,
screens symbols every day, generates strategy signals, and stores all outputs in
SQLite for review. The first version is signal-only and has no live broker
execution.

## Current MVP

Implemented modules:

- Manual watchlist: `config/watchlist.yml`
- Runtime settings: `.env.example` and `stock_trading.config`
- Market data providers: `yfinance` and local CSV provider
- SQLite persistence: bars, screener results, and signals
- Screener: price, liquidity, relative volume, and uptrend filters
- Strategy: momentum breakout with ATR stop and reward-risk target
- Risk sizing: risk-per-trade and max-position-percent caps
- Signal state: typed direction/status fields and optional expiry
- CLI: `init-db`, `ingest`, `screen`, `signals`, and `run-daily`
- Tests: risk sizing, screener, and strategy signal generation

## Manual Stock Setup

Edit `config/watchlist.yml`:

```yaml
symbols:
  - AAPL
  - MSFT
  - NVDA
```

Keep this list intentionally small at first. Start with liquid stocks or ETFs
that you understand well.

## Local Setup

```bash
cd ../stock_trading
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Initialize the database:

```bash
python -m stock_trading.cli init-db
```

Run the daily pipeline:

```bash
python -m stock_trading.cli run-daily --start 2025-01-01 --equity 10000
```

## Data Provider Plan

The MVP uses `yfinance` for convenience and local CSV files for repeatable
testing. For production-grade research, add a paid provider behind the existing
`MarketDataProvider` interface.

Recommended next providers:

- Polygon for stock market data.
- Alpaca for data plus paper/live brokerage workflow.
- Interactive Brokers for broker execution after the signal engine is validated.

Do not mix data-provider logic into strategies. Keep providers behind
`src/stock_trading/data/providers.py`.

## Screener Rules

The default screener passes symbols when:

- price is above the minimum price
- average dollar volume is above the liquidity floor
- relative volume is acceptable
- latest close is above fast SMA and fast SMA is above slow SMA

Config lives in `config/watchlist.yml`:

```yaml
screener:
  min_price: 5.0
  min_avg_dollar_volume: 20000000
  min_relative_volume: 0.8
  require_uptrend: true
```

## Strategy Rules

The initial strategy is `momentum_breakout`.

Signal logic:

- calculate the prior N-bar high
- calculate ATR
- trigger a long signal when close breaks above the prior N-bar high
- set stop at `close - ATR * atr_stop_multiple`
- set target from configured reward-risk multiple
- size shares from account equity and risk limits

Config:

```yaml
strategy:
  active: momentum_breakout
  breakout_lookback: 20
  atr_period: 14
  atr_stop_multiple: 2.0
  reward_risk: 2.0
  signal_expiry_days: 5
```

## Risk Defaults

Environment variables:

```bash
STOCK_TRADING_RISK_PER_TRADE=0.005
STOCK_TRADING_MAX_POSITION_PCT=0.10
STOCK_TRADING_MAX_OPEN_POSITIONS=5
```

For stock research, start conservative:

- 0.25% to 0.50% risk per trade during paper validation
- max 5 open positions
- no short selling until borrow and margin rules are explicitly modeled
- avoid holding through earnings until earnings-date filtering exists

## Next Implementation Phases

### Phase 1: Strengthen Signal State

Track every generated signal with explicit state:

- direction: `buy`, `sell`, `hold`, or `exit_watch`
- status: `new`, `watching`, `entered_manual`, `closed_manual`, `expired`, or
  `cancelled`
- optional expiry datetime
- idempotent SQLite schema updates for existing local databases

This phase creates the data model needed for dashboards, manual trading
decisions, price tracking, and later paper/live workflows.

### Phase 1.5: Local Dashboard

Add a small local dashboard before broker execution work:

- latest screener table
- signal table
- entry, stop, target, shares, risk, notional, status, and expiry values
- manual watchlist editing for target stocks
- chart links
- historical signal outcome
- strategy performance summary

Streamlit is the fastest first choice for a personal local tool. FastAPI or
Flask can replace it later when the project needs an API, auth, or multi-user
access.

### Phase 2: Broker API Read-Only Integration

Add broker/data adapters for quotes, bars, account snapshots, and watchlist
verification. Keep order methods absent or explicitly blocked.

Preferred first adapter: KIS. Add Kiwoom behind the same interface later.

### Phase 3: Price Tracking And Alerts

Track active signals and persist price observations and signal events:

- entry zone reached
- stop touched
- target touched
- signal expired
- price moved against setup before entry

### Phase 4: Manual Trade Journal

Store manual entry and exit records linked to signal IDs:

- planned entry vs actual entry
- planned exit vs actual exit
- realized PnL
- target hit rate
- stop hit rate

### Phase 5: Public Data Validation

Add DART, ECOS, KRX/KIND, and earnings/event filters to block or annotate risky
signals.

### Phase 6: Backtesting And Paper Trading

Add a backtest engine that replays daily bars and stores:

- trades
- equity curve
- drawdown
- win rate
- average win/loss
- exposure
- turnover

Use adjusted OHLCV and account for slippage and fees. Paper trading should run
for several weeks before any live automation is considered.

### Phase 7: Alerts Delivery

Add Telegram or email alerts after the daily run:

- send only actionable signals
- include entry, stop, target, shares, status, expiry, and reason
- include "research only" label until paper trading is complete

### Phase 8: Broker Execution

Only add execution after backtesting and paper mode are stable.

Execution requirements:

- separate `TRADING_MODE=paper/live`
- live mode impossible to trigger by default
- broker reconciliation before and after each order
- complete order and position persistence
- max daily loss and max position limits enforced before orders
- manual kill switch

## Testing

Run:

```bash
pytest
```

Add tests for every new strategy:

- enough-data and insufficient-data behavior
- signal trigger and non-trigger cases
- stop/target math
- position sizing boundaries
- persistence behavior

## Operating Checklist

Before trusting any signal:

- confirm data source adjustment behavior
- inspect at least 20 generated signals manually
- backtest across bull, bear, and sideways periods
- verify slippage assumptions
- verify earnings filter coverage
- paper trade for several weeks
- compare expected fills with real market prices

Do not add live trading until the system can explain every signal and recover
state from SQLite without relying on memory.
