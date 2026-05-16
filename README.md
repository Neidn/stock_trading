# Stock Trading Screener and Signal System

Python project for manually maintained stock watchlists, daily market-data ingestion,
rule-based screening, strategy signals, risk sizing, and SQLite persistence.

This project intentionally starts as an alert/signal system. It does not place live
orders. Add broker execution only after backtesting and paper validation.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
python -m stock_trading.cli init-db
python -m stock_trading.cli ingest --start 2025-01-01
python -m stock_trading.cli screen
python -m stock_trading.cli signals --equity 10000
```

Edit `config/watchlist.yml` to set the stocks manually.

## Target Stocks

Target stocks are controlled by `config/watchlist.yml`:

```yaml
symbols:
  - AAPL
  - MSFT
  - NVDA
```

Use uppercase ticker symbols. Keep the list small while validating the strategy;
start with liquid stocks or ETFs you can review manually. After editing the
list, run:

```bash
python -m stock_trading.cli run-daily --start 2025-01-01 --equity 10000
```

The pipeline ingests bars for those symbols, screens them, and stores generated
signals with entry, stop, target, share count, risk, status, and expiry values.

## Docker Image CI

GitHub Actions builds and pushes a Docker image on pushes to `main` after tests
pass. Add these repository secrets in GitHub:

- `DOCKERHUB_TOKEN`

The workflow pushes:

```text
neidn/stock-trading:latest
neidn/stock-trading:main
neidn/stock-trading:sha-<commit>
```

For `DOCKERHUB_TOKEN`, paste only the raw access-token value, not the token name,
`Bearer ...`, quotes, or extra lines.

## Commands

```bash
# Create SQLite tables
python -m stock_trading.cli init-db

# Download daily OHLCV for symbols in config/watchlist.yml
python -m stock_trading.cli ingest --start 2025-01-01

# Run the screener and persist results
python -m stock_trading.cli screen

# Generate strategy signals for the latest screened symbols
python -m stock_trading.cli signals --equity 10000

# Ingest, screen, and generate signals in one run
python -m stock_trading.cli run-daily --start 2025-01-01 --equity 10000
```

## Project Layout

```text
src/stock_trading/
  alerts/        Console and future notification formatters
  data/          Market-data providers and ingestion
  db.py          SQLite schema and repository helpers
  risk/          Position sizing and portfolio limits
  screener/      Screening rules and engine
  signals/       Signal orchestration
  strategies/    Strategy implementations
  cli.py         Command-line entry point
  config.py      Runtime settings and watchlist loading
```

## Safety Position

- Manual stock list only.
- No live execution module in the initial build.
- Risk sizing is calculated for review; it is not sent to a broker.
- Signals include entry, stop, target, risk amount, and share quantity.
- Treat all output as research until validated with historical and paper results.

See `docs/IMPLEMENTATION.md` for the implementation plan and next phases.
