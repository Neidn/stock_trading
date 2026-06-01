-- ============================================================
-- 005_kr_schema.sql
-- Binance Futures → KR Spot: schema changes
-- All ALTER TABLE ADD COLUMN are idempotent (duplicate column
-- errors are swallowed by init_db).
-- ============================================================

-- ── symbols: add KR market metadata ──────────────────────────
ALTER TABLE symbols ADD COLUMN market TEXT;         -- KOSPI | KOSDAQ
ALTER TABLE symbols ADD COLUMN sector TEXT;
ALTER TABLE symbols ADD COLUMN market_cap TEXT;     -- 시가총액 (억원, string)
ALTER TABLE symbols ADD COLUMN strategy TEXT;       -- assigned strategy name

-- ── orders: broker-neutral order ID column ────────────────────
ALTER TABLE orders ADD COLUMN broker_order_id TEXT;

-- ── positions: remove futures cols (mark nullable / add KR cols)
-- SQLite <3.35 cannot DROP COLUMN; leave legacy cols as NULL.
-- New KR-specific columns:
ALTER TABLE positions ADD COLUMN market TEXT DEFAULT 'KOSPI';
ALTER TABLE positions ADD COLUMN tax_paid TEXT;         -- 증권거래세 (원)
ALTER TABLE positions ADD COLUMN t2_settle_date TEXT;   -- T+2 결제일
ALTER TABLE positions ADD COLUMN fill_price TEXT;       -- actual fill price
ALTER TABLE positions ADD COLUMN slippage_bps TEXT;     -- slippage in bps

-- ── T+2 settlement tracker ────────────────────────────────────
CREATE TABLE IF NOT EXISTS settlements (
    settle_id       TEXT PRIMARY KEY,
    position_id     TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    sell_date       TEXT NOT NULL,
    settle_date     TEXT NOT NULL,  -- sell_date + 2 영업일
    amount_krw      TEXT NOT NULL,
    settled         INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT '1970-01-01',
    FOREIGN KEY (position_id) REFERENCES positions(position_id)
);
CREATE INDEX IF NOT EXISTS idx_settlements_settle ON settlements (settle_date, settled);

-- ── Cash availability ledger (T+2 constraint) ─────────────────
CREATE TABLE IF NOT EXISTS cash_ledger (
    ledger_id       TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL
                        CHECK (event_type IN ('buy','sell','settle','deposit','withdrawal')),
    amount_krw      TEXT NOT NULL,  -- positive = credit, negative = debit
    available_date  TEXT NOT NULL,  -- when cash becomes usable
    position_id     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cash_ledger_avail ON cash_ledger (available_date);
