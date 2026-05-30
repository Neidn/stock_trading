-- ============================================================
-- src/db/migrations/001_init.sql
-- Binance Futures Autotrading — initial schema
-- ============================================================

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA cache_size=-64000;


-- 1. 심볼 목록
CREATE TABLE IF NOT EXISTS symbols (
    symbol          TEXT PRIMARY KEY,
    base_asset      TEXT NOT NULL,
    quote_asset     TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1,
    added_at        TEXT NOT NULL DEFAULT (datetime('now'))
);


-- 2. OHLCV 캔들
CREATE TABLE IF NOT EXISTS klines (
    id              TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    interval_type   TEXT NOT NULL,
    open_time       TEXT NOT NULL,
    open            TEXT NOT NULL,
    high            TEXT NOT NULL,
    low             TEXT NOT NULL,
    close           TEXT NOT NULL,
    volume          TEXT NOT NULL,
    close_time      TEXT NOT NULL,
    FOREIGN KEY (symbol) REFERENCES symbols(symbol),
    UNIQUE (symbol, interval_type, open_time)
);
CREATE INDEX IF NOT EXISTS idx_klines ON klines (symbol, interval_type, open_time DESC);


-- 3. 신호 이력
CREATE TABLE IF NOT EXISTS signals (
    signal_id       TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    signal_type     TEXT NOT NULL CHECK (signal_type IN ('long','short','close')),
    strategy_name   TEXT NOT NULL,
    strength_score  INTEGER NOT NULL CHECK (strength_score BETWEEN 1 AND 3),
    entry_price     TEXT,
    tp_price        TEXT,
    sl_price        TEXT,
    indicators_json TEXT,
    blocked         INTEGER NOT NULL DEFAULT 0,
    block_reason    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
);
CREATE INDEX IF NOT EXISTS idx_signals ON signals (symbol, created_at DESC);


-- 4. 주문 이력
CREATE TABLE IF NOT EXISTS orders (
    order_id            TEXT PRIMARY KEY,
    binance_order_id    INTEGER,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL CHECK (side IN ('buy','sell')),
    position_side       TEXT NOT NULL DEFAULT 'both' CHECK (position_side IN ('long','short','both')),
    order_type          TEXT NOT NULL,
    price               TEXT,
    quantity            TEXT NOT NULL,
    filled_qty          TEXT NOT NULL DEFAULT '0',
    avg_fill_price      TEXT,
    status              TEXT NOT NULL,
    signal_id           TEXT,
    fee                 TEXT NOT NULL DEFAULT '0',
    fee_asset           TEXT,
    trading_mode        TEXT NOT NULL DEFAULT 'paper'
                            CHECK (trading_mode IN ('paper','live')),
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT,
    FOREIGN KEY (symbol)    REFERENCES symbols(symbol),
    FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_mode   ON orders (trading_mode, created_at DESC);


-- 5. 포지션 이력
CREATE TABLE IF NOT EXISTS positions (
    position_id             TEXT PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    side                    TEXT NOT NULL CHECK (side IN ('long','short')),
    leverage                INTEGER NOT NULL DEFAULT 1,
    entry_price             TEXT NOT NULL,
    exit_price              TEXT,
    quantity                TEXT NOT NULL,
    liquidation_price       TEXT DEFAULT '0',
    stop_loss               TEXT NOT NULL,
    take_profit_1           TEXT,
    take_profit_2           TEXT,
    initial_stop_loss       TEXT NOT NULL,
    trailing_activated      INTEGER DEFAULT 0,
    realized_pnl            TEXT DEFAULT '0',
    unrealized_pnl          TEXT DEFAULT '0',
    status                  TEXT NOT NULL DEFAULT 'open'
                                CHECK (status IN ('open','closed','liquidated','force_closed')),
    close_reason            TEXT,
    trading_mode            TEXT NOT NULL DEFAULT 'paper'
                                CHECK (trading_mode IN ('paper','live')),
    opened_at               TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at               TEXT,
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status, symbol);
CREATE INDEX IF NOT EXISTS idx_positions_mode   ON positions (trading_mode, status);


-- 6. 청산 위험 이벤트
CREATE TABLE IF NOT EXISTS liquidation_events (
    event_id            TEXT PRIMARY KEY,
    position_id         TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    event_type          TEXT NOT NULL
                            CHECK (event_type IN ('WATCH','WARNING','CRITICAL','CASCADE_RISK')),
    current_price       TEXT NOT NULL,
    liquidation_price   TEXT NOT NULL,
    distance_pct        TEXT NOT NULL,
    action_taken        TEXT NOT NULL,
    oi_change_5m        TEXT,
    large_liquidations  TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (position_id) REFERENCES positions(position_id)
);


-- 7. 시장 급변 감지 로그
CREATE TABLE IF NOT EXISTS market_shock_events (
    event_id            TEXT PRIMARY KEY,
    risk_level          TEXT NOT NULL CHECK (risk_level IN ('ELEVATED','DANGER')),
    oi_change_5m        TEXT,
    large_liquidations  TEXT,
    price_change_1m     TEXT,
    funding_rate        TEXT,
    risk_score          INTEGER NOT NULL,
    action_taken        TEXT NOT NULL,
    affected_positions  TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);


-- 8. 안전 모드 이벤트
CREATE TABLE IF NOT EXISTS safe_mode_events (
    event_id    TEXT PRIMARY KEY,
    action      TEXT NOT NULL CHECK (action IN ('activated','deactivated')),
    reason      TEXT NOT NULL,
    by          TEXT NOT NULL DEFAULT 'system',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);


-- 9. 포지션 동기화 이력
CREATE TABLE IF NOT EXISTS sync_events (
    event_id        TEXT PRIMARY KEY,
    success         INTEGER NOT NULL,
    discrepancies   INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);


-- 10. 일일 성과
CREATE TABLE IF NOT EXISTS daily_performance (
    perf_date               TEXT NOT NULL,
    trading_mode            TEXT NOT NULL DEFAULT 'paper'
                                CHECK (trading_mode IN ('paper','live')),
    total_trades            INTEGER DEFAULT 0,
    winning_trades          INTEGER DEFAULT 0,
    losing_trades           INTEGER DEFAULT 0,
    liquidated_trades       INTEGER DEFAULT 0,
    gross_profit            TEXT DEFAULT '0',
    gross_loss              TEXT DEFAULT '0',
    net_pnl                 TEXT DEFAULT '0',
    total_fees              TEXT DEFAULT '0',
    max_drawdown            TEXT DEFAULT '0',
    win_rate                TEXT DEFAULT '0',
    avg_liquidation_distance TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (perf_date, trading_mode)
);


-- 11. 시스템 이벤트 로그
CREATE TABLE IF NOT EXISTS system_events (
    event_id    TEXT PRIMARY KEY,
    event_type  TEXT NOT NULL,
    severity    TEXT NOT NULL CHECK (severity IN ('info','warning','error','critical')),
    module      TEXT NOT NULL,
    message     TEXT,
    metadata    TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sys_events ON system_events (severity, created_at DESC);
