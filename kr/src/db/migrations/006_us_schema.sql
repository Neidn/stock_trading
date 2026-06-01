-- ============================================================
-- 006_us_schema.sql
-- US overseas stock support via KIS API
-- ============================================================

-- ── symbols: add US exchange code ────────────────────────────
-- excd: KIS 3-char exchange code — NAS (NASDAQ), NYS (NYSE), AMS (AMEX)
-- NULL for KRX domestic stocks.
ALTER TABLE symbols ADD COLUMN excd TEXT;

-- ── symbols: add currency tag ────────────────────────────────
-- currency: KRW for domestic, USD for US stocks.
ALTER TABLE symbols ADD COLUMN currency TEXT DEFAULT 'KRW';

-- ── positions: store USD amounts for US trades ───────────────
-- entry/exit/sl/tp are already TEXT so they hold USD decimals.
-- Add currency column so P&L display knows the denomination.
ALTER TABLE positions ADD COLUMN currency TEXT DEFAULT 'KRW';

-- ── signals: currency tag ────────────────────────────────────
ALTER TABLE signals ADD COLUMN currency TEXT DEFAULT 'KRW';
