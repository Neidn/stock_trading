-- Migration 002: per-symbol strategy assignment
-- Screener writes strategy name here; StrategyRunner reads it.
-- NULL = use global ACTIVE_STRATEGY fallback.

-- guarded via init_db ignore-duplicate logic; ALTER TABLE has no IF NOT EXISTS in SQLite
ALTER TABLE symbols ADD COLUMN strategy TEXT;
