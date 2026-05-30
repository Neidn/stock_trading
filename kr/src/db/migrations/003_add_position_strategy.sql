-- Migration 003: per-position strategy tracking
-- signal_engine writes strategy_name here on entry so weekly reports
-- can break down PF/win-rate per strategy without joining signals.
ALTER TABLE positions ADD COLUMN strategy_name TEXT;
