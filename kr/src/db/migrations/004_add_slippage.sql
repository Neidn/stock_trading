-- Migration 004: slippage tracking
-- Records actual exchange fill price vs signal entry price.
-- slippage_bps = (fill_price - entry_price) / entry_price * 10000
-- Positive = filled higher than expected; negative = filled lower.
ALTER TABLE positions ADD COLUMN fill_price TEXT;
ALTER TABLE positions ADD COLUMN slippage_bps REAL;
