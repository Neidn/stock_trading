-- 007_api_tokens.sql
-- Persist KIS access tokens across pod restarts to avoid EGW00133 rate-limit
-- on startup (KIS allows only 1 token fetch per minute per app key).
CREATE TABLE IF NOT EXISTS api_tokens (
    key_id       TEXT PRIMARY KEY,   -- 'order' | 'data'
    access_token TEXT NOT NULL,
    expires_at   REAL NOT NULL       -- unix timestamp
);
