-- Enable TimescaleDB compression
c:\Users\AlexJ\Documents\Coding\Repos\my-repos\myUSDCarbitrage\enable_compression.sql
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'exchange, symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy (compress data older than 7 days)
SELECT add_compression_policy('market_data', INTERVAL '7 days');