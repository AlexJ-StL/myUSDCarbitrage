-- Run this in PostgreSQL to check data:
SELECT
    exchange,
    timeframe,
    COUNT(**) as records,
    MIN(timestamp) as start,
    MAX(timestamp) as end
FROM market_data
GROUP BY exchange, timeframe;