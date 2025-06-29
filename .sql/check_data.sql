-- Check data completeness by exchange and timeframe
SELECT
    exchange,
    timeframe,
    COUNT(*) AS record_count, -- More descriptive alias
    MIN(timestamp) AS first_timestamp, -- Avoid reserved words
    MAX(timestamp) AS last_timestamp
FROM market_data
    -- Optional: Add WHERE clause to filter specific time ranges
    -- WHERE timestamp BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY
    exchange,
    timeframe;