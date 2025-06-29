CREATE DATABASE usdc_arbitrage;

CREATE USER arb_user WITH PASSWORD 'strongpassword';

GRANT ALL PRIVILEGES ON DATABASE usdc_arbitrage TO arb_user;

-- Connect to database
\c usdc_arbitrage

-- Create timeseries-optimized table
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(12, 6) NOT NULL,
    high NUMERIC(12, 6) NOT NULL,
    low NUMERIC(12, 6) NOT NULL,
    close NUMERIC(12, 6) NOT NULL,
    volume NUMERIC(20, 8) NOT NULL,
    UNIQUE (
        exchange,
        symbol,
        timeframe,
        timestamp
    )
);

-- Optimize for time-based queries
CREATE INDEX idx_market_data_time ON market_data (timestamp);

CREATE INDEX idx_market_data_exchange ON market_data (exchange);

SELECT create_hypertable ('market_data', 'timestamp');

-- Grant permissions
GRANT SELECT, INSERT , UPDATE, DELETE ON market_data TO arb_user;

GRANT USAGE, SELECT ON SEQUENCE market_data_id_seq TO arb_user;