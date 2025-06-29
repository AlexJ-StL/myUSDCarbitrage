import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv() # take environment variables from .env.

class DBConnector:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "usdc_arbitrage"),
            user=os.getenv("DB_USER", "arb_user"),
            password=os.getenv("DB_PASSWORD", "strongpassword"),
            host=os.getenv("DB_HOST", "localhost")
        )
        self.create_tables()

    def create_tables(self):
        with self.conn:
            with self.conn.cursor() as cur:
                # User management (simplified)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        username VARCHAR(50) PRIMARY KEY,
                        roles TEXT[] NOT NULL
                    );
                    INSERT INTO users (username, roles)
                    VALUES ('admin', ARRAY['admin']), ('trader', ARRAY['user'])
                    ON CONFLICT DO NOTHING;
                """)

                # Strategy registry
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS strategies (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) UNIQUE NOT NULL,
                        file_path VARCHAR(255) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Backtest results
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50) NOT NULL REFERENCES users(username),
                        strategy VARCHAR(100) NOT NULL,
                        parameters JSONB NOT NULL,
                        results JSONB NOT NULL,
                        executed_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        exchange VARCHAR(20) NOT NULL,
                        symbol VARCHAR(10) NOT NULL,
                        timeframe VARCHAR(5) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        open NUMERIC(12,6) NOT NULL,
                        high NUMERIC(12,6) NOT NULL,
                        low NUMERIC(12,6) NOT NULL,
                        close NUMERIC(12,6) NOT NULL,
                        volume NUMERIC(20,8) NOT NULL,
                        UNIQUE (exchange, symbol, timeframe, timestamp)
                    );
                """)

    def get_user_roles(self, username):
        with self.conn.cursor() as cur:
            cur.execute("SELECT roles FROM users WHERE username = %s", (username,))
            return cur.fetchone()[0] if cur.rowcount > 0 else []

    def save_backtest_results(self, user, strategy, parameters, results):
        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO backtest_results
                    (user_id, strategy, parameters, results)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (user, strategy, psycopg2.extras.Json(parameters), psycopg2.extras.Json(results)))
                return cur.fetchone()[0]

    def get_ohlcv_data(self, exchange, symbol, timeframe, start=None, end=None):
        """Retrieve OHLCV data for validation/testing"""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE exchange = %s AND symbol = %s AND timeframe = %s
        """
        params = [exchange, symbol, timeframe]

        if start:
            query += " AND timestamp >= %s"
            params.append(start)
        if end:
            query += " AND timestamp <= %s"
            params.append(end)

        query += " ORDER BY timestamp"

        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
            return df
        except Exception as e:
            #logger.error(f"Database query failed: {e}")
            return pd.DataFrame()

    def get_connection(self):
        return self.conn