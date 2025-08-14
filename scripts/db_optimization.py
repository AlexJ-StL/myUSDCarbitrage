#!/usr/bin/env python3
"""
Database and Storage Optimization Script

This script implements:
1. TimescaleDB compression policies for historical data
2. Automated backup and recovery procedures
3. Database connection pooling configuration
4. Query optimization through index creation and analysis
5. Data archiving strategy for long-term storage
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta

import psycopg2
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("database_optimization.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("db_optimization")

# Load environment variables
load_dotenv()

# Database connection parameters
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME", "usdc_arbitrage"),
    "user": os.getenv("DB_USER", "arb_user"),
    "password": os.getenv("DB_PASSWORD", "strongpassword"),
    "host": os.getenv("DB_HOST", "localhost"),
}


def execute_sql(sql, params=None):
    """Execute SQL statement and return results."""
    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                conn.commit()
                try:
                    return cur.fetchall()
                except psycopg2.ProgrammingError:
                    return None
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise


def setup_compression_policies():
    """Set up TimescaleDB compression policies for historical data."""
    logger.info("Setting up TimescaleDB compression policies")

    # Check if compression is already enabled
    check_sql = """
    SELECT compression_enabled 
    FROM timescaledb_information.hypertables 
    WHERE hypertable_name = 'market_data';
    """

    try:
        result = execute_sql(check_sql)
        if result and result[0][0]:
            logger.info("Compression already enabled for market_data table")
        else:
            # Enable compression
            enable_sql = """
            ALTER TABLE market_data SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'exchange, symbol',
                timescaledb.compress_orderby = 'timestamp DESC'
            );
            """
            execute_sql(enable_sql)
            logger.info("Compression enabled for market_data table")

            # Add compression policy (compress data older than 7 days)
            policy_sql = """
            SELECT add_compression_policy('market_data', INTERVAL '7 days');
            """
            execute_sql(policy_sql)
            logger.info("Compression policy added: compress data older than 7 days")
    except Exception as e:
        logger.error(f"Failed to set up compression: {e}")


def create_backup(backup_dir="backups"):
    """Create a database backup."""
    logger.info("Creating database backup")

    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)

    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"usdc_arbitrage_{timestamp}.sql")

    # Create backup using pg_dump
    cmd = [
        "pg_dump",
        f"--dbname=postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}/{DB_PARAMS['dbname']}",
        "--format=custom",
        f"--file={backup_file}",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Backup created successfully: {backup_file}")
        return backup_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Backup failed: {e.stderr.decode()}")
        raise


def restore_backup(backup_file):
    """Restore database from backup."""
    logger.info(f"Restoring database from backup: {backup_file}")

    if not os.path.exists(backup_file):
        logger.error(f"Backup file not found: {backup_file}")
        return False

    # Restore using pg_restore
    cmd = [
        "pg_restore",
        "--clean",
        "--if-exists",
        f"--dbname=postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}/{DB_PARAMS['dbname']}",
        backup_file,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("Database restored successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Restore failed: {e.stderr.decode()}")
        return False


def optimize_indexes():
    """Create and optimize database indexes."""
    logger.info("Optimizing database indexes")

    # Check existing indexes
    check_indexes_sql = """
    SELECT indexname, indexdef
    FROM pg_indexes
    WHERE tablename = 'market_data';
    """

    existing_indexes = execute_sql(check_indexes_sql)
    existing_index_names = (
        [idx[0] for idx in existing_indexes] if existing_indexes else []
    )

    # Define indexes to ensure they exist
    indexes = [
        {
            "name": "idx_market_data_time",
            "sql": "CREATE INDEX IF NOT EXISTS idx_market_data_time ON market_data (timestamp);",
        },
        {
            "name": "idx_market_data_exchange",
            "sql": "CREATE INDEX IF NOT EXISTS idx_market_data_exchange ON market_data (exchange);",
        },
        {
            "name": "idx_market_data_symbol",
            "sql": "CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol);",
        },
        {
            "name": "idx_market_data_exchange_symbol",
            "sql": "CREATE INDEX IF NOT EXISTS idx_market_data_exchange_symbol ON market_data (exchange, symbol);",
        },
        {
            "name": "idx_market_data_timeframe",
            "sql": "CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data (timeframe);",
        },
    ]

    # Create missing indexes
    for idx in indexes:
        if idx["name"] not in existing_index_names:
            execute_sql(idx["sql"])
            logger.info(f"Created index: {idx['name']}")
        else:
            logger.info(f"Index already exists: {idx['name']}")

    # Analyze tables for query optimization
    analyze_sql = "ANALYZE market_data;"
    execute_sql(analyze_sql)
    logger.info("Analyzed market_data table for query optimization")


def setup_connection_pooling():
    """Configure database connection pooling settings."""
    logger.info("Setting up database connection pooling")

    # These settings would typically be in postgresql.conf
    # Here we're just logging the recommended settings
    pooling_settings = """
    # Connection Pooling Settings for postgresql.conf
    
    # Maximum number of connections
    max_connections = 100
    
    # Connection pooling timeout (in milliseconds)
    idle_in_transaction_session_timeout = 60000
    
    # Statement timeout (in milliseconds)
    statement_timeout = 30000
    
    # Connection lifetime (in milliseconds)
    idle_session_timeout = 600000
    """

    logger.info("Connection pooling settings to add to postgresql.conf:")
    logger.info(pooling_settings)

    # For application-side pooling, we would configure SQLAlchemy
    sqlalchemy_pooling = """
    # SQLAlchemy Connection Pooling Configuration
    
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,               # Maximum number of connections to keep
        max_overflow=20,            # Maximum number of connections that can be created beyond pool_size
        pool_timeout=30,            # Seconds to wait before timing out on getting a connection from the pool
        pool_recycle=1800,          # Recycle connections after 30 minutes
        pool_pre_ping=True,         # Enable connection health checks
    )
    """

    logger.info("SQLAlchemy connection pooling configuration:")
    logger.info(sqlalchemy_pooling)


def implement_data_archiving(archive_days=90):
    """Implement data archiving strategy for long-term storage."""
    logger.info(f"Implementing data archiving for data older than {archive_days} days")

    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=archive_days)

    # Create archive table if it doesn't exist
    create_archive_table_sql = """
    CREATE TABLE IF NOT EXISTS market_data_archive (
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
        archived_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (exchange, symbol, timeframe, timestamp)
    );
    """
    execute_sql(create_archive_table_sql)

    # Move old data to archive table
    archive_data_sql = """
    WITH archived_rows AS (
        DELETE FROM market_data
        WHERE timestamp < %s
        RETURNING *
    )
    INSERT INTO market_data_archive (
        exchange, symbol, timeframe, timestamp, open, high, low, close, volume
    )
    SELECT 
        exchange, symbol, timeframe, timestamp, open, high, low, close, volume
    FROM archived_rows
    ON CONFLICT (exchange, symbol, timeframe, timestamp) DO NOTHING;
    """

    try:
        execute_sql(archive_data_sql, (cutoff_date,))
        logger.info(f"Archived data older than {cutoff_date}")
    except Exception as e:
        logger.error(f"Failed to archive data: {e}")


def main():
    """Main function to run database optimization tasks."""
    parser = argparse.ArgumentParser(description="Database optimization tools")
    parser.add_argument(
        "--compress", action="store_true", help="Set up TimescaleDB compression"
    )
    parser.add_argument("--backup", action="store_true", help="Create database backup")
    parser.add_argument("--restore", help="Restore database from backup file")
    parser.add_argument(
        "--optimize-indexes", action="store_true", help="Optimize database indexes"
    )
    parser.add_argument(
        "--setup-pooling", action="store_true", help="Setup connection pooling"
    )
    parser.add_argument("--archive", action="store_true", help="Archive old data")
    parser.add_argument(
        "--archive-days", type=int, default=90, help="Days threshold for archiving"
    )
    parser.add_argument("--all", action="store_true", help="Run all optimization tasks")

    args = parser.parse_args()

    if args.all or args.compress:
        setup_compression_policies()

    if args.all or args.backup:
        create_backup()

    if args.restore:
        restore_backup(args.restore)

    if args.all or args.optimize_indexes:
        optimize_indexes()

    if args.all or args.setup_pooling:
        setup_connection_pooling()

    if args.all or args.archive:
        implement_data_archiving(args.archive_days)

    logger.info("Database optimization tasks completed")


if __name__ == "__main__":
    main()
