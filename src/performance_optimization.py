"""
Performance Optimization Module for USDC Arbitrage Backtesting System.

This module implements various performance optimizations including:
1. Database query optimization and indexing strategies
2. Caching strategies for frequently accessed data
3. Memory usage optimization and garbage collection for long-running processes
4. System configuration tuning for production workloads
"""

import gc
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import psutil
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("performance_optimization.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# =====================================================================
# Database Query Optimization
# =====================================================================


def create_database_indexes(connection_string: str) -> None:
    """
    Create optimized database indexes for common query patterns.

    Args:
        connection_string: Database connection string
    """
    try:
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            # Composite index for market data queries (most common query pattern)
            connection.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_market_data_composite 
                ON market_data (exchange, symbol, timeframe, timestamp);
            """)
            )

            # Index for time range queries
            connection.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_market_data_time_range 
                ON market_data (timestamp, exchange, symbol, timeframe);
            """)
            )

            # Index for data validation queries
            connection.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_validation_issues 
                ON data_validation_issues (exchange, symbol, timeframe, issue_type);
            """)
            )

            # Index for gap detection queries
            connection.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_data_gaps 
                ON data_gaps (exchange, symbol, timeframe, filled);
            """)
            )

            # Index for strategy queries
            connection.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_strategies_type_active 
                ON strategies (strategy_type, is_active);
            """)
            )

            # Index for backtest results queries
            connection.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy 
                ON backtest_results (strategy_id, strategy_version);
            """)
            )

            # Index for user role queries
            connection.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_user_roles_user 
                ON user_roles (user_id);
            """)
            )

            connection.commit()
            logger.info("Database indexes created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error creating database indexes: {e}")
        raise


def optimize_query_plans(connection_string: str) -> None:
    """
    Update database statistics and optimize query plans.

    Args:
        connection_string: Database connection string
    """
    try:
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            # Update statistics for the query planner
            connection.execute(text("ANALYZE;"))

            # Set work_mem for complex sorts and joins
            connection.execute(text("SET work_mem = '32MB';"))

            # Set maintenance_work_mem for index creation and vacuum
            connection.execute(text("SET maintenance_work_mem = '256MB';"))

            # Set effective_cache_size for query planning
            connection.execute(text("SET effective_cache_size = '4GB';"))

            connection.commit()
            logger.info("Query plans optimized successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error optimizing query plans: {e}")
        raise


def setup_timescaledb_compression(connection_string: str) -> None:
    """
    Configure TimescaleDB compression policies for historical data.

    Args:
        connection_string: Database connection string
    """
    try:
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            # Enable compression on market_data table
            connection.execute(
                text("""
                ALTER TABLE market_data SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'exchange, symbol',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
            """)
            )

            # Add compression policy (compress data older than 7 days)
            connection.execute(
                text("""
                SELECT add_compression_policy('market_data', INTERVAL '7 days');
            """)
            )

            connection.commit()
            logger.info("TimescaleDB compression configured successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error setting up TimescaleDB compression: {e}")
        raise


def optimize_bulk_operations(connection_string: str) -> None:
    """
    Configure database for efficient bulk operations.

    Args:
        connection_string: Database connection string
    """
    try:
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            # Disable synchronous commit for bulk operations
            connection.execute(text("SET synchronous_commit = OFF;"))

            # Increase checkpoint segments
            connection.execute(text("SET checkpoint_segments = 32;"))

            # Increase WAL buffers
            connection.execute(text("SET wal_buffers = '16MB';"))

            connection.commit()
            logger.info("Bulk operations optimized successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error optimizing bulk operations: {e}")
        raise


def create_optimized_query(query_type: str, **params) -> str:
    """
    Generate optimized SQL queries for common operations.

    Args:
        query_type: Type of query to generate
        **params: Parameters for the query

    Returns:
        Optimized SQL query string
    """
    if query_type == "market_data_range":
        return """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE exchange = :exchange 
        AND symbol = :symbol 
        AND timeframe = :timeframe
        AND timestamp BETWEEN :start_date AND :end_date
        ORDER BY timestamp
        """

    elif query_type == "latest_data":
        return """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE exchange = :exchange 
        AND symbol = :symbol 
        AND timeframe = :timeframe
        ORDER BY timestamp DESC
        LIMIT :limit
        """

    elif query_type == "aggregated_data":
        # Optimized time-bucket aggregation using TimescaleDB
        return """
        SELECT 
            time_bucket(:interval, timestamp) AS bucket,
            first(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, timestamp) AS close,
            sum(volume) AS volume
        FROM market_data
        WHERE exchange = :exchange 
        AND symbol = :symbol 
        AND timestamp BETWEEN :start_date AND :end_date
        GROUP BY bucket
        ORDER BY bucket
        """

    elif query_type == "validation_issues":
        return """
        SELECT issue_type, severity, count(*) as count
        FROM data_validation_issues
        WHERE exchange = :exchange 
        AND symbol = :symbol 
        AND timeframe = :timeframe
        AND detected_at > :since_date
        GROUP BY issue_type, severity
        ORDER BY count DESC
        """

    else:
        raise ValueError(f"Unknown query type: {query_type}")


# =====================================================================
# Caching Strategies
# =====================================================================


class CacheManager:
    """Manager for various caching strategies."""

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize cache manager.

        Args:
            redis_url: Redis connection URL for distributed caching
        """
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                logger.info("Redis cache initialized successfully")
            except redis.RedisError as e:
                logger.warning(
                    f"Redis connection failed, falling back to local cache: {e}"
                )
                self.redis_client = None

    def timed_lru_cache(self, seconds: int = 300, maxsize: int = 128):
        """
        Decorator for time-based LRU cache with expiration.

        Args:
            seconds: Cache expiration time in seconds
            maxsize: Maximum cache size

        Returns:
            Decorated function with timed LRU cache
        """

        def decorator(func):
            # Create cache with specified size
            func_cache = lru_cache(maxsize=maxsize)(func)

            # Track cache timestamps
            cache_timestamps = {}

            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create a cache key from function arguments
                key = str(args) + str(sorted(kwargs.items()))

                # Check if cache entry has expired
                now = time.time()
                if key in cache_timestamps and now - cache_timestamps[key] > seconds:
                    # Clear this specific entry from cache
                    func_cache.cache_clear()

                # Update timestamp and return result
                result = func_cache(*args, **kwargs)
                cache_timestamps[key] = now
                return result

            # Add clear method to wrapper
            wrapper.cache_clear = func_cache.cache_clear
            wrapper.cache_info = func_cache.cache_info

            return wrapper

        return decorator

    def redis_cache(self, prefix: str, expire_seconds: int = 300):
        """
        Decorator for Redis-based distributed caching.

        Args:
            prefix: Cache key prefix
            expire_seconds: Cache expiration time in seconds

        Returns:
            Decorated function with Redis caching
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.redis_client:
                    # Fall back to direct function call if Redis is unavailable
                    return func(*args, **kwargs)

                # Create a cache key from function arguments
                key = f"{prefix}:{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

                # Try to get from cache
                cached_result = self.redis_client.get(key)
                if cached_result:
                    try:
                        import pickle

                        return pickle.loads(cached_result)
                    except Exception as e:
                        logger.warning(f"Error deserializing cached result: {e}")

                # Execute function and cache result
                result = func(*args, **kwargs)
                try:
                    import pickle

                    self.redis_client.setex(key, expire_seconds, pickle.dumps(result))
                except Exception as e:
                    logger.warning(f"Error caching result: {e}")

                return result

            return wrapper

        return decorator

    def dataframe_cache(self, max_entries: int = 50):
        """
        Cache manager specifically for pandas DataFrames.

        Args:
            max_entries: Maximum number of DataFrames to cache

        Returns:
            Decorated function with DataFrame caching
        """
        cache = {}
        cache_order = []

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create a cache key from function arguments
                key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

                if key in cache:
                    # Move to end of cache_order (most recently used)
                    cache_order.remove(key)
                    cache_order.append(key)
                    return cache[
                        key
                    ].copy()  # Return a copy to prevent modification of cached data

                # Execute function
                result = func(*args, **kwargs)

                # Cache result if it's a DataFrame
                if isinstance(result, pd.DataFrame):
                    # Enforce cache size limit
                    if len(cache) >= max_entries:
                        # Remove least recently used item
                        oldest_key = cache_order.pop(0)
                        del cache[oldest_key]

                    # Add new item to cache
                    cache[key] = (
                        result.copy()
                    )  # Store a copy to prevent modification of cached data
                    cache_order.append(key)

                return result

            # Add method to clear cache
            def clear_cache():
                cache.clear()
                cache_order.clear()

            wrapper.clear_cache = clear_cache

            return wrapper

        return decorator


# =====================================================================
# Memory Usage Optimization
# =====================================================================


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage of pandas DataFrame by downcasting numeric types.

    Args:
        df: Input DataFrame

    Returns:
        Memory-optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")

    # Optimize numeric columns
    for col in df.select_dtypes(include=["int"]).columns:
        c_min = df[col].min()
        c_max = df[col].max()

        # Downcast based on min/max values
        if c_min >= 0:
            if c_max < 256:
                df[col] = df[col].astype("uint8")
            elif c_max < 65536:
                df[col] = df[col].astype("uint16")
            elif c_max < 4294967296:
                df[col] = df[col].astype("uint32")
            else:
                df[col] = df[col].astype("uint64")
        else:
            if c_min > -128 and c_max < 128:
                df[col] = df[col].astype("int8")
            elif c_min > -32768 and c_max < 32768:
                df[col] = df[col].astype("int16")
            elif c_min > -2147483648 and c_max < 2147483648:
                df[col] = df[col].astype("int32")
            else:
                df[col] = df[col].astype("int64")

    # Optimize float columns
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = df[col].astype("float32")

    # Optimize object columns (if they contain only strings)
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / len(df) < 0.5:  # If column has low cardinality
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"DataFrame memory usage after optimization: {end_mem:.2f} MB")
    logger.info(f"Memory reduced by {100 * (start_mem - end_mem) / start_mem:.2f}%")

    return df


def trigger_garbage_collection() -> Dict[str, float]:
    """
    Trigger garbage collection and return memory usage statistics.

    Returns:
        Dictionary with memory usage statistics
    """
    # Get memory usage before collection
    process = psutil.Process(os.getpid())
    before_mem = process.memory_info().rss / 1024**2

    # Trigger garbage collection
    collected = gc.collect()

    # Get memory usage after collection
    after_mem = process.memory_info().rss / 1024**2

    stats = {
        "before_mem_mb": before_mem,
        "after_mem_mb": after_mem,
        "freed_mb": before_mem - after_mem,
        "objects_collected": collected,
    }

    logger.info(
        f"Garbage collection: {collected} objects collected, {before_mem - after_mem:.2f} MB freed"
    )
    return stats


def setup_memory_monitoring(interval_seconds: int = 300) -> None:
    """
    Setup periodic memory monitoring and garbage collection.

    Args:
        interval_seconds: Interval between memory checks in seconds
    """
    import threading

    def monitor_memory():
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / 1024**2
        logger.info(f"Memory usage: {mem_usage:.2f} MB")

        # Trigger garbage collection if memory usage is high
        if mem_usage > 1000:  # 1 GB threshold
            logger.warning(
                f"High memory usage detected: {mem_usage:.2f} MB, triggering garbage collection"
            )
            trigger_garbage_collection()

        # Schedule next check
        threading.Timer(interval_seconds, monitor_memory).start()

    # Start monitoring
    monitor_memory()
    logger.info(f"Memory monitoring started with {interval_seconds}s interval")


def optimize_pandas_operations() -> None:
    """Configure pandas for optimized operations."""
    # Use numexpr for faster numerical operations
    try:
        import numexpr

        num_cores = psutil.cpu_count(logical=True)
        numexpr.set_num_threads(num_cores)
        logger.info(f"Numexpr configured to use {num_cores} threads")
    except ImportError:
        logger.warning("Numexpr not available, some pandas operations may be slower")

    # Use bottleneck for faster nan functions
    try:
        pd.set_option("use_bottleneck", True)
        logger.info("Pandas configured to use bottleneck")
    except:
        logger.warning("Failed to configure pandas to use bottleneck")

    # Use numba for faster computations if available
    try:
        import numba

        logger.info(f"Numba JIT compiler available: version {numba.__version__}")
    except ImportError:
        logger.warning("Numba not available, some computations may be slower")


# =====================================================================
# System Configuration Tuning
# =====================================================================


def configure_connection_pooling(
    engine, pool_size: int = 10, max_overflow: int = 20
) -> None:
    """
    Configure database connection pooling.

    Args:
        engine: SQLAlchemy engine
        pool_size: Base pool size
        max_overflow: Maximum number of connections to allow beyond pool_size
    """
    engine.dispose()

    # Create new engine with connection pooling
    new_engine = create_engine(
        engine.url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=30,
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_pre_ping=True,  # Check connection validity before using
    )

    # Replace engine
    engine.dispose()
    engine = new_engine

    logger.info(
        f"Connection pooling configured: pool_size={pool_size}, max_overflow={max_overflow}"
    )


def configure_worker_threads(num_workers: Optional[int] = None) -> int:
    """
    Configure optimal number of worker threads based on system resources.

    Args:
        num_workers: Number of workers to use, or None to auto-detect

    Returns:
        Configured number of worker threads
    """
    if num_workers is None:
        # Use CPU count - 1 for optimal performance, minimum 2
        cpu_count = psutil.cpu_count(logical=True)
        num_workers = max(2, cpu_count - 1)

    # Set environment variable for worker processes
    os.environ["NUM_WORKERS"] = str(num_workers)

    logger.info(f"Worker threads configured: {num_workers}")
    return num_workers


def configure_system_limits() -> None:
    """Configure system resource limits for optimal performance."""
    try:
        import resource

        # Increase file descriptor limit
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 4096), hard))

        # Increase max memory limit
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))

        logger.info("System resource limits configured")
    except (ImportError, ValueError) as e:
        logger.warning(f"Failed to configure system limits: {e}")


def optimize_system() -> Dict[str, Any]:
    """
    Apply all system optimizations and return configuration summary.

    Returns:
        Dictionary with optimization settings
    """
    # Get database connection string
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")
        database_url = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

    # Get Redis URL
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Apply optimizations
    create_database_indexes(database_url)
    optimize_query_plans(database_url)
    setup_timescaledb_compression(database_url)

    # Configure connection pooling
    engine = create_engine(database_url)
    configure_connection_pooling(engine)

    # Configure worker threads
    num_workers = configure_worker_threads()

    # Configure system limits
    configure_system_limits()

    # Configure pandas
    optimize_pandas_operations()

    # Setup memory monitoring
    setup_memory_monitoring()

    # Return configuration summary
    return {
        "database_optimized": True,
        "connection_pooling": {"pool_size": 10, "max_overflow": 20},
        "worker_threads": num_workers,
        "memory_monitoring": {"interval_seconds": 300},
        "pandas_optimized": True,
        "system_limits_configured": True,
        "timestamp": datetime.now(UTC).isoformat(),
    }


if __name__ == "__main__":
    # Apply all optimizations when run as script
    config = optimize_system()
    print("System optimization complete:")
    for key, value in config.items():
        print(f"  {key}: {value}")
