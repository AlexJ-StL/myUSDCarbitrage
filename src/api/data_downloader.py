"""Enhanced data downloader with retry logic for USDC arbitrage application."""

import logging
import time
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import ccxt
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .data_validation import AdvancedDataValidator
from .database import DBConnector

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("data_downloader.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Not allowing requests
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreaker:
    """Circuit breaker implementation for API calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0

    def allow_request(self) -> bool:
        """Check if request is allowed based on circuit state.

        Returns:
            bool: True if request is allowed, False otherwise
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit transitioning from OPEN to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def record_success(self) -> None:
        """Record successful API call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit transitioning from HALF_OPEN to CLOSED")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0

    def record_failure(self) -> None:
        """Record failed API call."""
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit transitioning from HALF_OPEN to OPEN")
            self.state = CircuitState.OPEN
            return

        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                logger.warning("Circuit transitioning from CLOSED to OPEN")
                self.state = CircuitState.OPEN


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, max_calls: int, time_period: int):
        """Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in time period
            time_period: Time period in seconds
        """
        self.max_calls = max_calls
        self.time_period = time_period
        self.calls = []

    def allow_request(self) -> bool:
        """Check if request is allowed based on rate limits.

        Returns:
            bool: True if request is allowed, False otherwise
        """
        current_time = time.time()
        # Remove calls outside the time window
        self.calls = [
            call_time
            for call_time in self.calls
            if current_time - call_time <= self.time_period
        ]

        # Check if we're under the limit
        if len(self.calls) < self.max_calls:
            return True
        return False

    def record_call(self) -> None:
        """Record an API call."""
        self.calls.append(time.time())

    def wait_time(self) -> float:
        """Calculate time to wait before next call is allowed.

        Returns:
            float: Time to wait in seconds
        """
        if self.allow_request():
            return 0

        current_time = time.time()
        oldest_call = min(self.calls)
        return max(0, self.time_period - (current_time - oldest_call))


class EnhancedDataDownloader:
    """Enhanced data downloader with retry logic and circuit breaker."""

    def __init__(self, db_connection_string: str):
        """Initialize data downloader.

        Args:
            db_connection_string: Database connection string
        """
        self.db = DBConnector(db_connection_string)
        self.validator = AdvancedDataValidator(db_connection_string)

        # Initialize exchanges
        self.exchanges = {
            "coinbase": self._init_exchange("coinbase"),
            "kraken": self._init_exchange("kraken"),
            "binance": self._init_exchange("binance"),
            "bitfinex": self._init_exchange("bitfinex"),
            "bitstamp": self._init_exchange("bitstamp"),
        }

        # Circuit breakers for each exchange
        self.circuit_breakers = {
            exchange_id: CircuitBreaker() for exchange_id in self.exchanges
        }

        # Rate limiters for each exchange (values based on typical exchange limits)
        self.rate_limiters = {
            "coinbase": RateLimiter(max_calls=3, time_period=1),  # 3 calls per second
            "kraken": RateLimiter(max_calls=1, time_period=3),  # 1 call per 3 seconds
            "binance": RateLimiter(max_calls=10, time_period=1),  # 10 calls per second
            "bitfinex": RateLimiter(max_calls=1, time_period=6),  # 1 call per 6 seconds
            "bitstamp": RateLimiter(
                max_calls=8, time_period=10
            ),  # 8 calls per 10 seconds
        }

    def _init_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Initialize exchange with appropriate settings.

        Args:
            exchange_id: Exchange identifier

        Returns:
            ccxt.Exchange: Initialized exchange
        """
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                "enableRateLimit": True,  # Enable CCXT's built-in rate limiting
                "timeout": 30000,  # 30 seconds timeout
            })
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize exchange {exchange_id}: {e}")
            return None

    @retry(
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    def _fetch_ohlcv_with_retry(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int = 1000,
    ) -> List[List[float]]:
        """Fetch OHLCV data with retry logic.

        Args:
            exchange_id: Exchange identifier
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h', '1d')
            since: Start timestamp in milliseconds
            limit: Maximum number of records to fetch

        Returns:
            List[List[float]]: OHLCV data
        """
        exchange = self.exchanges.get(exchange_id)
        circuit_breaker = self.circuit_breakers.get(exchange_id)
        rate_limiter = self.rate_limiters.get(exchange_id)

        if not exchange:
            logger.error(f"Exchange {exchange_id} not initialized")
            return []

        if not circuit_breaker.allow_request():
            logger.warning(f"Circuit breaker open for {exchange_id}, skipping request")
            raise ccxt.ExchangeError(f"Circuit breaker open for {exchange_id}")

        # Check rate limiter
        wait_seconds = rate_limiter.wait_time()
        if wait_seconds > 0:
            logger.info(
                f"Rate limit reached for {exchange_id}, waiting {wait_seconds:.2f} seconds"
            )
            time.sleep(wait_seconds)

        try:
            # Record the API call
            rate_limiter.record_call()

            # Make the API call
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=limit)

            # Record success
            circuit_breaker.record_success()

            return ohlcv
        except Exception as e:
            # Record failure
            circuit_breaker.record_failure()

            logger.error(f"Error fetching data from {exchange_id}: {e}")
            raise

    def fetch_incremental_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """Fetch incremental data from exchange.

        Args:
            exchange_id: Exchange identifier
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h', '1d')
            start_date: Start date (defaults to last stored timestamp + 1 timeframe)
            end_date: End date (defaults to current time)
            batch_size: Number of records to fetch per request

        Returns:
            pd.DataFrame: OHLCV data as DataFrame
        """
        exchange = self.exchanges.get(exchange_id)
        if not exchange:
            logger.error(f"Exchange {exchange_id} not initialized")
            return pd.DataFrame()

        # Get timeframe duration in milliseconds
        if timeframe not in exchange.timeframes:
            logger.error(f"Timeframe {timeframe} not supported by {exchange_id}")
            return pd.DataFrame()

        timeframe_duration = exchange.timeframes[timeframe] * 1000

        # Determine start date (use last stored timestamp + 1 timeframe if not provided)
        if start_date is None:
            last_timestamp = self.db.get_last_timestamp(exchange_id, symbol, timeframe)
            if last_timestamp:
                # Add one timeframe to avoid duplicates
                start_date = datetime.fromtimestamp(
                    (last_timestamp + timeframe_duration) / 1000, UTC
                )
            else:
                # Default to 30 days ago if no data exists
                start_date = datetime.now(UTC) - timedelta(days=30)

        # Determine end date (use current time if not provided)
        if end_date is None:
            end_date = datetime.now(UTC)

        # Convert dates to timestamps
        since_ts = int(start_date.timestamp() * 1000)
        until_ts = int(end_date.timestamp() * 1000)

        logger.info(
            f"Fetching incremental data for {exchange_id}/{symbol}/{timeframe} "
            f"from {start_date} to {end_date}"
        )

        all_ohlcv = []
        current_since = since_ts

        while current_since < until_ts:
            try:
                logger.info(
                    f"Fetching batch from {datetime.fromtimestamp(current_since / 1000, UTC)}"
                )

                ohlcv = self._fetch_ohlcv_with_retry(
                    exchange_id, symbol, timeframe, current_since, limit=batch_size
                )

                if not ohlcv:
                    logger.info("No more data available")
                    break

                all_ohlcv.extend(ohlcv)

                # Update since for next batch (add 1ms to avoid duplicates)
                current_since = ohlcv[-1][0] + 1

                if ohlcv[-1][0] >= until_ts:
                    logger.info("Reached end date")
                    break

            except Exception as e:
                logger.error(f"Failed to fetch batch: {e}")
                # Continue with next batch after error
                current_since += timeframe_duration * batch_size

        # Convert to DataFrame
        if not all_ohlcv:
            logger.warning(f"No data fetched for {exchange_id}/{symbol}/{timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        return df

    def download_and_store_data(
        self,
        exchange_ids: List[str],
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Download and store data from multiple exchanges.

        Args:
            exchange_ids: List of exchange identifiers
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h', '1d')
            start_date: Start date (defaults to last stored timestamp + 1 timeframe)
            end_date: End date (defaults to current time)

        Returns:
            Dict[str, Any]: Summary of download results
        """
        results = {}

        for exchange_id in exchange_ids:
            if exchange_id not in self.exchanges:
                logger.warning(f"Exchange {exchange_id} not supported, skipping")
                results[exchange_id] = {
                    "status": "skipped",
                    "reason": "Exchange not supported",
                }
                continue

            try:
                # Fetch incremental data
                df = self.fetch_incremental_data(
                    exchange_id, symbol, timeframe, start_date, end_date
                )

                if df.empty:
                    logger.warning(
                        f"No new data for {exchange_id}/{symbol}/{timeframe}"
                    )
                    results[exchange_id] = {"status": "no_data", "records": 0}
                    continue

                # Validate data
                validation_results, quality_score = (
                    self.validator.comprehensive_validation(
                        exchange_id, symbol, timeframe
                    )
                )

                # Check for critical validation issues
                critical_issues = [
                    result
                    for result in validation_results
                    if result.severity.value in ["critical", "error"]
                ]

                if critical_issues:
                    logger.error(
                        f"Critical validation issues for {exchange_id}/{symbol}/{timeframe}: "
                        f"{[issue.message for issue in critical_issues]}"
                    )
                    results[exchange_id] = {
                        "status": "validation_failed",
                        "issues": [issue.message for issue in critical_issues],
                        "records": len(df),
                    }
                    continue

                # Convert DataFrame to OHLCV format for database
                ohlcv_data = []
                for _, row in df.iterrows():
                    ohlcv_data.append([
                        int(row["timestamp"].timestamp() * 1000),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                    ])

                # Store data in database
                stored_count = self.db.insert_data(
                    exchange_id, symbol, timeframe, ohlcv_data
                )

                results[exchange_id] = {
                    "status": "success",
                    "records": stored_count,
                    "quality_score": quality_score.overall_score,
                }

                logger.info(
                    f"Successfully stored {stored_count} records for "
                    f"{exchange_id}/{symbol}/{timeframe}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to download and store data for "
                    f"{exchange_id}/{symbol}/{timeframe}: {e}"
                )
                results[exchange_id] = {"status": "error", "error": str(e)}

        return results

    def resolve_conflicts(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        conflict_resolution: str = "newer",
    ) -> int:
        """Resolve conflicts in stored data.

        Args:
            exchange_id: Exchange identifier
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h', '1d')
            conflict_resolution: Conflict resolution strategy ('newer', 'older', 'average')

        Returns:
            int: Number of conflicts resolved
        """
        try:
            # Get duplicate timestamps
            duplicates = self.db.get_duplicate_timestamps(
                exchange_id, symbol, timeframe
            )

            if not duplicates:
                logger.info(
                    f"No conflicts found for {exchange_id}/{symbol}/{timeframe}"
                )
                return 0

            resolved_count = 0

            for timestamp in duplicates:
                # Get all records with this timestamp
                records = self.db.get_records_by_timestamp(
                    exchange_id, symbol, timeframe, timestamp
                )

                if len(records) <= 1:
                    continue

                # Apply conflict resolution strategy
                if conflict_resolution == "newer":
                    # Keep the newest record (by insertion time)
                    keep_record = max(records, key=lambda r: r["inserted_at"])

                elif conflict_resolution == "older":
                    # Keep the oldest record (by insertion time)
                    keep_record = min(records, key=lambda r: r["inserted_at"])

                elif conflict_resolution == "average":
                    # Average all values
                    avg_record = {
                        "open": sum(r["open"] for r in records) / len(records),
                        "high": max(r["high"] for r in records),
                        "low": min(r["low"] for r in records),
                        "close": sum(r["close"] for r in records) / len(records),
                        "volume": sum(r["volume"] for r in records) / len(records),
                        "timestamp": timestamp,
                    }
                    keep_record = avg_record
                else:
                    logger.error(
                        f"Unknown conflict resolution strategy: {conflict_resolution}"
                    )
                    continue

                # Delete all records with this timestamp
                self.db.delete_records_by_timestamp(
                    exchange_id, symbol, timeframe, timestamp
                )

                # Insert the record to keep
                self.db.insert_single_record(
                    exchange_id, symbol, timeframe, keep_record
                )

                resolved_count += 1

            logger.info(
                f"Resolved {resolved_count} conflicts for {exchange_id}/{symbol}/{timeframe} "
                f"using '{conflict_resolution}' strategy"
            )

            return resolved_count

        except Exception as e:
            logger.error(f"Failed to resolve conflicts: {e}")
            return 0

    def backfill_missing_data(
        self,
        primary_exchange: str,
        fallback_exchanges: List[str],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Backfill missing data using fallback exchanges.

        Args:
            primary_exchange: Primary exchange identifier
            fallback_exchanges: List of fallback exchange identifiers
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h', '1d')
            start_date: Start date
            end_date: End date

        Returns:
            Dict[str, Any]: Summary of backfill results
        """
        try:
            # Get missing timestamps in primary exchange
            missing_timestamps = self.db.get_missing_timestamps(
                primary_exchange, symbol, timeframe, start_date, end_date
            )

            if not missing_timestamps:
                logger.info(
                    f"No missing data for {primary_exchange}/{symbol}/{timeframe}"
                )
                return {"status": "no_gaps", "filled": 0}

            logger.info(
                f"Found {len(missing_timestamps)} missing timestamps for "
                f"{primary_exchange}/{symbol}/{timeframe}"
            )

            filled_count = 0

            # Try each fallback exchange
            for fallback_exchange in fallback_exchanges:
                if filled_count == len(missing_timestamps):
                    break

                remaining_timestamps = [
                    ts
                    for ts in missing_timestamps
                    if ts not in [r["timestamp"] for r in filled_records]
                ]

                if not remaining_timestamps:
                    break

                # Get available data from fallback exchange
                fallback_data = self.db.get_records_by_timestamps(
                    fallback_exchange, symbol, timeframe, remaining_timestamps
                )

                if not fallback_data:
                    logger.info(
                        f"No matching data found in {fallback_exchange} for backfill"
                    )
                    continue

                # Insert fallback data into primary exchange
                for record in fallback_data:
                    # Mark the record as backfilled
                    record["backfilled"] = True
                    record["source_exchange"] = fallback_exchange

                    self.db.insert_single_record(
                        primary_exchange, symbol, timeframe, record
                    )

                filled_count += len(fallback_data)

                logger.info(
                    f"Backfilled {len(fallback_data)} records from {fallback_exchange} "
                    f"to {primary_exchange}"
                )

            return {
                "status": "success",
                "total_gaps": len(missing_timestamps),
                "filled": filled_count,
                "remaining": len(missing_timestamps) - filled_count,
            }

        except Exception as e:
            logger.error(f"Failed to backfill missing data: {e}")
            return {"status": "error", "error": str(e)}


def main():
    """Main function to download and store data."""
    # Initialize downloader
    downloader = EnhancedDataDownloader("connection_string")

    # Define parameters
    exchanges = ["coinbase", "kraken", "binance", "bitfinex", "bitstamp"]
    symbol = "USDC/USD"
    timeframe = "1h"
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=7)  # Get last 7 days by default

    # Download and store data
    results = downloader.download_and_store_data(
        exchanges, symbol, timeframe, start_date, end_date
    )

    # Resolve any conflicts
    for exchange in exchanges:
        downloader.resolve_conflicts(exchange, symbol, timeframe, "newer")

    # Backfill missing data (using coinbase as primary)
    backfill_results = downloader.backfill_missing_data(
        "coinbase",
        ["kraken", "binance", "bitfinex", "bitstamp"],
        symbol,
        timeframe,
        start_date,
        end_date,
    )

    # Log results
    logger.info(f"Download results: {results}")
    logger.info(f"Backfill results: {backfill_results}")


if __name__ == "__main__":
    main()
