"""Tests for enhanced data download functionality."""

import os
import sys
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add src directory to Python path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from api.data_downloader import (
    CircuitBreaker,
    CircuitState,
    EnhancedDataDownloader,
    RateLimiter,
)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test initial circuit breaker state."""
        cb = CircuitBreaker()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.allow_request())

    def test_open_circuit(self):
        """Test opening the circuit after failures."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures
        for _ in range(3):
            self.assertTrue(cb.allow_request())
            cb.record_failure()

        # Circuit should be open now
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.allow_request())

    def test_half_open_circuit(self):
        """Test transition to half-open state after timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Open the circuit
        for _ in range(2):
            cb.record_failure()

        self.assertEqual(cb.state, CircuitState.OPEN)

        # Wait for recovery timeout
        import time

        time.sleep(0.2)

        # Should be half-open now
        self.assertTrue(cb.allow_request())
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)

    def test_circuit_recovery(self):
        """Test circuit recovery after successful request."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Open the circuit
        for _ in range(2):
            cb.record_failure()

        self.assertEqual(cb.state, CircuitState.OPEN)

        # Wait for recovery timeout
        import time

        time.sleep(0.2)

        # Should be half-open now
        self.assertTrue(cb.allow_request())
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)

        # Record success
        cb.record_success()

        # Should be closed now
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.allow_request())


class TestRateLimiter(unittest.TestCase):
    """Test rate limiter functionality."""

    def test_initial_state(self):
        """Test initial rate limiter state."""
        rl = RateLimiter(max_calls=3, time_period=1)
        self.assertTrue(rl.allow_request())
        self.assertEqual(rl.wait_time(), 0)

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        rl = RateLimiter(max_calls=2, time_period=1)

        # Make 2 calls (should be allowed)
        self.assertTrue(rl.allow_request())
        rl.record_call()
        self.assertTrue(rl.allow_request())
        rl.record_call()

        # Third call should be denied
        self.assertFalse(rl.allow_request())

        # Wait time should be > 0
        self.assertGreater(rl.wait_time(), 0)

        # Wait for time period to expire
        import time

        time.sleep(1.1)

        # Should be allowed again
        self.assertTrue(rl.allow_request())


@patch("api.data_downloader.DBConnector")
@patch("api.data_downloader.AdvancedDataValidator")
@patch("api.data_downloader.ccxt")
class TestEnhancedDataDownloader(unittest.TestCase):
    """Test enhanced data downloader functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_exchange = MagicMock()
        self.mock_exchange.timeframes = {"1h": 3600}

        # Sample OHLCV data
        self.sample_ohlcv = [
            [1625097600000, 1.0, 1.1, 0.9, 1.0, 100],
            [1625101200000, 1.0, 1.2, 0.8, 1.1, 200],
            [1625104800000, 1.1, 1.3, 1.0, 1.2, 150],
        ]

    def test_initialization(self, mock_ccxt, mock_validator, mock_db):
        """Test downloader initialization."""
        # Setup mock
        mock_ccxt.coinbase.return_value = self.mock_exchange
        mock_ccxt.kraken.return_value = self.mock_exchange
        mock_ccxt.binance.return_value = self.mock_exchange

        downloader = EnhancedDataDownloader("dummy_connection")

        # Check that exchanges were initialized
        self.assertIn("coinbase", downloader.exchanges)
        self.assertIn("kraken", downloader.exchanges)
        self.assertIn("binance", downloader.exchanges)

        # Check that circuit breakers were initialized
        self.assertIn("coinbase", downloader.circuit_breakers)
        self.assertIn("kraken", downloader.circuit_breakers)
        self.assertIn("binance", downloader.circuit_breakers)

        # Check that rate limiters were initialized
        self.assertIn("coinbase", downloader.rate_limiters)
        self.assertIn("kraken", downloader.rate_limiters)
        self.assertIn("binance", downloader.rate_limiters)

    def test_fetch_ohlcv_with_retry(self, mock_ccxt, mock_validator, mock_db):
        """Test fetching OHLCV data with retry logic."""
        # Setup mock
        mock_ccxt.coinbase.return_value = self.mock_exchange
        self.mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv

        downloader = EnhancedDataDownloader("dummy_connection")

        # Test successful fetch
        result = downloader._fetch_ohlcv_with_retry(
            "coinbase", "USDC/USD", "1h", 1625097600000
        )

        self.assertEqual(result, self.sample_ohlcv)
        self.mock_exchange.fetch_ohlcv.assert_called_once()

    def test_fetch_incremental_data(self, mock_ccxt, mock_validator, mock_db):
        """Test fetching incremental data."""
        # Setup mock
        mock_ccxt.coinbase.return_value = self.mock_exchange
        self.mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv

        # Mock database to return last timestamp
        mock_db.return_value.get_last_timestamp.return_value = (
            1625090400000  # Earlier timestamp
        )

        downloader = EnhancedDataDownloader("dummy_connection")

        # Test with explicit date range
        start_date = datetime(2023, 7, 1, tzinfo=UTC)
        end_date = datetime(2023, 7, 2, tzinfo=UTC)

        result = downloader.fetch_incremental_data(
            "coinbase", "USDC/USD", "1h", start_date, end_date
        )

        # Check that result is a DataFrame with expected columns
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("timestamp", result.columns)
        self.assertIn("open", result.columns)
        self.assertIn("high", result.columns)
        self.assertIn("low", result.columns)
        self.assertIn("close", result.columns)
        self.assertIn("volume", result.columns)

        # Check that fetch_ohlcv was called
        self.mock_exchange.fetch_ohlcv.assert_called()

    def test_download_and_store_data(self, mock_ccxt, mock_validator, mock_db):
        """Test downloading and storing data."""
        # Setup mocks
        mock_ccxt.coinbase.return_value = self.mock_exchange
        self.mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv

        # Mock validator to return no critical issues
        mock_validation_result = MagicMock()
        mock_validation_result.severity.value = "info"
        mock_quality_score = MagicMock()
        mock_quality_score.overall_score = 0.95
        mock_validator.return_value.comprehensive_validation.return_value = (
            [mock_validation_result],
            mock_quality_score,
        )

        # Mock database to insert data successfully
        mock_db.return_value.insert_data.return_value = len(self.sample_ohlcv)

        downloader = EnhancedDataDownloader("dummy_connection")

        # Test with single exchange
        result = downloader.download_and_store_data(
            ["coinbase"],
            "USDC/USD",
            "1h",
            datetime(2023, 7, 1, tzinfo=UTC),
            datetime(2023, 7, 2, tzinfo=UTC),
        )

        # Check result structure
        self.assertIn("coinbase", result)
        self.assertEqual(result["coinbase"]["status"], "success")
        self.assertEqual(result["coinbase"]["records"], len(self.sample_ohlcv))
        self.assertEqual(result["coinbase"]["quality_score"], 0.95)

        # Check that database insert was called
        mock_db.return_value.insert_data.assert_called_once()

    def test_resolve_conflicts(self, mock_ccxt, mock_validator, mock_db):
        """Test resolving data conflicts."""
        # Setup mock database
        mock_db.return_value.get_duplicate_timestamps.return_value = [1625097600000]
        mock_db.return_value.get_records_by_timestamp.return_value = [
            {
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
                "volume": 100,
                "timestamp": 1625097600000,
                "inserted_at": 1625097700000,
            },
            {
                "open": 1.0,
                "high": 1.2,
                "low": 0.8,
                "close": 1.1,
                "volume": 110,
                "timestamp": 1625097600000,
                "inserted_at": 1625097800000,
            },
        ]

        downloader = EnhancedDataDownloader("dummy_connection")

        # Test with "newer" strategy
        result = downloader.resolve_conflicts("coinbase", "USDC/USD", "1h", "newer")

        # Check that conflicts were resolved
        self.assertEqual(result, 1)
        mock_db.return_value.delete_records_by_timestamp.assert_called_once()
        mock_db.return_value.insert_single_record.assert_called_once()

    def test_backfill_missing_data(self, mock_ccxt, mock_validator, mock_db):
        """Test backfilling missing data."""
        # Setup mock database
        mock_db.return_value.get_missing_timestamps.return_value = [1625097600000]
        mock_db.return_value.get_records_by_timestamps.return_value = [
            {
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
                "volume": 100,
                "timestamp": 1625097600000,
            }
        ]

        downloader = EnhancedDataDownloader("dummy_connection")

        # Test backfilling
        result = downloader.backfill_missing_data(
            "coinbase",
            ["kraken", "binance"],
            "USDC/USD",
            "1h",
            datetime(2023, 7, 1, tzinfo=UTC),
            datetime(2023, 7, 2, tzinfo=UTC),
        )

        # Check result structure
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_gaps"], 1)
        self.assertEqual(result["filled"], 1)
        self.assertEqual(result["remaining"], 0)

        # Check that database methods were called
        mock_db.return_value.get_missing_timestamps.assert_called_once()
        mock_db.return_value.get_records_by_timestamps.assert_called_once()


@patch("api.data_downloader.EnhancedDataDownloader")
def test_main_function(mock_downloader):
    """Test the main function."""
    # Import the main function
    from download_usdc_data import main

    # Setup mock
    mock_instance = MagicMock()
    mock_downloader.return_value = mock_instance
    mock_instance.download_and_store_data.return_value = {
        "coinbase": {"status": "success", "records": 24},
        "kraken": {"status": "success", "records": 24},
        "binance": {"status": "success", "records": 24},
    }

    # Mock sys.argv to simulate command line arguments
    with patch("sys.argv", ["download_usdc_data.py", "--days", "1"]):
        # Run the main function
        main()

        # Check that downloader methods were called
        mock_instance.download_and_store_data.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
