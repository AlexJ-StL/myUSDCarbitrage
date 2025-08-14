"""
Performance and Load Testing Module

This module contains tests for performance and load testing of the application,
including concurrent backtest execution, data ingestion under high volume,
strategy execution speed benchmarking, and memory/resource usage profiling.
"""

import pytest
import time
import threading
import multiprocessing
import psutil
import os
import concurrent.futures
from unittest.mock import MagicMock, patch
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import application modules
# These imports will be adjusted based on the actual application structure
try:
    from src.strategies.base import BaseStrategy
    from src.data.downloader import DataDownloader
    from src.api.routers.backtest import run_backtest
except ImportError:
    logger.warning("Some imports failed. Tests will use mocks instead.")


class TestConcurrentBacktestExecution:
    """Test class for concurrent backtest execution scenarios."""

    @pytest.fixture
    def mock_backtest_engine(self):
        """Create a mock backtest engine for testing."""
        mock_engine = MagicMock()
        mock_engine.run_backtest.side_effect = lambda *args, **kwargs: {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.05,
            "trade_count": 50,
            "execution_time": 0.5,  # seconds
        }
        return mock_engine

    @pytest.mark.performance
    def test_concurrent_backtest_throughput(self, mock_backtest_engine):
        """
        Test the system's ability to handle multiple concurrent backtest requests.

        This test simulates multiple users requesting backtests simultaneously
        and measures the throughput and response times.
        """
        # Test parameters
        num_concurrent_requests = 10
        num_strategies = 5

        # Track execution times
        execution_times = []

        def run_single_backtest(strategy_id):
            start_time = time.time()
            result = mock_backtest_engine.run_backtest(
                strategy_id=strategy_id,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                initial_capital=10000.0,
            )
            end_time = time.time()
            execution_times.append(end_time - start_time)
            return result

        # Run concurrent backtests using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_concurrent_requests
        ) as executor:
            # Submit tasks
            futures = [
                executor.submit(run_single_backtest, strategy_id % num_strategies + 1)
                for strategy_id in range(num_concurrent_requests)
            ]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

            # Get results
            results = [future.result() for future in futures]

        # Calculate metrics
        total_time = max(execution_times)
        avg_time = sum(execution_times) / len(execution_times)
        throughput = num_concurrent_requests / total_time

        # Log results
        logger.info(f"Concurrent Backtest Test Results:")
        logger.info(f"Total requests: {num_concurrent_requests}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average response time: {avg_time:.2f} seconds")
        logger.info(f"Throughput: {throughput:.2f} requests/second")

        # Assert acceptable performance
        # These thresholds should be adjusted based on system requirements
        assert throughput >= 1.0, f"Throughput below threshold: {throughput:.2f} req/s"
        assert avg_time <= 2.0, f"Average response time too high: {avg_time:.2f}s"

    @pytest.mark.performance
    def test_backtest_scaling_performance(self, mock_backtest_engine):
        """
        Test how backtest performance scales with increasing concurrent load.

        This test measures how response times and throughput change as the
        number of concurrent requests increases.
        """
        # Test with different concurrency levels
        concurrency_levels = [1, 2, 5, 10, 20]
        results = {}

        for concurrency in concurrency_levels:
            start_time = time.time()
            execution_times = []

            def run_single_backtest():
                local_start = time.time()
                result = mock_backtest_engine.run_backtest(
                    strategy_id=1,
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now(),
                    initial_capital=10000.0,
                )
                local_end = time.time()
                execution_times.append(local_end - local_start)
                return result

            # Run concurrent backtests
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency
            ) as executor:
                futures = [
                    executor.submit(run_single_backtest) for _ in range(concurrency)
                ]
                concurrent.futures.wait(futures)

            end_time = time.time()
            total_time = end_time - start_time
            avg_time = sum(execution_times) / len(execution_times)
            throughput = concurrency / total_time

            results[concurrency] = {
                "total_time": total_time,
                "avg_response_time": avg_time,
                "throughput": throughput,
            }

            # Log results for this concurrency level
            logger.info(f"Concurrency level: {concurrency}")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Avg response time: {avg_time:.2f}s")
            logger.info(f"Throughput: {throughput:.2f} req/s")

        # Verify scaling behavior
        # As concurrency increases, throughput should increase (or at least not decrease significantly)
        # until system resources are saturated
        for i in range(1, len(concurrency_levels) - 1):
            current = results[concurrency_levels[i]]["throughput"]
            next_level = results[concurrency_levels[i + 1]]["throughput"]

            # Allow for some degradation at higher concurrency levels
            if concurrency_levels[i + 1] <= 10:
                assert next_level >= current * 0.8, (
                    f"Significant throughput degradation at concurrency {concurrency_levels[i + 1]}"
                )


class TestDataIngestionUnderLoad:
    """Test class for data ingestion pipeline under high volume."""

    @pytest.fixture
    def mock_data_downloader(self):
        """Create a mock data downloader for testing."""
        mock_downloader = MagicMock()
        mock_downloader.download_historical_data.side_effect = lambda *args, **kwargs: {
            "success": True,
            "records_count": 1000,
            "execution_time": 0.2,  # seconds
        }
        return mock_downloader

    @pytest.fixture
    def mock_data_validator(self):
        """Create a mock data validator for testing."""
        mock_validator = MagicMock()
        mock_validator.validate_data.side_effect = lambda data: {
            "valid_records": len(data) * 0.98,  # 98% valid
            "invalid_records": len(data) * 0.02,
            "execution_time": 0.1,  # seconds
        }
        return mock_validator

    @pytest.mark.performance
    def test_high_volume_data_ingestion(
        self, mock_data_downloader, mock_data_validator
    ):
        """
        Test the data ingestion pipeline's performance under high data volume.

        This test simulates downloading and processing large volumes of market data
        and measures throughput and processing times.
        """
        # Test parameters
        exchanges = ["binance", "coinbase", "kraken", "bitfinex", "bitstamp"]
        symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT"]
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        days_of_data = 30

        # Calculate expected data points
        # For each exchange, symbol, timeframe combination:
        # 1m: 1440 points per day
        # 5m: 288 points per day
        # 15m: 96 points per day
        # 1h: 24 points per day
        # 4h: 6 points per day
        # 1d: 1 point per day
        timeframe_points = {
            "1m": 1440,
            "5m": 288,
            "15m": 96,
            "1h": 24,
            "4h": 6,
            "1d": 1,
        }

        total_data_points = sum(
            days_of_data * timeframe_points[tf] * len(exchanges) * len(symbols)
            for tf in timeframes
        )

        logger.info(
            f"Testing data ingestion with approximately {total_data_points:,} data points"
        )

        # Simulate data ingestion process
        start_time = time.time()

        # Track metrics
        download_times = []
        validation_times = []
        storage_times = []  # Simulated

        # Process each exchange, symbol, timeframe combination
        for exchange in exchanges:
            for symbol in symbols:
                for timeframe in timeframes:
                    # Simulate download
                    download_start = time.time()
                    download_result = mock_data_downloader.download_historical_data(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=datetime.now() - timedelta(days=days_of_data),
                        end_date=datetime.now(),
                    )
                    download_end = time.time()
                    download_times.append(download_end - download_start)

                    # Simulate data points
                    data_points = [
                        {
                            "timestamp": datetime.now(),
                            "open": 100,
                            "high": 101,
                            "low": 99,
                            "close": 100.5,
                            "volume": 1000,
                        }
                        for _ in range(days_of_data * timeframe_points[timeframe])
                    ]

                    # Simulate validation
                    validation_start = time.time()
                    validation_result = mock_data_validator.validate_data(data_points)
                    validation_end = time.time()
                    validation_times.append(validation_end - validation_start)

                    # Simulate storage (just timing)
                    storage_start = time.time()
                    time.sleep(0.01)  # Simulate DB write time
                    storage_end = time.time()
                    storage_times.append(storage_end - storage_start)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate metrics
        avg_download_time = sum(download_times) / len(download_times)
        avg_validation_time = sum(validation_times) / len(validation_times)
        avg_storage_time = sum(storage_times) / len(storage_times)
        throughput = total_data_points / total_time

        # Log results
        logger.info(f"Data Ingestion Test Results:")
        logger.info(f"Total data points: {total_data_points:,}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average download time: {avg_download_time:.4f} seconds")
        logger.info(f"Average validation time: {avg_validation_time:.4f} seconds")
        logger.info(f"Average storage time: {avg_storage_time:.4f} seconds")
        logger.info(f"Throughput: {throughput:.2f} data points/second")

        # Assert acceptable performance
        assert throughput >= 1000, (
            f"Data ingestion throughput below threshold: {throughput:.2f} points/s"
        )

    @pytest.mark.performance
    def test_data_ingestion_error_resilience(self, mock_data_downloader):
        """
        Test the data ingestion pipeline's resilience under error conditions.

        This test simulates various error scenarios during data ingestion and
        measures the system's ability to handle errors gracefully without
        significant performance degradation.
        """
        # Configure mock to simulate errors
        error_rate = 0.2  # 20% of requests will fail

        def download_with_errors(*args, **kwargs):
            if random.random() < error_rate:
                raise Exception("Simulated API error")
            return {"success": True, "records_count": 1000, "execution_time": 0.2}

        mock_data_downloader.download_historical_data.side_effect = download_with_errors

        # Test parameters
        exchanges = ["binance", "coinbase", "kraken"]
        symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        timeframes = ["1m", "5m", "15m", "1h"]

        # Track metrics
        success_count = 0
        error_count = 0
        retry_count = 0

        start_time = time.time()

        # Process each combination with retry logic
        for exchange in exchanges:
            for symbol in symbols:
                for timeframe in timeframes:
                    max_retries = 3
                    current_retry = 0

                    while current_retry <= max_retries:
                        try:
                            result = mock_data_downloader.download_historical_data(
                                exchange=exchange,
                                symbol=symbol,
                                timeframe=timeframe,
                                start_date=datetime.now() - timedelta(days=7),
                                end_date=datetime.now(),
                            )
                            success_count += 1
                            break
                        except Exception as e:
                            current_retry += 1
                            retry_count += 1
                            if current_retry > max_retries:
                                error_count += 1
                                logger.warning(
                                    f"Failed after {max_retries} retries: {exchange} {symbol} {timeframe}"
                                )
                            time.sleep(0.1)  # Backoff

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate metrics
        total_requests = len(exchanges) * len(symbols) * len(timeframes)
        success_rate = success_count / total_requests

        # Log results
        logger.info(f"Data Ingestion Error Resilience Test Results:")
        logger.info(f"Total requests: {total_requests}")
        logger.info(f"Successful requests: {success_count}")
        logger.info(f"Failed requests: {error_count}")
        logger.info(f"Retry attempts: {retry_count}")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Total time: {total_time:.2f} seconds")

        # Assert acceptable performance
        assert success_rate >= 0.95, f"Success rate below threshold: {success_rate:.2%}"


import random
import gc  # For garbage collection during memory tests


class TestStrategyExecutionBenchmarking:
    """Test class for benchmarking strategy execution speed."""

    @pytest.fixture
    def mock_strategy_factory(self):
        """Create a mock strategy factory for testing."""
        mock_factory = MagicMock()

        # Create different strategy types with different execution speeds
        strategies = {
            "threshold": MagicMock(execution_time=0.001),  # Fast
            "ml_based": MagicMock(execution_time=0.01),  # Medium
            "statistical": MagicMock(execution_time=0.005),  # Medium-fast
        }

        def get_strategy(strategy_type):
            strategy = strategies.get(strategy_type, strategies["threshold"])

            def execute(data_point):
                time.sleep(strategy.execution_time)
                return {
                    "action": random.choice(["buy", "sell", "hold"]),
                    "confidence": random.random(),
                }

            strategy.execute = execute
            return strategy

        mock_factory.get_strategy = get_strategy
        return mock_factory

    @pytest.mark.performance
    def test_strategy_execution_speed(self, mock_strategy_factory):
        """
        Benchmark the execution speed of different strategy types.

        This test measures the execution time for different strategy types
        and compares their performance.
        """
        # Test parameters
        strategy_types = ["threshold", "ml_based", "statistical"]
        data_points_count = 10000  # Number of data points to process

        # Generate test data
        test_data = [
            {
                "timestamp": datetime.now() - timedelta(minutes=i),
                "open": 100 + random.random(),
                "high": 101 + random.random(),
                "low": 99 + random.random(),
                "close": 100.5 + random.random(),
                "volume": 1000 + random.random() * 500,
            }
            for i in range(data_points_count)
        ]

        results = {}

        # Test each strategy type
        for strategy_type in strategy_types:
            strategy = mock_strategy_factory.get_strategy(strategy_type)

            # Warm-up run
            for _ in range(100):
                strategy.execute(test_data[0])

            # Benchmark run
            start_time = time.time()

            for data_point in test_data:
                strategy.execute(data_point)

            end_time = time.time()
            execution_time = end_time - start_time

            # Calculate metrics
            operations_per_second = data_points_count / execution_time
            avg_time_per_operation = execution_time / data_points_count

            results[strategy_type] = {
                "total_time": execution_time,
                "operations_per_second": operations_per_second,
                "avg_time_per_operation": avg_time_per_operation,
            }

            # Log results
            logger.info(f"Strategy Type: {strategy_type}")
            logger.info(f"Total execution time: {execution_time:.2f} seconds")
            logger.info(f"Operations per second: {operations_per_second:.2f}")
            logger.info(
                f"Average time per operation: {avg_time_per_operation * 1000:.4f} ms"
            )

        # Compare strategies
        fastest_strategy = min(results.items(), key=lambda x: x[1]["total_time"])
        slowest_strategy = max(results.items(), key=lambda x: x[1]["total_time"])

        logger.info(
            f"Fastest strategy: {fastest_strategy[0]} - {fastest_strategy[1]['operations_per_second']:.2f} ops/sec"
        )
        logger.info(
            f"Slowest strategy: {slowest_strategy[0]} - {slowest_strategy[1]['operations_per_second']:.2f} ops/sec"
        )

        # Assert acceptable performance
        for strategy_type, metrics in results.items():
            assert metrics["operations_per_second"] >= 100, (
                f"Strategy {strategy_type} performance below threshold: {metrics['operations_per_second']:.2f} ops/sec"
            )

    @pytest.mark.performance
    def test_strategy_scaling_with_data_size(self, mock_strategy_factory):
        """
        Test how strategy execution time scales with increasing data size.

        This test measures execution time for different data sizes to identify
        any non-linear scaling issues.
        """
        # Test parameters
        strategy_type = "threshold"  # Use the fastest strategy for this test
        data_sizes = [100, 1000, 10000, 100000]

        strategy = mock_strategy_factory.get_strategy(strategy_type)
        results = {}

        for size in data_sizes:
            # Generate test data
            test_data = [
                {
                    "timestamp": datetime.now() - timedelta(minutes=i),
                    "open": 100 + random.random(),
                    "high": 101 + random.random(),
                    "low": 99 + random.random(),
                    "close": 100.5 + random.random(),
                    "volume": 1000 + random.random() * 500,
                }
                for i in range(size)
            ]

            # Benchmark run
            start_time = time.time()

            for data_point in test_data:
                strategy.execute(data_point)

            end_time = time.time()
            execution_time = end_time - start_time

            # Calculate metrics
            operations_per_second = size / execution_time

            results[size] = {
                "total_time": execution_time,
                "operations_per_second": operations_per_second,
            }

            # Log results
            logger.info(f"Data Size: {size}")
            logger.info(f"Total execution time: {execution_time:.2f} seconds")
            logger.info(f"Operations per second: {operations_per_second:.2f}")

        # Check for linear scaling
        # If scaling is linear, the operations per second should remain relatively constant
        base_ops = results[data_sizes[0]]["operations_per_second"]
        for size in data_sizes[1:]:
            current_ops = results[size]["operations_per_second"]
            ratio = current_ops / base_ops

            logger.info(f"Scaling ratio for size {size}: {ratio:.2f}x")

            # Allow for some degradation at larger sizes, but it shouldn't be dramatic
            # For truly linear algorithms, this should be close to 1.0
            assert ratio >= 0.7, (
                f"Significant performance degradation at size {size}: {ratio:.2f}x"
            )


class TestMemoryResourceProfiling:
    """Test class for memory and resource usage profiling."""

    def get_process_memory(self):
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB

    def get_cpu_usage(self):
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

    @pytest.mark.performance
    def test_backtest_memory_usage(self):
        """
        Test memory usage during backtest execution.

        This test profiles memory consumption during backtest execution
        with different data sizes and configurations.
        """
        # Skip test if psutil is not available
        if not hasattr(psutil, "Process"):
            pytest.skip("psutil not available for memory profiling")

        # Test parameters
        data_sizes = [1000, 10000, 50000]

        # Create a simple mock backtest function
        def run_mock_backtest(data_size):
            # Allocate memory to simulate data loading
            data = [
                {"timestamp": datetime.now(), "price": i, "volume": i * 10}
                for i in range(data_size)
            ]

            # Simulate processing
            results = []
            for i in range(len(data) - 1):
                # Simple calculation to simulate strategy execution
                if data[i]["price"] < data[i + 1]["price"]:
                    results.append({
                        "action": "buy",
                        "profit": data[i + 1]["price"] - data[i]["price"],
                    })
                else:
                    results.append({
                        "action": "sell",
                        "profit": data[i]["price"] - data[i + 1]["price"],
                    })

            return results

        results = {}

        for size in data_sizes:
            # Measure baseline memory
            gc.collect()  # Force garbage collection
            baseline_memory = self.get_process_memory()

            # Run backtest and measure peak memory
            start_time = time.time()
            start_memory = self.get_process_memory()

            backtest_results = run_mock_backtest(size)

            end_memory = self.get_process_memory()
            end_time = time.time()

            # Calculate metrics
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            memory_per_data_point = memory_used / size if size > 0 else 0

            results[size] = {
                "execution_time": execution_time,
                "baseline_memory": baseline_memory,
                "peak_memory": end_memory,
                "memory_used": memory_used,
                "memory_per_data_point": memory_per_data_point,
            }

            # Log results
            logger.info(f"Data Size: {size}")
            logger.info(f"Execution time: {execution_time:.2f} seconds")
            logger.info(f"Baseline memory: {baseline_memory:.2f} MB")
            logger.info(f"Peak memory: {end_memory:.2f} MB")
            logger.info(f"Memory used: {memory_used:.2f} MB")
            logger.info(f"Memory per data point: {memory_per_data_point * 1000:.2f} KB")

        # Check memory scaling
        # Memory usage should scale roughly linearly with data size
        if len(data_sizes) >= 2:
            size_ratio = data_sizes[-1] / data_sizes[0]
            memory_ratio = (
                results[data_sizes[-1]]["memory_used"]
                / results[data_sizes[0]]["memory_used"]
            )

            logger.info(f"Data size ratio (largest/smallest): {size_ratio:.2f}x")
            logger.info(f"Memory usage ratio: {memory_ratio:.2f}x")

            # Memory usage should grow more slowly than data size due to fixed overhead
            assert memory_ratio <= size_ratio * 1.5, (
                f"Memory usage scaling worse than expected: {memory_ratio:.2f}x vs {size_ratio:.2f}x"
            )

    @pytest.mark.performance
    def test_data_processing_resource_usage(self):
        """
        Test CPU and memory usage during data processing operations.

        This test profiles resource consumption during typical data processing
        operations like validation, transformation, and analysis.
        """
        # Skip test if psutil is not available
        if not hasattr(psutil, "Process"):
            pytest.skip("psutil not available for resource profiling")

        # Test parameters
        data_size = 50000

        # Generate test data
        test_data = [
            {
                "timestamp": datetime.now() - timedelta(minutes=i),
                "open": 100 + random.random(),
                "high": 101 + random.random(),
                "low": 99 + random.random(),
                "close": 100.5 + random.random(),
                "volume": 1000 + random.random() * 500,
            }
            for i in range(data_size)
        ]

        # Define processing operations
        def validate_data(data):
            """Simulate data validation."""
            valid_count = 0
            for item in data:
                if (
                    item["high"] >= item["open"]
                    and item["high"] >= item["close"]
                    and item["low"] <= item["open"]
                    and item["low"] <= item["close"]
                    and item["volume"] > 0
                ):
                    valid_count += 1
            return valid_count

        def transform_data(data):
            """Simulate data transformation."""
            result = []
            for i in range(len(data) - 1):
                result.append({
                    "timestamp": data[i]["timestamp"],
                    "price_change": data[i + 1]["close"] - data[i]["close"],
                    "volume_change": data[i + 1]["volume"] - data[i]["volume"],
                    "high_low_range": data[i]["high"] - data[i]["low"],
                })
            return result

        def analyze_data(data):
            """Simulate data analysis."""
            if not data:
                return {}

            price_changes = [item["price_change"] for item in data]
            volume_changes = [item["volume_change"] for item in data]

            return {
                "avg_price_change": sum(price_changes) / len(price_changes),
                "max_price_change": max(price_changes),
                "min_price_change": min(price_changes),
                "avg_volume_change": sum(volume_changes) / len(volume_changes),
            }

        # Profile each operation
        operations = [
            ("Validation", lambda: validate_data(test_data)),
            ("Transformation", lambda: transform_data(test_data)),
            ("Analysis", lambda: analyze_data(transform_data(test_data))),
        ]

        for name, operation in operations:
            # Measure baseline
            gc.collect()  # Force garbage collection
            baseline_memory = self.get_process_memory()
            baseline_cpu = self.get_cpu_usage()

            # Run operation and measure resources
            start_time = time.time()
            start_memory = self.get_process_memory()

            result = operation()

            # Measure peak CPU during operation
            peak_cpu = self.get_cpu_usage()
            end_memory = self.get_process_memory()
            end_time = time.time()

            # Calculate metrics
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            cpu_used = peak_cpu - baseline_cpu

            # Log results
            logger.info(f"Operation: {name}")
            logger.info(f"Execution time: {execution_time:.2f} seconds")
            logger.info(f"Memory used: {memory_used:.2f} MB")
            logger.info(f"CPU usage: {peak_cpu:.1f}%")

            # Assert acceptable resource usage
            # These thresholds should be adjusted based on system requirements
            assert execution_time < 10.0, (
                f"{name} execution time too high: {execution_time:.2f}s"
            )
            assert memory_used < 1000.0, (
                f"{name} memory usage too high: {memory_used:.2f} MB"
            )


if __name__ == "__main__":
    # This allows running the tests directly with more detailed output
    logging.basicConfig(level=logging.INFO)
    pytest.main(["-xvs", __file__])
