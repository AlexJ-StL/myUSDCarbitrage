"""
Unit tests for the on-demand report generator module.

This module contains tests for the on-demand report generator functionality,
including report generation for arbitrage opportunities and strategy performance.
"""

import os
import sys
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from sqlalchemy.orm import Session

# Add src directory to Python path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from reporting.on_demand_report_generator import (
    OnDemandReportGenerator,
    get_on_demand_report_generator,
)


class TestOnDemandReportGenerator(unittest.TestCase):
    """Test the on-demand report generator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock database session
        self.mock_db_session = MagicMock(spec=Session)

        # Create mock database connector
        self.mock_db_connector = MagicMock()

        # Create report generator instance
        self.report_generator = OnDemandReportGenerator(
            self.mock_db_session, self.mock_db_connector
        )

    def test_init(self):
        """Test initialization of on-demand report generator."""
        self.assertEqual(self.report_generator.db_session, self.mock_db_session)
        self.assertEqual(self.report_generator.db_connector, self.mock_db_connector)

    @patch("src.reporting.on_demand_report_generator.generate_arbitrage_report")
    def test_generate_arbitrage_opportunity_report(self, mock_generate_report):
        """Test generation of arbitrage opportunity report."""
        # Mock generate_arbitrage_report
        mock_generate_report.return_value = "<html>Test Arbitrage Report</html>"

        # Generate report
        start_time = datetime(2023, 1, 1, tzinfo=UTC)
        end_time = datetime(2023, 1, 2, tzinfo=UTC)
        report = self.report_generator.generate_arbitrage_opportunity_report(
            exchanges=["coinbase", "kraken"],
            symbol="USDC/USD",
            start_time=start_time,
            end_time=end_time,
            threshold=0.001,
            output_format="html",
        )

        # Check that generate_arbitrage_report was called with correct arguments
        mock_generate_report.assert_called_once_with(
            db=self.mock_db_session,
            start_time=start_time,
            end_time=end_time,
            exchanges=["coinbase", "kraken"],
            symbol="USDC/USD",
            threshold=0.001,
        )

        # Check that report was returned with correct structure
        self.assertEqual(report["content"], "<html>Test Arbitrage Report</html>")
        self.assertEqual(report["content_type"], "text/html")
        self.assertIn("generated_at", report)
        self.assertEqual(report["report_type"], "arbitrage_opportunity")
        self.assertIn("parameters", report)
        self.assertEqual(report["parameters"]["exchanges"], ["coinbase", "kraken"])
        self.assertEqual(report["parameters"]["symbol"], "USDC/USD")
        self.assertEqual(report["parameters"]["threshold"], 0.001)

    @patch("src.reporting.on_demand_report_generator.generate_arbitrage_report")
    def test_generate_arbitrage_opportunity_report_default_times(
        self, mock_generate_report
    ):
        """Test generation of arbitrage opportunity report with default times."""
        # Mock generate_arbitrage_report
        mock_generate_report.return_value = "<html>Test Arbitrage Report</html>"

        # Generate report with default times
        report = self.report_generator.generate_arbitrage_opportunity_report(
            exchanges=["coinbase", "kraken"],
            symbol="USDC/USD",
            threshold=0.001,
            output_format="html",
        )

        # Check that generate_arbitrage_report was called
        mock_generate_report.assert_called_once()

        # Check that report was returned with correct structure
        self.assertEqual(report["content"], "<html>Test Arbitrage Report</html>")
        self.assertEqual(report["content_type"], "text/html")

    @patch("src.reporting.on_demand_report_generator.generate_arbitrage_report")
    def test_generate_arbitrage_opportunity_report_json_format(
        self, mock_generate_report
    ):
        """Test generation of arbitrage opportunity report in JSON format."""
        # Mock generate_arbitrage_report
        mock_generate_report.return_value = "<html>Test Arbitrage Report</html>"

        # Generate report in JSON format
        report = self.report_generator.generate_arbitrage_opportunity_report(
            exchanges=["coinbase", "kraken"],
            symbol="USDC/USD",
            threshold=0.001,
            output_format="json",
        )

        # Check that report was returned with correct structure
        self.assertEqual(report["content_type"], "application/json")
        self.assertIsInstance(report["content"], dict)

    @patch("src.reporting.on_demand_report_generator.generate_arbitrage_report")
    def test_generate_arbitrage_opportunity_report_csv_format(
        self, mock_generate_report
    ):
        """Test generation of arbitrage opportunity report in CSV format."""
        # Mock generate_arbitrage_report
        mock_generate_report.return_value = "<html>Test Arbitrage Report</html>"

        # Generate report in CSV format
        report = self.report_generator.generate_arbitrage_opportunity_report(
            exchanges=["coinbase", "kraken"],
            symbol="USDC/USD",
            threshold=0.001,
            output_format="csv",
        )

        # Check that report was returned with correct structure
        self.assertEqual(report["content_type"], "text/csv")
        self.assertIsInstance(report["content"], str)

    @patch("src.reporting.on_demand_report_generator.generate_arbitrage_report")
    def test_generate_arbitrage_opportunity_report_error_handling(
        self, mock_generate_report
    ):
        """Test error handling in arbitrage opportunity report generation."""
        # Mock generate_arbitrage_report to raise an exception
        mock_generate_report.side_effect = ValueError("Test error")

        # Generate report
        report = self.report_generator.generate_arbitrage_opportunity_report(
            exchanges=["coinbase", "kraken"],
            symbol="USDC/USD",
            threshold=0.001,
            output_format="html",
        )

        # Check that error was handled correctly
        self.assertIn("error", report)
        self.assertEqual(report["error"], "Test error")
        self.assertIn("Error Generating Report", report["content"])
        self.assertEqual(report["content_type"], "text/html")

    @patch("src.reporting.on_demand_report_generator.generate_arbitrage_report")
    def test_generate_arbitrage_opportunity_report_validation(
        self, mock_generate_report
    ):
        """Test validation in arbitrage opportunity report generation."""
        # Test with invalid start/end times
        with self.assertRaises(ValueError):
            self.report_generator.generate_arbitrage_opportunity_report(
                exchanges=["coinbase", "kraken"],
                symbol="USDC/USD",
                start_time=datetime(2023, 1, 2, tzinfo=UTC),
                end_time=datetime(2023, 1, 1, tzinfo=UTC),
                threshold=0.001,
                output_format="html",
            )

        # Test with insufficient exchanges
        with self.assertRaises(ValueError):
            self.report_generator.generate_arbitrage_opportunity_report(
                exchanges=["coinbase"],
                symbol="USDC/USD",
                threshold=0.001,
                output_format="html",
            )

    @patch(
        "src.reporting.on_demand_report_generator.generate_strategy_performance_report"
    )
    def test_generate_strategy_performance_report_with_strategy_id(
        self, mock_generate_report
    ):
        """Test generation of strategy performance report with strategy ID."""
        # Mock generate_strategy_performance_report
        mock_generate_report.return_value = "<html>Test Strategy Report</html>"

        # Mock database queries
        mock_backtest_result = MagicMock()
        mock_backtest_result.strategy_id = 1
        mock_backtest_result.start_date = datetime(2023, 1, 1, tzinfo=UTC)
        mock_backtest_result.end_date = datetime(2023, 1, 31, tzinfo=UTC)
        mock_backtest_result.metrics = {"total_return": 0.15}
        mock_backtest_result.results = {"trades": []}

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.first.return_value = mock_backtest_result

        self.mock_db_session.query.return_value = mock_query

        # Mock strategy name query
        mock_strategy_query = MagicMock()
        mock_strategy_query.filter.return_value = mock_strategy_query
        mock_strategy_query.scalar.return_value = "Test Strategy"

        self.mock_db_session.query.return_value = mock_strategy_query

        # Generate report
        report = self.report_generator.generate_strategy_performance_report(
            strategy_id=1,
            output_format="html",
        )

        # Check that generate_strategy_performance_report was called with correct arguments
        mock_generate_report.assert_called_once()
        args = mock_generate_report.call_args[1]
        self.assertEqual(args["backtest_result"]["strategy_name"], "Test Strategy")
        self.assertEqual(args["benchmark_data"], None)

        # Check that report was returned with correct structure
        self.assertEqual(report["content"], "<html>Test Strategy Report</html>")
        self.assertEqual(report["content_type"], "text/html")
        self.assertIn("generated_at", report)
        self.assertEqual(report["report_type"], "strategy_performance")
        self.assertIn("parameters", report)
        self.assertEqual(report["parameters"]["strategy_id"], 1)

    @patch(
        "src.reporting.on_demand_report_generator.generate_strategy_performance_report"
    )
    def test_generate_strategy_performance_report_with_backtest_id(
        self, mock_generate_report
    ):
        """Test generation of strategy performance report with backtest ID."""
        # Mock generate_strategy_performance_report
        mock_generate_report.return_value = "<html>Test Strategy Report</html>"

        # Mock database queries
        mock_backtest_result = MagicMock()
        mock_backtest_result.strategy_id = 1
        mock_backtest_result.start_date = datetime(2023, 1, 1, tzinfo=UTC)
        mock_backtest_result.end_date = datetime(2023, 1, 31, tzinfo=UTC)
        mock_backtest_result.metrics = {"total_return": 0.15}
        mock_backtest_result.results = {"trades": []}

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_backtest_result

        self.mock_db_session.query.return_value = mock_query

        # Generate report
        report = self.report_generator.generate_strategy_performance_report(
            backtest_id=1,
            output_format="html",
        )

        # Check that generate_strategy_performance_report was called
        mock_generate_report.assert_called_once()

        # Check that report was returned with correct structure
        self.assertEqual(report["content"], "<html>Test Strategy Report</html>")
        self.assertEqual(report["content_type"], "text/html")
        self.assertIn("parameters", report)
        self.assertEqual(report["parameters"]["backtest_id"], 1)

    @patch(
        "src.reporting.on_demand_report_generator.generate_strategy_performance_report"
    )
    def test_generate_strategy_performance_report_with_benchmark(
        self, mock_generate_report
    ):
        """Test generation of strategy performance report with benchmark data."""
        # Mock generate_strategy_performance_report
        mock_generate_report.return_value = "<html>Test Strategy Report</html>"

        # Mock database queries
        mock_backtest_result = MagicMock()
        mock_backtest_result.strategy_id = 1
        mock_backtest_result.start_date = datetime(2023, 1, 1, tzinfo=UTC)
        mock_backtest_result.end_date = datetime(2023, 1, 31, tzinfo=UTC)
        mock_backtest_result.metrics = {"total_return": 0.15}
        mock_backtest_result.results = {"trades": []}

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_backtest_result

        self.mock_db_session.query.return_value = mock_query

        # Mock benchmark data
        mock_benchmark_df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=31, freq="1D"),
            "close": [16500 + i * 100 for i in range(31)],
        })
        self.mock_db_connector.get_ohlcv_data_range.return_value = mock_benchmark_df

        # Generate report with benchmark
        report = self.report_generator.generate_strategy_performance_report(
            backtest_id=1,
            include_benchmark=True,
            benchmark_symbol="BTC/USD",
            output_format="html",
        )

        # Check that generate_strategy_performance_report was called with benchmark data
        mock_generate_report.assert_called_once()
        args = mock_generate_report.call_args[1]
        self.assertIsNotNone(args["benchmark_data"])
        self.assertEqual(args["benchmark_data"]["name"], "BTC/USD")

        # Check that report was returned with correct structure
        self.assertEqual(report["content"], "<html>Test Strategy Report</html>")
        self.assertEqual(report["content_type"], "text/html")
        self.assertIn("parameters", report)
        self.assertEqual(report["parameters"]["include_benchmark"], True)
        self.assertEqual(report["parameters"]["benchmark_symbol"], "BTC/USD")

    @patch(
        "src.reporting.on_demand_report_generator.generate_strategy_performance_report"
    )
    def test_generate_strategy_performance_report_json_format(
        self, mock_generate_report
    ):
        """Test generation of strategy performance report in JSON format."""
        # Mock generate_strategy_performance_report
        mock_generate_report.return_value = "<html>Test Strategy Report</html>"

        # Mock database queries
        mock_backtest_result = MagicMock()
        mock_backtest_result.strategy_id = 1
        mock_backtest_result.start_date = datetime(2023, 1, 1, tzinfo=UTC)
        mock_backtest_result.end_date = datetime(2023, 1, 31, tzinfo=UTC)
        mock_backtest_result.metrics = {"total_return": 0.15}
        mock_backtest_result.results = {"trades": []}

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_backtest_result

        self.mock_db_session.query.return_value = mock_query

        # Generate report in JSON format
        report = self.report_generator.generate_strategy_performance_report(
            backtest_id=1,
            output_format="json",
        )

        # Check that report was returned with correct structure
        self.assertEqual(report["content_type"], "application/json")
        self.assertIsInstance(report["content"], dict)

    @patch(
        "src.reporting.on_demand_report_generator.generate_strategy_performance_report"
    )
    def test_generate_strategy_performance_report_error_handling(
        self, mock_generate_report
    ):
        """Test error handling in strategy performance report generation."""
        # Mock generate_strategy_performance_report to raise an exception
        mock_generate_report.side_effect = ValueError("Test error")

        # Mock database queries
        mock_backtest_result = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_backtest_result

        self.mock_db_session.query.return_value = mock_query

        # Generate report
        report = self.report_generator.generate_strategy_performance_report(
            backtest_id=1,
            output_format="html",
        )

        # Check that error was handled correctly
        self.assertIn("error", report)
        self.assertEqual(report["error"], "Test error")
        self.assertIn("Error Generating Report", report["content"])
        self.assertEqual(report["content_type"], "text/html")

    @patch(
        "src.reporting.on_demand_report_generator.generate_strategy_performance_report"
    )
    def test_generate_strategy_performance_report_validation(
        self, mock_generate_report
    ):
        """Test validation in strategy performance report generation."""
        # Test with neither strategy_id nor backtest_id
        with self.assertRaises(ValueError):
            self.report_generator.generate_strategy_performance_report(
                output_format="html",
            )

        # Test with backtest not found
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None

        self.mock_db_session.query.return_value = mock_query

        with self.assertRaises(ValueError):
            self.report_generator.generate_strategy_performance_report(
                backtest_id=999,
                output_format="html",
            )

    def test_factory_function(self):
        """Test factory function for creating on-demand report generator."""
        # Create mock database session and connector
        mock_db_session = MagicMock()
        mock_db_connector = MagicMock()

        # Call factory function
        generator = get_on_demand_report_generator(mock_db_session, mock_db_connector)

        # Check that generator was created correctly
        self.assertIsInstance(generator, OnDemandReportGenerator)
        self.assertEqual(generator.db_session, mock_db_session)
        self.assertEqual(generator.db_connector, mock_db_connector)


if __name__ == "__main__":
    unittest.main()
