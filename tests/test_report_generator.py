"""
Unit tests for the report generator module.

This module contains tests for the report generator functionality,
including HTML report generation for arbitrage opportunities and strategy performance.
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

from reporting.report_generator import (
    ReportGenerator,
    generate_arbitrage_report,
    generate_strategy_performance_report,
)


class TestReportGenerator(unittest.TestCase):
    """Test the report generator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock template environment
        self.mock_template_env = MagicMock()
        self.mock_template = MagicMock()
        self.mock_template_env.get_template.return_value = self.mock_template
        self.mock_template.render.return_value = "<html>Test Report</html>"

        # Create patches
        self.env_patcher = patch("jinja2.Environment")
        self.mock_env = self.env_patcher.start()
        self.mock_env.return_value = self.mock_template_env

        self.performance_viz_patcher = patch(
            "src.api.visualization.performance_visualization.PerformanceVisualization"
        )
        self.mock_performance_viz = self.performance_viz_patcher.start()

        self.portfolio_analytics_patcher = patch(
            "src.api.visualization.portfolio_analytics.PortfolioAnalytics"
        )
        self.mock_portfolio_analytics = self.portfolio_analytics_patcher.start()

        self.risk_analysis_patcher = patch(
            "src.api.visualization.risk_analysis.RiskAnalysis"
        )
        self.mock_risk_analysis = self.risk_analysis_patcher.start()

        self.setup_filters_patcher = patch(
            "src.reporting.jinja_filters.setup_jinja_filters"
        )
        self.mock_setup_filters = self.setup_filters_patcher.start()

        # Create report generator instance
        self.report_generator = ReportGenerator()

    def tearDown(self):
        """Tear down test fixtures."""
        self.env_patcher.stop()
        self.performance_viz_patcher.stop()
        self.portfolio_analytics_patcher.stop()
        self.risk_analysis_patcher.stop()
        self.setup_filters_patcher.stop()

    def test_init(self):
        """Test initialization of report generator."""
        self.assertIsNotNone(self.report_generator.template_env)
        self.assertIsNotNone(self.report_generator.performance_viz)
        self.assertIsNotNone(self.report_generator.portfolio_analytics)
        self.assertIsNotNone(self.report_generator.risk_analysis)
        self.mock_setup_filters.assert_called_once()

    def test_generate_strategy_performance_report(self):
        """Test generation of strategy performance report."""
        # Mock performance dashboard
        mock_dashboard = {"equity_curve": {"data": "test"}}
        self.report_generator.performance_viz.create_performance_dashboard.return_value = mock_dashboard

        # Mock portfolio dashboard
        mock_portfolio = {"portfolio_composition": {"data": "test"}}
        self.report_generator.portfolio_analytics.create_portfolio_dashboard.return_value = mock_portfolio

        # Create test data
        backtest_result = {
            "strategy_name": "Test Strategy",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.05,
            },
        }

        benchmark_data = {
            "name": "BTC/USD",
            "data": [{"timestamp": "2023-01-01", "close": 16500}],
        }

        # Generate report
        report = self.report_generator.generate_strategy_performance_report(
            backtest_result, benchmark_data
        )

        # Check that template was rendered
        self.mock_template.render.assert_called_once()
        args = self.mock_template.render.call_args[1]

        # Check that data was passed correctly
        self.assertEqual(args["strategy_name"], "Test Strategy")
        self.assertEqual(args["backtest_period"], "2023-01-01 to 2023-01-31")
        self.assertEqual(args["metrics"], backtest_result["metrics"])
        self.assertTrue(args["has_benchmark"])
        self.assertEqual(args["benchmark_name"], "BTC/USD")

        # Check that visualizations were included
        self.assertIn("visualizations", args)
        self.assertIn("equity_curve", args["visualizations"])
        self.assertIn("portfolio_composition", args["visualizations"])

        # Check that report was returned
        self.assertEqual(report, "<html>Test Report</html>")

    def test_generate_arbitrage_report(self):
        """Test generation of arbitrage report."""
        # Mock database session
        mock_db = MagicMock(spec=Session)

        # Mock query_arbitrage_data
        with patch.object(self.report_generator, "_query_arbitrage_data") as mock_query:
            # Create sample data
            sample_data = pd.DataFrame({
                "exchange": ["coinbase", "kraken"] * 10,
                "timestamp": pd.date_range(start="2023-01-01", periods=20, freq="1H"),
                "close": [1.0 + i * 0.001 for i in range(20)],
            })
            mock_query.return_value = sample_data

            # Mock analyze_arbitrage_opportunities
            with patch.object(
                self.report_generator, "_analyze_arbitrage_opportunities"
            ) as mock_analyze:
                # Create sample opportunities
                sample_opportunities = [
                    {
                        "timestamp": datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
                        "buy_exchange": "coinbase",
                        "sell_exchange": "kraken",
                        "buy_price": 1.0,
                        "sell_price": 1.002,
                        "diff": 0.002,
                        "pct_diff": 0.2,
                        "profit_potential": 0.002,
                    }
                ]
                mock_analyze.return_value = sample_opportunities

                # Mock calculate_arbitrage_metrics
                with patch.object(
                    self.report_generator, "_calculate_arbitrage_metrics"
                ) as mock_metrics:
                    # Create sample metrics
                    sample_metrics = {
                        "total_opportunities": 1,
                        "avg_profit_potential": 0.002,
                        "max_profit_potential": 0.002,
                        "total_profit_potential": 0.002,
                        "opportunity_frequency": 1.0,
                    }
                    mock_metrics.return_value = sample_metrics

                    # Mock generate_arbitrage_visualizations
                    with patch.object(
                        self.report_generator, "_generate_arbitrage_visualizations"
                    ) as mock_viz:
                        # Create sample visualizations
                        sample_viz = {
                            "price_comparison": {"data": "test"},
                            "opportunity_distribution": {"data": "test"},
                            "profit_potential": {"data": "test"},
                        }
                        mock_viz.return_value = sample_viz

                        # Generate report
                        start_time = datetime(2023, 1, 1, tzinfo=UTC)
                        end_time = datetime(2023, 1, 2, tzinfo=UTC)
                        report = self.report_generator.generate_arbitrage_report(
                            mock_db,
                            start_time,
                            end_time,
                            ["coinbase", "kraken"],
                            "USDC/USD",
                            0.001,
                        )

                        # Check that functions were called with correct arguments
                        mock_query.assert_called_once_with(
                            mock_db,
                            start_time,
                            end_time,
                            ["coinbase", "kraken"],
                            "USDC/USD",
                        )
                        mock_analyze.assert_called_once_with(
                            sample_data, ["coinbase", "kraken"], 0.001
                        )
                        mock_metrics.assert_called_once_with(
                            sample_opportunities, sample_data
                        )
                        mock_viz.assert_called_once_with(
                            sample_data, sample_opportunities
                        )

                        # Check that template was rendered
                        self.mock_template.render.assert_called_once()
                        args = self.mock_template.render.call_args[1]

                        # Check that data was passed correctly
                        self.assertEqual(args["symbol"], "USDC/USD")
                        self.assertEqual(args["exchanges"], ["coinbase", "kraken"])
                        self.assertEqual(args["threshold"], 0.1)  # 0.001 * 100
                        self.assertEqual(args["metrics"], sample_metrics)
                        self.assertEqual(args["opportunities"], sample_opportunities)
                        self.assertEqual(args["visualizations"], sample_viz)

                        # Check that report was returned
                        self.assertEqual(report, "<html>Test Report</html>")

    def test_analyze_arbitrage_opportunities(self):
        """Test analysis of arbitrage opportunities."""
        # Create sample data
        data = pd.DataFrame({
            "exchange": ["coinbase", "kraken", "binance"] * 5,
            "timestamp": pd.date_range(
                start="2023-01-01", periods=15, freq="1H"
            ).repeat(3),
            "close": [
                # coinbase prices
                1.000,
                1.001,
                1.002,
                1.003,
                1.004,
                # kraken prices
                1.002,
                1.000,
                1.004,
                1.001,
                1.003,
                # binance prices
                1.001,
                1.003,
                1.000,
                1.002,
                1.005,
            ],
        })

        # Analyze opportunities
        opportunities = self.report_generator._analyze_arbitrage_opportunities(
            data, ["coinbase", "kraken", "binance"], 0.001
        )

        # Check that opportunities were identified
        self.assertGreater(len(opportunities), 0)

        # Check structure of first opportunity
        first_op = opportunities[0]
        self.assertIn("timestamp", first_op)
        self.assertIn("buy_exchange", first_op)
        self.assertIn("sell_exchange", first_op)
        self.assertIn("buy_price", first_op)
        self.assertIn("sell_price", first_op)
        self.assertIn("diff", first_op)
        self.assertIn("pct_diff", first_op)
        self.assertIn("profit_potential", first_op)

        # Check that buy price is always less than sell price
        self.assertLess(first_op["buy_price"], first_op["sell_price"])

        # Check that opportunities are sorted by profit potential
        for i in range(len(opportunities) - 1):
            self.assertGreaterEqual(
                opportunities[i]["profit_potential"],
                opportunities[i + 1]["profit_potential"],
            )

    def test_calculate_arbitrage_metrics(self):
        """Test calculation of arbitrage metrics."""
        # Create sample opportunities
        opportunities = [
            {
                "timestamp": datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
                "buy_exchange": "coinbase",
                "sell_exchange": "kraken",
                "buy_price": 1.000,
                "sell_price": 1.002,
                "diff": 0.002,
                "pct_diff": 0.2,
                "profit_potential": 0.002,
            },
            {
                "timestamp": datetime(2023, 1, 1, 2, 0, tzinfo=UTC),
                "buy_exchange": "binance",
                "sell_exchange": "kraken",
                "buy_price": 1.001,
                "sell_price": 1.003,
                "diff": 0.002,
                "pct_diff": 0.2,
                "profit_potential": 0.002,
            },
            {
                "timestamp": datetime(2023, 1, 1, 3, 0, tzinfo=UTC),
                "buy_exchange": "coinbase",
                "sell_exchange": "binance",
                "buy_price": 1.000,
                "sell_price": 1.003,
                "diff": 0.003,
                "pct_diff": 0.3,
                "profit_potential": 0.003,
            },
        ]

        # Create sample data
        data = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=24, freq="1H"),
            "exchange": ["coinbase"] * 24,
            "close": [1.0] * 24,
        })

        # Calculate metrics
        metrics = self.report_generator._calculate_arbitrage_metrics(
            opportunities, data
        )

        # Check metrics
        self.assertEqual(metrics["total_opportunities"], 3)
        self.assertAlmostEqual(metrics["avg_profit_potential"], 0.00233, places=5)
        self.assertEqual(metrics["max_profit_potential"], 0.003)
        self.assertEqual(metrics["total_profit_potential"], 0.007)
        self.assertAlmostEqual(metrics["opportunity_frequency"], 3.0, places=1)

        # Check exchange pairs
        self.assertIn("top_exchange_pairs", metrics)
        self.assertGreater(len(metrics["top_exchange_pairs"]), 0)

    def test_create_price_comparison_chart(self):
        """Test creation of price comparison chart."""
        # Create sample data
        data = pd.DataFrame({
            "exchange": ["coinbase", "kraken"] * 10,
            "timestamp": pd.date_range(start="2023-01-01", periods=20, freq="1H"),
            "close": [1.0 + i * 0.001 for i in range(20)],
        })

        # Create chart
        chart = self.report_generator._create_price_comparison_chart(data)

        # Check that chart was created
        self.assertIsInstance(chart, dict)
        self.assertIn("data", chart)

    def test_create_opportunity_distribution_chart(self):
        """Test creation of opportunity distribution chart."""
        # Create sample opportunities
        opportunities = [
            {
                "timestamp": datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
                "buy_exchange": "coinbase",
                "sell_exchange": "kraken",
                "buy_price": 1.000,
                "sell_price": 1.002,
                "diff": 0.002,
                "pct_diff": 0.2,
                "profit_potential": 0.002,
            },
            {
                "timestamp": datetime(2023, 1, 1, 2, 0, tzinfo=UTC),
                "buy_exchange": "binance",
                "sell_exchange": "kraken",
                "buy_price": 1.001,
                "sell_price": 1.003,
                "diff": 0.002,
                "pct_diff": 0.2,
                "profit_potential": 0.002,
            },
            {
                "timestamp": datetime(2023, 1, 1, 3, 0, tzinfo=UTC),
                "buy_exchange": "coinbase",
                "sell_exchange": "binance",
                "buy_price": 1.000,
                "sell_price": 1.003,
                "diff": 0.003,
                "pct_diff": 0.3,
                "profit_potential": 0.003,
            },
        ]

        # Create chart
        chart = self.report_generator._create_opportunity_distribution_chart(
            opportunities
        )

        # Check that chart was created
        self.assertIsInstance(chart, dict)
        self.assertIn("data", chart)

    def test_create_profit_potential_chart(self):
        """Test creation of profit potential chart."""
        # Create sample opportunities
        opportunities = [
            {
                "timestamp": datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
                "buy_exchange": "coinbase",
                "sell_exchange": "kraken",
                "buy_price": 1.000,
                "sell_price": 1.002,
                "diff": 0.002,
                "pct_diff": 0.2,
                "profit_potential": 0.002,
            },
            {
                "timestamp": datetime(2023, 1, 1, 2, 0, tzinfo=UTC),
                "buy_exchange": "binance",
                "sell_exchange": "kraken",
                "buy_price": 1.001,
                "sell_price": 1.003,
                "diff": 0.002,
                "pct_diff": 0.2,
                "profit_potential": 0.002,
            },
        ]

        # Create chart
        chart = self.report_generator._create_profit_potential_chart(opportunities)

        # Check that chart was created
        self.assertIsInstance(chart, dict)
        self.assertIn("data", chart)

    def test_empty_opportunities(self):
        """Test handling of empty opportunities."""
        # Create empty opportunities list
        opportunities = []

        # Create sample data
        data = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=24, freq="1H"),
            "exchange": ["coinbase"] * 24,
            "close": [1.0] * 24,
        })

        # Calculate metrics
        metrics = self.report_generator._calculate_arbitrage_metrics(
            opportunities, data
        )

        # Check metrics
        self.assertEqual(metrics["total_opportunities"], 0)
        self.assertEqual(metrics["avg_profit_potential"], 0)
        self.assertEqual(metrics["max_profit_potential"], 0)
        self.assertEqual(metrics["total_profit_potential"], 0)
        self.assertEqual(metrics["opportunity_frequency"], 0)

        # Create charts
        dist_chart = self.report_generator._create_opportunity_distribution_chart(
            opportunities
        )
        profit_chart = self.report_generator._create_profit_potential_chart(
            opportunities
        )

        # Check that charts handle empty data
        self.assertIn("error", dist_chart)
        self.assertIn("error", profit_chart)

    def test_module_functions(self):
        """Test module-level functions."""
        # Mock ReportGenerator instance
        with patch("src.reporting.report_generator.report_generator") as mock_generator:
            # Mock generate_arbitrage_report
            mock_generator.generate_arbitrage_report.return_value = (
                "<html>Test Arbitrage Report</html>"
            )

            # Call module function
            mock_db = MagicMock()
            start_time = datetime(2023, 1, 1, tzinfo=UTC)
            end_time = datetime(2023, 1, 2, tzinfo=UTC)
            report = generate_arbitrage_report(
                mock_db, start_time, end_time, ["coinbase", "kraken"]
            )

            # Check that generator method was called
            mock_generator.generate_arbitrage_report.assert_called_once_with(
                mock_db, start_time, end_time, ["coinbase", "kraken"], "USDC/USD", 0.001
            )

            # Check that report was returned
            self.assertEqual(report, "<html>Test Arbitrage Report</html>")

            # Mock generate_strategy_performance_report
            mock_generator.generate_strategy_performance_report.return_value = (
                "<html>Test Strategy Report</html>"
            )

            # Call module function
            backtest_result = {"strategy_name": "Test Strategy"}
            report = generate_strategy_performance_report(backtest_result)

            # Check that generator method was called
            mock_generator.generate_strategy_performance_report.assert_called_once_with(
                backtest_result, None, None
            )

            # Check that report was returned
            self.assertEqual(report, "<html>Test Strategy Report</html>")


if __name__ == "__main__":
    unittest.main()
