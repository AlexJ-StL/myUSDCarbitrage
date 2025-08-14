"""Tests for the backtesting performance metrics calculator."""

import os
import sys
import unittest
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add src directory to Python path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from api.backtesting.metrics import PerformanceMetrics


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics calculations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample equity curve
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        np.random.seed(42)  # For reproducible results

        # Generate synthetic returns with some volatility
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        equity_values = [10000]  # Starting equity

        for ret in returns:
            equity_values.append(equity_values[-1] * (1 + ret))

        self.equity_curve = pd.Series(equity_values[1:], index=dates)

        # Create sample trades data
        self.trades_data = pd.DataFrame({
            "entry_price": [1.0, 1.1, 0.9, 1.05, 0.95],
            "exit_price": [1.1, 1.0, 1.0, 1.15, 0.85],
            "entry_time": [
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
                datetime(2023, 3, 1, tzinfo=UTC),
                datetime(2023, 4, 1, tzinfo=UTC),
                datetime(2023, 5, 1, tzinfo=UTC),
            ],
            "exit_time": [
                datetime(2023, 1, 15, tzinfo=UTC),
                datetime(2023, 2, 15, tzinfo=UTC),
                datetime(2023, 3, 15, tzinfo=UTC),
                datetime(2023, 4, 15, tzinfo=UTC),
                datetime(2023, 5, 15, tzinfo=UTC),
            ],
            "pnl": [100, -100, 100, 100, -100],
        })

        # Create benchmark data
        benchmark_returns = np.random.normal(0.0003, 0.015, len(dates))
        benchmark_values = [10000]

        for ret in benchmark_returns:
            benchmark_values.append(benchmark_values[-1] * (1 + ret))

        self.benchmark = pd.Series(benchmark_values[1:], index=dates)

        # Initialize metrics calculator
        self.metrics_calculator = PerformanceMetrics(risk_free_rate=0.02)

    def test_initialization(self):
        """Test metrics calculator initialization."""
        calculator = PerformanceMetrics(risk_free_rate=0.03)
        self.assertEqual(calculator.risk_free_rate, 0.03)

    def test_calculate_drawdown_metrics(self):
        """Test drawdown metrics calculation."""
        # Create equity curve with known drawdown
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        equity_values = [
            10000,
            11000,
            12000,
            10000,
            9000,
            8000,
            9000,
            10000,
            11000,
            12000,
        ]
        equity_curve = pd.Series(equity_values, index=dates)

        drawdown_metrics = self.metrics_calculator.calculate_drawdown_metrics(
            equity_curve
        )

        # Check that metrics are calculated
        self.assertIn("max_drawdown", drawdown_metrics)
        self.assertIn("drawdown_duration", drawdown_metrics)
        self.assertIn("recovery_duration", drawdown_metrics)
        self.assertIn("ulcer_index", drawdown_metrics)

        # Max drawdown should be negative
        self.assertLess(drawdown_metrics["max_drawdown"], 0)

        # Drawdown duration should be positive
        self.assertGreaterEqual(drawdown_metrics["drawdown_duration"], 0)

    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        returns = self.equity_curve.pct_change().dropna()
        risk_metrics = self.metrics_calculator.calculate_risk_metrics(returns)

        # Check that all risk metrics are present
        expected_metrics = [
            "annualized_volatility",
            "downside_deviation",
            "semi_deviation",
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
            "skewness",
            "kurtosis",
            "max_consecutive_losses",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, risk_metrics)

        # Volatility should be positive
        self.assertGreater(risk_metrics["annualized_volatility"], 0)

        # VaR should be negative (representing losses)
        self.assertLess(risk_metrics["var_95"], 0)
        self.assertLess(risk_metrics["var_99"], 0)

        # CVaR should be more negative than VaR
        self.assertLessEqual(risk_metrics["cvar_95"], risk_metrics["var_95"])
        self.assertLessEqual(risk_metrics["cvar_99"], risk_metrics["var_99"])

    def test_calculate_return_metrics(self):
        """Test return metrics calculation."""
        returns = self.equity_curve.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)

        return_metrics = self.metrics_calculator.calculate_return_metrics(
            self.equity_curve, returns, volatility
        )

        # Check that all return metrics are present
        expected_metrics = [
            "cagr",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "mar_ratio",
            "omega_ratio",
            "gain_to_pain",
            "kestner_ratio",
            "avg_monthly_return",
            "monthly_volatility",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, return_metrics)

        # CAGR should be reasonable
        self.assertIsInstance(return_metrics["cagr"], (int, float))

        # Sharpe ratio should be finite
        self.assertTrue(np.isfinite(return_metrics["sharpe_ratio"]))

        # Sortino ratio should be finite
        self.assertTrue(np.isfinite(return_metrics["sortino_ratio"]))

    def test_calculate_trade_metrics(self):
        """Test trade metrics calculation."""
        trade_metrics = self.metrics_calculator.calculate_trade_metrics(
            self.trades_data
        )

        # Check that all trade metrics are present
        expected_metrics = [
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "avg_trade_pnl",
            "avg_winning_trade",
            "avg_losing_trade",
            "profit_factor",
            "expectancy",
            "max_consecutive_wins",
            "max_consecutive_losses",
            "largest_win",
            "largest_loss",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, trade_metrics)

        # Check specific values
        self.assertEqual(trade_metrics["total_trades"], 5)
        self.assertEqual(trade_metrics["winning_trades"], 3)
        self.assertEqual(trade_metrics["losing_trades"], 2)
        self.assertEqual(trade_metrics["win_rate"], 0.6)
        self.assertEqual(trade_metrics["avg_trade_pnl"], 0.0)  # (100-100+100+100-100)/5
        self.assertEqual(trade_metrics["largest_win"], 100)
        self.assertEqual(trade_metrics["largest_loss"], -100)

    def test_calculate_benchmark_metrics(self):
        """Test benchmark comparison metrics."""
        returns = self.equity_curve.pct_change().dropna()
        benchmark_returns = self.benchmark.pct_change().dropna()

        # Align the series
        common_index = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        benchmark_metrics = self.metrics_calculator.calculate_benchmark_metrics(
            returns, benchmark_returns
        )

        # Check that all benchmark metrics are present
        expected_metrics = [
            "alpha",
            "beta",
            "information_ratio",
            "up_capture",
            "down_capture",
            "correlation",
            "r_squared",
            "outperformance",
            "tracking_error",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, benchmark_metrics)

        # Beta should be reasonable
        self.assertTrue(np.isfinite(benchmark_metrics["beta"]))

        # Correlation should be between -1 and 1
        self.assertGreaterEqual(benchmark_metrics["correlation"], -1)
        self.assertLessEqual(benchmark_metrics["correlation"], 1)

        # R-squared should be between 0 and 1
        self.assertGreaterEqual(benchmark_metrics["r_squared"], 0)
        self.assertLessEqual(benchmark_metrics["r_squared"], 1)

    def test_comprehensive_metrics_calculation(self):
        """Test comprehensive metrics calculation."""
        metrics = self.metrics_calculator.calculate_metrics(
            equity_curve=self.equity_curve,
            trades=self.trades_data,
            benchmark=self.benchmark,
        )

        # Check that all main categories are present
        expected_categories = [
            "total_return",
            "drawdown",
            "risk",
            "returns",
            "trades",
            "benchmark",
        ]

        for category in expected_categories:
            self.assertIn(category, metrics)

        # Check that total return is calculated
        self.assertIsInstance(metrics["total_return"], (int, float))

        # Check that drawdown metrics are present
        self.assertIn("max_drawdown", metrics["drawdown"])

        # Check that risk metrics are present
        self.assertIn("annualized_volatility", metrics["risk"])

        # Check that return metrics are present
        self.assertIn("sharpe_ratio", metrics["returns"])

        # Check that trade metrics are present
        self.assertIn("win_rate", metrics["trades"])

        # Check that benchmark metrics are present
        self.assertIn("alpha", metrics["benchmark"])

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        # Test with empty trades
        empty_trades = pd.DataFrame()
        trade_metrics = self.metrics_calculator.calculate_trade_metrics(empty_trades)
        self.assertEqual(trade_metrics, {})

        # Test with single data point
        single_point = pd.Series([10000], index=[datetime(2023, 1, 1)])
        returns = single_point.pct_change().dropna()

        if len(returns) == 0:
            # This is expected for a single data point
            self.assertEqual(len(returns), 0)

    def test_edge_cases(self):
        """Test edge cases in metrics calculation."""
        # Test with all positive returns
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        positive_equity = pd.Series([10000 * (1.01**i) for i in range(10)], index=dates)
        positive_returns = positive_equity.pct_change().dropna()

        risk_metrics = self.metrics_calculator.calculate_risk_metrics(positive_returns)

        # Downside deviation should be 0 (no negative returns)
        self.assertEqual(risk_metrics["downside_deviation"], 0)

        # Max consecutive losses should be 0
        self.assertEqual(risk_metrics["max_consecutive_losses"], 0)

        # Test with all negative returns
        negative_equity = pd.Series([10000 * (0.99**i) for i in range(10)], index=dates)
        negative_returns = negative_equity.pct_change().dropna()

        risk_metrics = self.metrics_calculator.calculate_risk_metrics(negative_returns)

        # Max consecutive losses should equal the number of returns
        self.assertEqual(risk_metrics["max_consecutive_losses"], len(negative_returns))

    def test_regime_metrics(self):
        """Test regime-based metrics calculation."""
        # Create regime data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        regime_data = pd.DataFrame(
            {"regime": ["bull"] * 50 + ["bear"] * 50}, index=dates
        )

        # Create returns data
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

        regime_metrics = self.metrics_calculator.calculate_regime_metrics(
            returns, regime_data
        )

        # Check that metrics are calculated for each regime
        self.assertIn("bull", regime_metrics)
        self.assertIn("bear", regime_metrics)

        # Check that each regime has the expected metrics
        for regime in ["bull", "bear"]:
            self.assertIn("total_return", regime_metrics[regime])
            self.assertIn("volatility", regime_metrics[regime])
            self.assertIn("sharpe_ratio", regime_metrics[regime])
            self.assertIn("win_rate", regime_metrics[regime])
            self.assertIn("num_periods", regime_metrics[regime])


if __name__ == "__main__":
    unittest.main()
