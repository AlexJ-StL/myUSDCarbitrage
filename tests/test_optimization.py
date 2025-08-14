"""Tests for the walk-forward optimization engine."""

import os
import sys
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add src directory to Python path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from api.backtesting.optimization import OptimizationResult, WalkForwardOptimizer
from api.backtesting.engine import BacktestEngine


class TestOptimizationResult(unittest.TestCase):
    """Test optimization result class."""

    def test_optimization_result_creation(self):
        """Test optimization result creation and to_dict method."""
        # Create sample data
        parameters = {"param1": 1.0, "param2": 2.0}
        metrics = {"sharpe_ratio": 1.5, "total_return": 0.2}

        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        equity_curve = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        trades = pd.DataFrame({
            "entry_price": [1.0, 1.1],
            "exit_price": [1.1, 1.0],
            "pnl": [100, -100],
        })

        backtest_results = {"portfolio": {}, "metrics": metrics}

        # Create optimization result
        result = OptimizationResult(
            parameters=parameters,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            backtest_results=backtest_results,
        )

        # Test attributes
        self.assertEqual(result.parameters, parameters)
        self.assertEqual(result.metrics, metrics)
        self.assertTrue(result.equity_curve.equals(equity_curve))
        self.assertTrue(result.trades.equals(trades))
        self.assertEqual(result.backtest_results, backtest_results)

        # Test to_dict method
        result_dict = result.to_dict()
        self.assertIn("parameters", result_dict)
        self.assertIn("metrics", result_dict)
        self.assertIn("equity_curve", result_dict)
        self.assertIn("trades", result_dict)
        self.assertIn("backtest_results", result_dict)


@patch("api.backtesting.optimization.BacktestEngine")
class TestWalkForwardOptimizer(unittest.TestCase):
    """Test walk-forward optimizer."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample equity curve data
        self.dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        equity_values = [10000]

        for ret in returns:
            equity_values.append(equity_values[-1] * (1 + ret))

        self.equity_curve = pd.Series(equity_values[1:], index=self.dates)

        # Create sample backtest results
        self.sample_backtest_results = {
            "portfolio": {
                "equity_curve": [
                    {"timestamp": date.isoformat(), "equity": equity}
                    for date, equity in zip(self.dates, equity_values[1:])
                ],
                "transactions": [
                    {
                        "exchange": "coinbase",
                        "symbol": "USDC/USD",
                        "side": "buy",
                        "amount": 100,
                        "price": 1.0,
                        "fee": 1.0,
                        "timestamp": "2023-01-01T00:00:00",
                    }
                ],
            },
            "metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.05,
                "cagr": 0.08,
            },
        }

    def test_optimizer_initialization(self, mock_engine):
        """Test optimizer initialization."""
        optimizer = WalkForwardOptimizer(
            backtest_engine=mock_engine,
            optimization_metric="sharpe_ratio",
            n_jobs=2,
        )

        self.assertEqual(optimizer.backtest_engine, mock_engine)
        self.assertEqual(optimizer.optimization_metric, "sharpe_ratio")
        self.assertEqual(optimizer.n_jobs, 2)

    def test_grid_search(self, mock_engine):
        """Test grid search optimization."""
        # Setup mock
        mock_engine.run_backtest.return_value = self.sample_backtest_results

        optimizer = WalkForwardOptimizer(mock_engine)

        # Define strategy function
        def sample_strategy(timestamp, portfolio, market_data, params):
            return []

        # Define parameter grid
        parameter_grid = {
            "param1": [1.0, 2.0],
            "param2": [0.1, 0.2],
        }

        # Run grid search
        results = optimizer.grid_search(
            strategy_func=sample_strategy,
            parameter_grid=parameter_grid,
            exchanges=["coinbase"],
            symbols=["USDC/USD"],
            timeframe="1h",
            start_date=datetime(2023, 1, 1, tzinfo=UTC),
            end_date=datetime(2023, 12, 31, tzinfo=UTC),
            validation_split=0.3,
        )

        # Check results
        self.assertEqual(len(results), 4)  # 2 * 2 parameter combinations

        # Check that results are sorted by optimization metric
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].metrics.get("sharpe_ratio", 0),
                results[i + 1].metrics.get("sharpe_ratio", 0),
            )

        # Check that backtest was called for each parameter combination
        self.assertEqual(mock_engine.run_backtest.call_count, 4)

    def test_walk_forward_analysis(self, mock_engine):
        """Test walk-forward analysis."""
        # Setup mock
        mock_engine.run_backtest.return_value = self.sample_backtest_results

        optimizer = WalkForwardOptimizer(mock_engine)

        # Define strategy function
        def sample_strategy(timestamp, portfolio, market_data, params):
            return []

        # Define parameter grid
        parameter_grid = {
            "param1": [1.0, 2.0],
        }

        # Run walk-forward analysis
        results = optimizer.walk_forward_analysis(
            strategy_func=sample_strategy,
            parameter_grid=parameter_grid,
            exchanges=["coinbase"],
            symbols=["USDC/USD"],
            timeframe="1h",
            start_date=datetime(2023, 1, 1, tzinfo=UTC),
            end_date=datetime(2023, 6, 30, tzinfo=UTC),
            training_window=60,  # 60 days
            reoptimization_frequency=30,  # 30 days
            min_training_periods=30,  # 30 days minimum
        )

        # Check results structure
        self.assertIn("periods", results)
        self.assertIn("combined_equity_curve", results)
        self.assertIn("walk_forward_metrics", results)
        self.assertIn("stability_metrics", results)
        self.assertIn("overfitting_metrics", results)
        self.assertIn("summary", results)

        # Check that periods were created
        self.assertGreater(len(results["periods"]), 0)

        # Check summary metrics
        summary = results["summary"]
        self.assertIn("total_periods", summary)
        self.assertIn("avg_oos_return", summary)
        self.assertIn("avg_oos_sharpe", summary)

    def test_statistical_significance_test(self, mock_engine):
        """Test statistical significance testing."""
        optimizer = WalkForwardOptimizer(mock_engine)

        # Create two optimization results with different performance
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Strategy 1: Better performance
        equity1 = pd.Series([10000 * (1.001**i) for i in range(100)], index=dates)
        result1 = OptimizationResult(
            parameters={"param1": 1.0},
            metrics={"sharpe_ratio": 1.5},
            equity_curve=equity1,
            trades=pd.DataFrame(),
            backtest_results={},
        )

        # Strategy 2: Worse performance
        equity2 = pd.Series([10000 * (1.0005**i) for i in range(100)], index=dates)
        result2 = OptimizationResult(
            parameters={"param1": 2.0},
            metrics={"sharpe_ratio": 1.0},
            equity_curve=equity2,
            trades=pd.DataFrame(),
            backtest_results={},
        )

        # Test t-test
        test_results = optimizer.statistical_significance_test(
            result1, result2, test_type="t_test", confidence_level=0.95
        )

        # Check results structure
        self.assertIn("test_name", test_results)
        self.assertIn("test_statistic", test_results)
        self.assertIn("p_value", test_results)
        self.assertIn("is_significant", test_results)
        self.assertIn("confidence_interval", test_results)
        self.assertIn("effect_size", test_results)
        self.assertIn("interpretation", test_results)

        # Test should detect that strategy 1 is better
        self.assertGreater(test_results["mean_difference"], 0)

        # Test Wilcoxon test
        wilcoxon_results = optimizer.statistical_significance_test(
            result1, result2, test_type="wilcoxon", confidence_level=0.95
        )

        self.assertEqual(wilcoxon_results["test_name"], "Wilcoxon signed-rank test")

        # Test bootstrap test
        bootstrap_results = optimizer.statistical_significance_test(
            result1, result2, test_type="bootstrap", confidence_level=0.95
        )

        self.assertEqual(bootstrap_results["test_name"], "Bootstrap test")

    def test_monte_carlo_analysis(self, mock_engine):
        """Test Monte Carlo analysis."""
        # Setup mock
        mock_engine.run_backtest.return_value = self.sample_backtest_results

        optimizer = WalkForwardOptimizer(mock_engine)

        # Define strategy function
        def sample_strategy(timestamp, portfolio, market_data, params):
            return []

        # Run Monte Carlo analysis
        results = optimizer.monte_carlo_analysis(
            strategy_func=sample_strategy,
            parameters={"param1": 1.0},
            exchanges=["coinbase"],
            symbols=["USDC/USD"],
            timeframe="1h",
            start_date=datetime(2023, 1, 1, tzinfo=UTC),
            end_date=datetime(2023, 3, 31, tzinfo=UTC),
            n_simulations=100,  # Small number for testing
            bootstrap_block_size=10,
        )

        # Check results structure
        self.assertIn("original_metrics", results)
        self.assertIn("simulation_results", results)
        self.assertIn("confidence_intervals", results)
        self.assertIn("risk_metrics", results)
        self.assertIn("summary", results)

        # Check simulation results
        self.assertEqual(len(results["simulation_results"]), 100)

        # Check confidence intervals
        ci = results["confidence_intervals"]
        for metric in ["total_return", "sharpe_ratio", "max_drawdown", "cagr"]:
            self.assertIn(metric, ci)
            self.assertIn("5th_percentile", ci[metric])
            self.assertIn("95th_percentile", ci[metric])
            self.assertIn("median", ci[metric])
            self.assertIn("mean", ci[metric])
            self.assertIn("std", ci[metric])

        # Check risk metrics
        risk_metrics = results["risk_metrics"]
        self.assertIn("probability_positive_return", risk_metrics)
        self.assertIn("var_95", risk_metrics)
        self.assertIn("var_99", risk_metrics)
        self.assertIn("expected_shortfall_95", risk_metrics)
        self.assertIn("expected_shortfall_99", risk_metrics)

        # Check that probability is between 0 and 1
        self.assertGreaterEqual(risk_metrics["probability_positive_return"], 0)
        self.assertLessEqual(risk_metrics["probability_positive_return"], 1)

    def test_block_bootstrap(self, mock_engine):
        """Test block bootstrap method."""
        optimizer = WalkForwardOptimizer(mock_engine)

        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        # Test block bootstrap
        bootstrap_sample = optimizer._block_bootstrap(returns, block_size=10)

        # Check that bootstrap sample has same length as original
        self.assertEqual(len(bootstrap_sample), len(returns))

        # Check that bootstrap sample is different from original
        # (with high probability, though not guaranteed)
        self.assertFalse(bootstrap_sample.equals(returns))

    def test_parameter_stability_calculation(self, mock_engine):
        """Test parameter stability calculation."""
        optimizer = WalkForwardOptimizer(mock_engine)

        # Test with no previous results
        current_params = {"param1": 1.0, "param2": 2.0}
        stability = optimizer._calculate_parameter_stability(current_params, [])

        self.assertEqual(stability["stability_score"], 1.0)

        # Test with previous results
        previous_results = [
            {"best_params": {"param1": 1.1, "param2": 2.1}},
        ]

        stability = optimizer._calculate_parameter_stability(
            current_params, previous_results
        )

        self.assertIn("stability_score", stability)
        self.assertIn("parameter_changes", stability)
        self.assertGreaterEqual(stability["stability_score"], 0)
        self.assertLessEqual(stability["stability_score"], 1)

    def test_overfitting_metrics_calculation(self, mock_engine):
        """Test overfitting metrics calculation."""
        optimizer = WalkForwardOptimizer(mock_engine)

        # Create sample walk-forward results
        wf_results = [
            {
                "training_metrics": {"total_return": 0.15, "sharpe_ratio": 1.8},
                "oos_metrics": {"total_return": 0.10, "sharpe_ratio": 1.2},
            },
            {
                "training_metrics": {"total_return": 0.12, "sharpe_ratio": 1.5},
                "oos_metrics": {"total_return": 0.08, "sharpe_ratio": 1.0},
            },
        ]

        overfitting_metrics = optimizer._calculate_overfitting_metrics(wf_results)

        # Check that metrics are calculated
        self.assertIn("return_degradation", overfitting_metrics)
        self.assertIn("sharpe_degradation", overfitting_metrics)
        self.assertIn("return_correlation", overfitting_metrics)
        self.assertIn("sharpe_correlation", overfitting_metrics)
        self.assertIn("overfitting_score", overfitting_metrics)

        # Check that degradation is positive (IS > OOS)
        self.assertGreater(overfitting_metrics["return_degradation"], 0)
        self.assertGreater(overfitting_metrics["sharpe_degradation"], 0)


if __name__ == "__main__":
    unittest.main()
