"""Walk-forward optimization engine for backtesting."""

import itertools
import logging
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import ParameterGrid

from .engine import BacktestEngine
from .metrics import PerformanceMetrics

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("backtesting.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class OptimizationResult:
    """Result of parameter optimization."""

    def __init__(
        self,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        backtest_results: Dict[str, Any],
    ):
        """Initialize optimization result.

        Args:
            parameters: Strategy parameters used
            metrics: Performance metrics
            equity_curve: Equity curve from backtest
            trades: Trades DataFrame
            backtest_results: Full backtest results
        """
        self.parameters = parameters
        self.metrics = metrics
        self.equity_curve = equity_curve
        self.trades = trades
        self.backtest_results = backtest_results

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "parameters": self.parameters,
            "metrics": self.metrics,
            "equity_curve": self.equity_curve.to_dict(),
            "trades": self.trades.to_dict("records") if not self.trades.empty else [],
            "backtest_results": self.backtest_results,
        }


class WalkForwardOptimizer:
    """Walk-forward optimization engine for strategy parameters."""

    def __init__(
        self,
        backtest_engine: BacktestEngine,
        optimization_metric: str = "sharpe_ratio",
        n_jobs: int = -1,
    ):
        """Initialize walk-forward optimizer.

        Args:
            backtest_engine: Backtesting engine to use
            optimization_metric: Metric to optimize for
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.backtest_engine = backtest_engine
        self.optimization_metric = optimization_metric
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.performance_calculator = PerformanceMetrics()

    def grid_search(
        self,
        strategy_func: Callable,
        parameter_grid: Dict[str, List[Any]],
        exchanges: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        validation_split: float = 0.3,
    ) -> List[OptimizationResult]:
        """Perform grid search optimization.

        Args:
            strategy_func: Strategy function to optimize
            parameter_grid: Grid of parameters to test
            exchanges: List of exchanges
            symbols: List of symbols
            timeframe: Timeframe for backtesting
            start_date: Start date
            end_date: End date
            validation_split: Fraction of data to use for validation

        Returns:
            List[OptimizationResult]: Optimization results sorted by performance
        """
        logger.info(
            f"Starting grid search optimization with {len(list(ParameterGrid(parameter_grid)))} parameter combinations"
        )

        # Split data into training and validation periods
        total_days = (end_date - start_date).days
        training_days = int(total_days * (1 - validation_split))
        training_end = start_date + timedelta(days=training_days)

        # Generate parameter combinations
        param_combinations = list(ParameterGrid(parameter_grid))

        # Run backtests for each parameter combination
        results = []
        for i, params in enumerate(param_combinations):
            logger.info(
                f"Testing parameter combination {i + 1}/{len(param_combinations)}: {params}"
            )

            try:
                # Run backtest on training data
                backtest_results = self.backtest_engine.run_backtest(
                    strategy_func=strategy_func,
                    exchanges=exchanges,
                    symbols=symbols,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=training_end,
                    strategy_params=params,
                )

                # Extract metrics and data
                metrics = backtest_results.get("metrics", {})
                portfolio_data = backtest_results.get("portfolio", {})

                # Create equity curve
                equity_curve_data = portfolio_data.get("equity_curve", [])
                if equity_curve_data:
                    timestamps = [
                        datetime.fromisoformat(item["timestamp"])
                        for item in equity_curve_data
                    ]
                    equity_values = [item["equity"] for item in equity_curve_data]
                    equity_curve = pd.Series(equity_values, index=timestamps)
                else:
                    equity_curve = pd.Series()

                # Create trades DataFrame
                trades_data = portfolio_data.get("transactions", [])
                trades_df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()

                # Create optimization result
                result = OptimizationResult(
                    parameters=params,
                    metrics=metrics,
                    equity_curve=equity_curve,
                    trades=trades_df,
                    backtest_results=backtest_results,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")
                continue

        # Sort results by optimization metric
        if results:
            results.sort(
                key=lambda x: x.metrics.get(self.optimization_metric, float("-inf")),
                reverse=True,
            )

        logger.info(
            f"Grid search completed. Best {self.optimization_metric}: {results[0].metrics.get(self.optimization_metric, 'N/A') if results else 'N/A'}"
        )

        return results

    def walk_forward_analysis(
        self,
        strategy_func: Callable,
        parameter_grid: Dict[str, List[Any]],
        exchanges: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        training_window: int = 252,  # Trading days
        reoptimization_frequency: int = 63,  # Trading days (quarterly)
        min_training_periods: int = 126,  # Minimum training periods
    ) -> Dict[str, Any]:
        """Perform walk-forward analysis to prevent overfitting.

        Args:
            strategy_func: Strategy function to optimize
            parameter_grid: Grid of parameters to test
            exchanges: List of exchanges
            symbols: List of symbols
            timeframe: Timeframe for backtesting
            start_date: Start date
            end_date: End date
            training_window: Number of trading days for training window
            reoptimization_frequency: How often to reoptimize (trading days)
            min_training_periods: Minimum training periods required

        Returns:
            Dict[str, Any]: Walk-forward analysis results
        """
        logger.info("Starting walk-forward analysis")

        # Calculate walk-forward periods
        total_days = (end_date - start_date).days
        periods = []
        current_date = start_date

        while current_date < end_date:
            # Training period
            training_start = current_date
            training_end = min(
                current_date + timedelta(days=training_window),
                end_date - timedelta(days=reoptimization_frequency),
            )

            # Out-of-sample period
            oos_start = training_end
            oos_end = min(
                training_end + timedelta(days=reoptimization_frequency), end_date
            )

            if (
                training_end - training_start
            ).days >= min_training_periods and oos_end > oos_start:
                periods.append({
                    "training_start": training_start,
                    "training_end": training_end,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                })

            current_date = oos_end

        logger.info(f"Created {len(periods)} walk-forward periods")

        # Run optimization for each period
        wf_results = []
        combined_equity_curve = pd.Series()
        combined_trades = pd.DataFrame()

        for i, period in enumerate(periods):
            logger.info(f"Processing walk-forward period {i + 1}/{len(periods)}")

            try:
                # Optimize on training data
                training_results = self.grid_search(
                    strategy_func=strategy_func,
                    parameter_grid=parameter_grid,
                    exchanges=exchanges,
                    symbols=symbols,
                    timeframe=timeframe,
                    start_date=period["training_start"],
                    end_date=period["training_end"],
                    validation_split=0.0,  # Use all training data
                )

                if not training_results:
                    logger.warning(f"No valid results for period {i + 1}")
                    continue

                # Get best parameters
                best_params = training_results[0].parameters

                # Test on out-of-sample data
                oos_results = self.backtest_engine.run_backtest(
                    strategy_func=strategy_func,
                    exchanges=exchanges,
                    symbols=symbols,
                    timeframe=timeframe,
                    start_date=period["oos_start"],
                    end_date=period["oos_end"],
                    strategy_params=best_params,
                )

                # Extract out-of-sample metrics
                oos_metrics = oos_results.get("metrics", {})
                oos_portfolio = oos_results.get("portfolio", {})

                # Create out-of-sample equity curve
                oos_equity_data = oos_portfolio.get("equity_curve", [])
                if oos_equity_data:
                    timestamps = [
                        datetime.fromisoformat(item["timestamp"])
                        for item in oos_equity_data
                    ]
                    equity_values = [item["equity"] for item in oos_equity_data]
                    oos_equity_curve = pd.Series(equity_values, index=timestamps)

                    # Combine with overall equity curve
                    if combined_equity_curve.empty:
                        combined_equity_curve = oos_equity_curve
                    else:
                        # Adjust for continuity
                        last_value = combined_equity_curve.iloc[-1]
                        first_value = oos_equity_curve.iloc[0]
                        adjustment_factor = last_value / first_value
                        adjusted_oos = oos_equity_curve * adjustment_factor
                        combined_equity_curve = pd.concat([
                            combined_equity_curve,
                            adjusted_oos[1:],
                        ])

                # Store period results
                period_result = {
                    "period": i + 1,
                    "training_period": {
                        "start": period["training_start"],
                        "end": period["training_end"],
                    },
                    "oos_period": {
                        "start": period["oos_start"],
                        "end": period["oos_end"],
                    },
                    "best_params": best_params,
                    "training_metrics": training_results[0].metrics,
                    "oos_metrics": oos_metrics,
                    "parameter_stability": self._calculate_parameter_stability(
                        best_params, wf_results
                    ),
                }
                wf_results.append(period_result)

            except Exception as e:
                logger.error(f"Error in walk-forward period {i + 1}: {e}")
                continue

        # Calculate overall walk-forward metrics
        if combined_equity_curve.empty:
            logger.error("No valid walk-forward results generated")
            return {"error": "No valid results"}

        # Calculate comprehensive metrics for the combined equity curve
        wf_metrics = self.performance_calculator.calculate_metrics(
            combined_equity_curve
        )

        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(wf_results)

        # Calculate overfitting metrics
        overfitting_metrics = self._calculate_overfitting_metrics(wf_results)

        return {
            "periods": wf_results,
            "combined_equity_curve": combined_equity_curve.to_dict(),
            "walk_forward_metrics": wf_metrics,
            "stability_metrics": stability_metrics,
            "overfitting_metrics": overfitting_metrics,
            "summary": {
                "total_periods": len(wf_results),
                "avg_oos_return": np.mean([
                    p["oos_metrics"].get("total_return", 0) for p in wf_results
                ]),
                "avg_oos_sharpe": np.mean([
                    p["oos_metrics"].get("sharpe_ratio", 0) for p in wf_results
                ]),
                "parameter_consistency": stability_metrics.get(
                    "parameter_consistency", 0
                ),
            },
        }

    def _calculate_parameter_stability(
        self, current_params: Dict[str, Any], previous_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate parameter stability metrics.

        Args:
            current_params: Current best parameters
            previous_results: Previous period results

        Returns:
            Dict[str, float]: Stability metrics
        """
        if not previous_results:
            return {"stability_score": 1.0}

        # Compare with previous period
        prev_params = previous_results[-1]["best_params"]

        # Calculate parameter changes
        param_changes = {}
        for param, value in current_params.items():
            if param in prev_params:
                if isinstance(value, (int, float)):
                    prev_value = prev_params[param]
                    if prev_value != 0:
                        change = abs(value - prev_value) / abs(prev_value)
                    else:
                        change = 1.0 if value != 0 else 0.0
                    param_changes[param] = change
                else:
                    param_changes[param] = 0.0 if value == prev_params[param] else 1.0

        # Calculate overall stability score
        if param_changes:
            avg_change = np.mean(list(param_changes.values()))
            stability_score = max(0, 1 - avg_change)
        else:
            stability_score = 1.0

        return {
            "stability_score": stability_score,
            "parameter_changes": param_changes,
        }

    def _calculate_stability_metrics(
        self, wf_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate overall stability metrics across all periods.

        Args:
            wf_results: Walk-forward results

        Returns:
            Dict[str, float]: Stability metrics
        """
        if len(wf_results) < 2:
            return {"parameter_consistency": 1.0}

        # Calculate parameter consistency
        all_params = [result["best_params"] for result in wf_results]
        param_names = set()
        for params in all_params:
            param_names.update(params.keys())

        consistency_scores = []
        for param_name in param_names:
            param_values = []
            for params in all_params:
                if param_name in params:
                    param_values.append(params[param_name])

            if len(param_values) > 1 and all(
                isinstance(v, (int, float)) for v in param_values
            ):
                # Calculate coefficient of variation for numeric parameters
                mean_val = np.mean(param_values)
                std_val = np.std(param_values)
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    consistency = max(0, 1 - cv)
                else:
                    consistency = 1.0 if std_val == 0 else 0.0
                consistency_scores.append(consistency)

        parameter_consistency = (
            np.mean(consistency_scores) if consistency_scores else 1.0
        )

        # Calculate performance consistency
        oos_returns = [
            result["oos_metrics"].get("total_return", 0) for result in wf_results
        ]
        oos_sharpes = [
            result["oos_metrics"].get("sharpe_ratio", 0) for result in wf_results
        ]

        return_consistency = 1 - (
            np.std(oos_returns) / (abs(np.mean(oos_returns)) + 1e-6)
        )
        sharpe_consistency = 1 - (
            np.std(oos_sharpes) / (abs(np.mean(oos_sharpes)) + 1e-6)
        )

        return {
            "parameter_consistency": parameter_consistency,
            "return_consistency": max(0, return_consistency),
            "sharpe_consistency": max(0, sharpe_consistency),
        }

    def _calculate_overfitting_metrics(
        self, wf_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate overfitting metrics.

        Args:
            wf_results: Walk-forward results

        Returns:
            Dict[str, float]: Overfitting metrics
        """
        if not wf_results:
            return {}

        # Compare in-sample vs out-of-sample performance
        is_returns = [
            result["training_metrics"].get("total_return", 0) for result in wf_results
        ]
        oos_returns = [
            result["oos_metrics"].get("total_return", 0) for result in wf_results
        ]

        is_sharpes = [
            result["training_metrics"].get("sharpe_ratio", 0) for result in wf_results
        ]
        oos_sharpes = [
            result["oos_metrics"].get("sharpe_ratio", 0) for result in wf_results
        ]

        # Calculate degradation
        return_degradation = np.mean(is_returns) - np.mean(oos_returns)
        sharpe_degradation = np.mean(is_sharpes) - np.mean(oos_sharpes)

        # Calculate correlation between IS and OOS performance
        return_correlation = (
            np.corrcoef(is_returns, oos_returns)[0, 1] if len(is_returns) > 1 else 0
        )
        sharpe_correlation = (
            np.corrcoef(is_sharpes, oos_sharpes)[0, 1] if len(is_sharpes) > 1 else 0
        )

        return {
            "return_degradation": return_degradation,
            "sharpe_degradation": sharpe_degradation,
            "return_correlation": return_correlation,
            "sharpe_correlation": sharpe_correlation,
            "overfitting_score": max(
                0, 1 - abs(return_degradation) - abs(sharpe_degradation)
            ),
        }

    def statistical_significance_test(
        self,
        results1: OptimizationResult,
        results2: OptimizationResult,
        test_type: str = "t_test",
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Test statistical significance between two strategy results.

        Args:
            results1: First strategy results
            results2: Second strategy results
            test_type: Type of test ('t_test', 'wilcoxon', 'bootstrap')
            confidence_level: Confidence level for the test

        Returns:
            Dict[str, Any]: Statistical test results
        """
        # Extract returns from equity curves
        returns1 = results1.equity_curve.pct_change().dropna()
        returns2 = results2.equity_curve.pct_change().dropna()

        # Align returns to common dates
        common_index = returns1.index.intersection(returns2.index)
        if len(common_index) < 10:
            return {
                "error": "Insufficient overlapping data for statistical test",
                "common_periods": len(common_index),
            }

        returns1_aligned = returns1.loc[common_index]
        returns2_aligned = returns2.loc[common_index]

        # Calculate return differences
        return_diff = returns1_aligned - returns2_aligned

        if test_type == "t_test":
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(returns1_aligned, returns2_aligned)
            test_name = "Paired t-test"

        elif test_type == "wilcoxon":
            # Wilcoxon signed-rank test (non-parametric)
            t_stat, p_value = stats.wilcoxon(returns1_aligned, returns2_aligned)
            test_name = "Wilcoxon signed-rank test"

        elif test_type == "bootstrap":
            # Bootstrap test
            n_bootstrap = 10000
            bootstrap_diffs = []

            for _ in range(n_bootstrap):
                # Resample with replacement
                sample_indices = np.random.choice(
                    len(return_diff), len(return_diff), replace=True
                )
                bootstrap_sample = return_diff.iloc[sample_indices]
                bootstrap_diffs.append(bootstrap_sample.mean())

            # Calculate p-value
            bootstrap_diffs = np.array(bootstrap_diffs)
            p_value = 2 * min(
                np.mean(bootstrap_diffs <= 0), np.mean(bootstrap_diffs >= 0)
            )
            t_stat = return_diff.mean() / (
                return_diff.std() / np.sqrt(len(return_diff))
            )
            test_name = "Bootstrap test"

        else:
            raise ValueError(f"Unknown test type: {test_type}")

        # Calculate confidence interval
        alpha = 1 - confidence_level
        if test_type == "bootstrap":
            ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        else:
            # t-distribution confidence interval
            df = len(return_diff) - 1
            t_critical = stats.t.ppf(1 - alpha / 2, df)
            margin_error = t_critical * (return_diff.std() / np.sqrt(len(return_diff)))
            ci_lower = return_diff.mean() - margin_error
            ci_upper = return_diff.mean() + margin_error

        # Determine significance
        is_significant = p_value < (1 - confidence_level)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((returns1_aligned.var() + returns2_aligned.var()) / 2)
        cohens_d = return_diff.mean() / pooled_std if pooled_std > 0 else 0

        return {
            "test_name": test_name,
            "test_statistic": t_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "confidence_level": confidence_level,
            "mean_difference": return_diff.mean(),
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
            },
            "effect_size": cohens_d,
            "sample_size": len(common_index),
            "interpretation": self._interpret_significance_test(
                p_value, cohens_d, confidence_level
            ),
        }

    def _interpret_significance_test(
        self, p_value: float, effect_size: float, confidence_level: float
    ) -> str:
        """Interpret statistical significance test results.

        Args:
            p_value: P-value from the test
            effect_size: Effect size (Cohen's d)
            confidence_level: Confidence level used

        Returns:
            str: Interpretation of the results
        """
        alpha = 1 - confidence_level

        if p_value < alpha:
            significance = "statistically significant"
        else:
            significance = "not statistically significant"

        # Interpret effect size
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            effect_magnitude = "negligible"
        elif abs_effect < 0.5:
            effect_magnitude = "small"
        elif abs_effect < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"

        direction = "better" if effect_size > 0 else "worse"

        return (
            f"The difference is {significance} (p={p_value:.4f}). "
            f"Strategy 1 performs {direction} than Strategy 2 with a {effect_magnitude} effect size "
            f"(Cohen's d = {effect_size:.3f})."
        )

    def monte_carlo_analysis(
        self,
        strategy_func: Callable,
        parameters: Dict[str, Any],
        exchanges: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        n_simulations: int = 1000,
        bootstrap_block_size: int = 22,  # Approximately 1 month of trading days
    ) -> Dict[str, Any]:
        """Perform Monte Carlo analysis of strategy performance.

        Args:
            strategy_func: Strategy function
            parameters: Strategy parameters
            exchanges: List of exchanges
            symbols: List of symbols
            timeframe: Timeframe for backtesting
            start_date: Start date
            end_date: End date
            n_simulations: Number of Monte Carlo simulations
            bootstrap_block_size: Block size for block bootstrap

        Returns:
            Dict[str, Any]: Monte Carlo analysis results
        """
        logger.info(f"Starting Monte Carlo analysis with {n_simulations} simulations")

        # Run original backtest
        original_results = self.backtest_engine.run_backtest(
            strategy_func=strategy_func,
            exchanges=exchanges,
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy_params=parameters,
        )

        # Extract original equity curve
        original_portfolio = original_results.get("portfolio", {})
        original_equity_data = original_portfolio.get("equity_curve", [])

        if not original_equity_data:
            return {"error": "No equity curve data available"}

        timestamps = [
            datetime.fromisoformat(item["timestamp"]) for item in original_equity_data
        ]
        equity_values = [item["equity"] for item in original_equity_data]
        original_equity = pd.Series(equity_values, index=timestamps)
        original_returns = original_equity.pct_change().dropna()

        # Perform Monte Carlo simulations using block bootstrap
        simulation_results = []

        for i in range(n_simulations):
            if i % 100 == 0:
                logger.info(f"Completed {i}/{n_simulations} simulations")

            # Generate bootstrap sample using block bootstrap
            bootstrap_returns = self._block_bootstrap(
                original_returns, bootstrap_block_size
            )

            # Create synthetic equity curve
            synthetic_equity = [original_equity.iloc[0]]
            for ret in bootstrap_returns:
                synthetic_equity.append(synthetic_equity[-1] * (1 + ret))

            synthetic_equity_series = pd.Series(
                synthetic_equity[1:],
                index=original_equity.index[: len(bootstrap_returns)],
            )

            # Calculate metrics for synthetic equity curve
            synthetic_metrics = self.performance_calculator.calculate_metrics(
                synthetic_equity_series
            )

            simulation_results.append({
                "simulation": i + 1,
                "final_equity": synthetic_equity_series.iloc[-1],
                "total_return": synthetic_metrics["total_return"],
                "sharpe_ratio": synthetic_metrics["returns"]["sharpe_ratio"],
                "max_drawdown": synthetic_metrics["drawdown"]["max_drawdown"],
                "cagr": synthetic_metrics["returns"]["cagr"],
            })

        # Calculate confidence intervals
        metrics_to_analyze = ["total_return", "sharpe_ratio", "max_drawdown", "cagr"]
        confidence_intervals = {}

        for metric in metrics_to_analyze:
            values = [result[metric] for result in simulation_results]
            confidence_intervals[metric] = {
                "5th_percentile": np.percentile(values, 5),
                "25th_percentile": np.percentile(values, 25),
                "median": np.percentile(values, 50),
                "75th_percentile": np.percentile(values, 75),
                "95th_percentile": np.percentile(values, 95),
                "mean": np.mean(values),
                "std": np.std(values),
            }

        # Calculate probability of positive returns
        positive_returns = sum(
            1 for result in simulation_results if result["total_return"] > 0
        )
        prob_positive = positive_returns / n_simulations

        # Calculate Value at Risk and Expected Shortfall
        returns_dist = [result["total_return"] for result in simulation_results]
        var_95 = np.percentile(returns_dist, 5)
        var_99 = np.percentile(returns_dist, 1)

        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean([r for r in returns_dist if r <= var_95])
        es_99 = np.mean([r for r in returns_dist if r <= var_99])

        return {
            "original_metrics": original_results.get("metrics", {}),
            "simulation_results": simulation_results,
            "confidence_intervals": confidence_intervals,
            "risk_metrics": {
                "probability_positive_return": prob_positive,
                "var_95": var_95,
                "var_99": var_99,
                "expected_shortfall_95": es_95,
                "expected_shortfall_99": es_99,
            },
            "summary": {
                "n_simulations": n_simulations,
                "bootstrap_block_size": bootstrap_block_size,
                "mean_return": confidence_intervals["total_return"]["mean"],
                "return_volatility": confidence_intervals["total_return"]["std"],
            },
        }

    def _block_bootstrap(self, returns: pd.Series, block_size: int) -> pd.Series:
        """Perform block bootstrap resampling to preserve autocorrelation.

        Args:
            returns: Original returns series
            block_size: Size of each bootstrap block

        Returns:
            pd.Series: Bootstrap sample of returns
        """
        n_returns = len(returns)
        n_blocks = int(np.ceil(n_returns / block_size))

        bootstrap_returns = []

        for _ in range(n_blocks):
            # Randomly select a starting point for the block
            start_idx = np.random.randint(0, max(1, n_returns - block_size + 1))
            end_idx = min(start_idx + block_size, n_returns)

            # Add the block to bootstrap sample
            block = returns.iloc[start_idx:end_idx]
            bootstrap_returns.extend(block.values)

        # Trim to original length
        bootstrap_returns = bootstrap_returns[:n_returns]

        return pd.Series(bootstrap_returns)
