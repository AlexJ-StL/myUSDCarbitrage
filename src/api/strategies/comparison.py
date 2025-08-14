"""Strategy comparison and A/B testing functionality."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .. import models
from ..backtesting import BacktestEngine

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("strategy_comparison.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class StrategyComparison:
    """Strategy comparison and A/B testing system."""

    def __init__(self, db_session: Session, backtest_engine: BacktestEngine):
        """Initialize strategy comparison system.

        Args:
            db_session: Database session
            backtest_engine: Backtesting engine
        """
        self.db = db_session
        self.backtest_engine = backtest_engine

    def compare_strategies(
        self,
        strategy_ids: List[int],
        start_date: datetime,
        end_date: datetime,
        exchanges: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        initial_capital: float = 10000.0,
    ) -> Dict[str, Any]:
        """Compare multiple strategies using backtesting.

        Args:
            strategy_ids: List of strategy IDs to compare
            start_date: Start date for comparison
            end_date: End date for comparison
            exchanges: List of exchanges to test on
            symbols: List of symbols to test on
            timeframe: Timeframe for testing
            initial_capital: Initial capital for backtesting

        Returns:
            Dict[str, Any]: Comparison results

        Raises:
            ValueError: If strategies not found or invalid parameters
        """
        if len(strategy_ids) < 2:
            raise ValueError("At least 2 strategies required for comparison")

        # Get strategies
        strategies = []
        for strategy_id in strategy_ids:
            strategy = (
                self.db.query(models.Strategy)
                .filter(models.Strategy.id == strategy_id)
                .first()
            )
            if not strategy:
                raise ValueError(f"Strategy with ID {strategy_id} not found")
            strategies.append(strategy)

        # Set defaults
        if not exchanges:
            exchanges = ["coinbase", "kraken", "binance"]
        if not symbols:
            symbols = ["USDC/USD"]

        # Run backtests for each strategy
        results = {}
        backtest_results = []

        for strategy in strategies:
            logger.info(f"Running backtest for strategy '{strategy.name}'")

            # Get strategy function
            strategy_func = strategy.get_strategy_function()

            # Run backtest
            backtest_result = self.backtest_engine.run_backtest(
                strategy_func=strategy_func,
                exchanges=exchanges,
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy_params=strategy.parameters,
                initial_capital=initial_capital,
            )

            results[strategy.name] = backtest_result
            backtest_results.append(backtest_result)

            # Store backtest result in database
            db_result = models.BacktestResult(
                strategy_id=strategy.id,
                start_date=start_date,
                end_date=end_date,
                parameters=strategy.parameters,
                results=backtest_result,
                metrics=backtest_result.get("metrics", {}),
                status="completed",
                created_at=datetime.now(),
                completed_at=datetime.now(),
            )
            self.db.add(db_result)

        self.db.commit()

        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(results)

        # Generate comparison report
        comparison_report = self._generate_comparison_report(
            strategies, results, comparison_metrics
        )

        return {
            "strategies": [{"id": s.id, "name": s.name} for s in strategies],
            "backtest_results": results,
            "comparison_metrics": comparison_metrics,
            "comparison_report": comparison_report,
            "test_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "test_parameters": {
                "exchanges": exchanges,
                "symbols": symbols,
                "timeframe": timeframe,
                "initial_capital": initial_capital,
            },
        }

    def _calculate_comparison_metrics(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comparison metrics between strategies.

        Args:
            results: Backtest results for each strategy

        Returns:
            Dict[str, Any]: Comparison metrics
        """
        metrics = {}
        strategy_names = list(results.keys())

        # Extract key metrics for comparison
        total_returns = {}
        sharpe_ratios = {}
        max_drawdowns = {}
        win_rates = {}
        profit_factors = {}

        for name, result in results.items():
            strategy_metrics = result.get("metrics", {})
            total_returns[name] = strategy_metrics.get("total_return", 0.0)
            sharpe_ratios[name] = strategy_metrics.get("sharpe_ratio", 0.0)
            max_drawdowns[name] = strategy_metrics.get("max_drawdown", 0.0)
            win_rates[name] = strategy_metrics.get("win_rate", 0.0)
            profit_factors[name] = strategy_metrics.get("profit_factor", 0.0)

        # Rank strategies by different metrics
        metrics["rankings"] = {
            "total_return": self._rank_strategies(total_returns, reverse=True),
            "sharpe_ratio": self._rank_strategies(sharpe_ratios, reverse=True),
            "max_drawdown": self._rank_strategies(max_drawdowns, reverse=False),
            "win_rate": self._rank_strategies(win_rates, reverse=True),
            "profit_factor": self._rank_strategies(profit_factors, reverse=True),
        }

        # Calculate statistical significance
        if len(strategy_names) == 2:
            metrics["statistical_tests"] = self._perform_statistical_tests(
                results[strategy_names[0]], results[strategy_names[1]]
            )

        # Best and worst performers
        best_return = max(total_returns, key=total_returns.get)
        worst_return = min(total_returns, key=total_returns.get)
        best_sharpe = max(sharpe_ratios, key=sharpe_ratios.get)
        best_drawdown = min(max_drawdowns, key=max_drawdowns.get)

        metrics["best_performers"] = {
            "highest_return": {
                "strategy": best_return,
                "value": total_returns[best_return],
            },
            "best_sharpe_ratio": {
                "strategy": best_sharpe,
                "value": sharpe_ratios[best_sharpe],
            },
            "lowest_drawdown": {
                "strategy": best_drawdown,
                "value": max_drawdowns[best_drawdown],
            },
        }

        metrics["worst_performers"] = {
            "lowest_return": {
                "strategy": worst_return,
                "value": total_returns[worst_return],
            },
        }

        # Calculate correlation between strategies
        if len(strategy_names) >= 2:
            metrics["correlations"] = self._calculate_strategy_correlations(results)

        return metrics

    def _rank_strategies(
        self, metric_values: Dict[str, float], reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """Rank strategies by a metric.

        Args:
            metric_values: Dictionary of strategy names and metric values
            reverse: Whether to sort in descending order

        Returns:
            List[Dict[str, Any]]: Ranked list of strategies
        """
        sorted_items = sorted(
            metric_values.items(), key=lambda x: x[1], reverse=reverse
        )

        rankings = []
        for rank, (strategy, value) in enumerate(sorted_items, 1):
            rankings.append({
                "rank": rank,
                "strategy": strategy,
                "value": value,
            })

        return rankings

    def _perform_statistical_tests(
        self, result1: Dict[str, Any], result2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform statistical tests between two strategies.

        Args:
            result1: First strategy results
            result2: Second strategy results

        Returns:
            Dict[str, Any]: Statistical test results
        """
        from scipy import stats

        # Extract daily returns if available
        returns1 = result1.get("daily_returns", [])
        returns2 = result2.get("daily_returns", [])

        if not returns1 or not returns2:
            return {"error": "Daily returns not available for statistical tests"}

        # Ensure same length
        min_length = min(len(returns1), len(returns2))
        returns1 = returns1[:min_length]
        returns2 = returns2[:min_length]

        tests = {}

        try:
            # T-test for difference in means
            t_stat, t_p_value = stats.ttest_ind(returns1, returns2)
            tests["t_test"] = {
                "statistic": t_stat,
                "p_value": t_p_value,
                "significant": t_p_value < 0.05,
                "interpretation": "Returns are significantly different"
                if t_p_value < 0.05
                else "No significant difference in returns",
            }

            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(
                returns1, returns2, alternative="two-sided"
            )
            tests["mann_whitney_u"] = {
                "statistic": u_stat,
                "p_value": u_p_value,
                "significant": u_p_value < 0.05,
                "interpretation": "Distributions are significantly different"
                if u_p_value < 0.05
                else "No significant difference in distributions",
            }

            # Correlation test
            corr_coef, corr_p_value = stats.pearsonr(returns1, returns2)
            tests["correlation"] = {
                "coefficient": corr_coef,
                "p_value": corr_p_value,
                "significant": corr_p_value < 0.05,
                "interpretation": f"Strategies are {'significantly' if corr_p_value < 0.05 else 'not significantly'} correlated (r={corr_coef:.3f})",
            }

        except Exception as e:
            tests["error"] = f"Statistical tests failed: {str(e)}"

        return tests

    def _calculate_strategy_correlations(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate correlations between strategy returns.

        Args:
            results: Backtest results for each strategy

        Returns:
            Dict[str, float]: Correlation matrix
        """
        strategy_names = list(results.keys())
        correlations = {}

        for i, name1 in enumerate(strategy_names):
            for j, name2 in enumerate(strategy_names):
                if i < j:  # Only calculate upper triangle
                    returns1 = results[name1].get("daily_returns", [])
                    returns2 = results[name2].get("daily_returns", [])

                    if returns1 and returns2:
                        # Ensure same length
                        min_length = min(len(returns1), len(returns2))
                        returns1 = returns1[:min_length]
                        returns2 = returns2[:min_length]

                        # Calculate correlation
                        correlation = np.corrcoef(returns1, returns2)[0, 1]
                        correlations[f"{name1}_vs_{name2}"] = correlation

        return correlations

    def _generate_comparison_report(
        self,
        strategies: List[models.Strategy],
        results: Dict[str, Dict[str, Any]],
        comparison_metrics: Dict[str, Any],
    ) -> str:
        """Generate a text report comparing strategies.

        Args:
            strategies: List of strategies
            results: Backtest results
            comparison_metrics: Comparison metrics

        Returns:
            str: Comparison report
        """
        report = []
        report.append("STRATEGY COMPARISON REPORT")
        report.append("=" * 50)
        report.append("")

        # Strategy overview
        report.append("STRATEGIES COMPARED:")
        for strategy in strategies:
            report.append(f"- {strategy.name}: {strategy.description}")
        report.append("")

        # Performance summary
        report.append("PERFORMANCE SUMMARY:")
        report.append("-" * 30)

        for name, result in results.items():
            metrics = result.get("metrics", {})
            report.append(f"{name}:")
            report.append(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            report.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            report.append(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            report.append(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            report.append("")

        # Rankings
        report.append("RANKINGS:")
        report.append("-" * 20)

        rankings = comparison_metrics.get("rankings", {})
        for metric, ranking in rankings.items():
            report.append(f"{metric.replace('_', ' ').title()}:")
            for item in ranking:
                report.append(
                    f"  {item['rank']}. {item['strategy']} ({item['value']:.4f})"
                )
            report.append("")

        # Best performers
        best_performers = comparison_metrics.get("best_performers", {})
        if best_performers:
            report.append("BEST PERFORMERS:")
            report.append("-" * 25)
            for category, performer in best_performers.items():
                report.append(
                    f"{category.replace('_', ' ').title()}: {performer['strategy']} ({performer['value']:.4f})"
                )
            report.append("")

        # Statistical tests (if available)
        statistical_tests = comparison_metrics.get("statistical_tests", {})
        if statistical_tests and "error" not in statistical_tests:
            report.append("STATISTICAL ANALYSIS:")
            report.append("-" * 30)
            for test_name, test_result in statistical_tests.items():
                if isinstance(test_result, dict) and "interpretation" in test_result:
                    report.append(
                        f"{test_name.replace('_', ' ').title()}: {test_result['interpretation']}"
                    )
            report.append("")

        # Correlations
        correlations = comparison_metrics.get("correlations", {})
        if correlations:
            report.append("STRATEGY CORRELATIONS:")
            report.append("-" * 35)
            for pair, correlation in correlations.items():
                strategies_pair = pair.replace("_vs_", " vs ")
                report.append(f"{strategies_pair}: {correlation:.3f}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 25)

        # Find best overall strategy
        total_return_ranking = rankings.get("total_return", [])
        sharpe_ranking = rankings.get("sharpe_ratio", [])

        if total_return_ranking and sharpe_ranking:
            best_return_strategy = total_return_ranking[0]["strategy"]
            best_sharpe_strategy = sharpe_ranking[0]["strategy"]

            if best_return_strategy == best_sharpe_strategy:
                report.append(
                    f"• {best_return_strategy} shows the best overall performance with highest returns and best risk-adjusted returns."
                )
            else:
                report.append(
                    f"• {best_return_strategy} has the highest total returns."
                )
                report.append(
                    f"• {best_sharpe_strategy} has the best risk-adjusted returns."
                )
                report.append(
                    "• Consider your risk tolerance when choosing between these strategies."
                )

        # Diversification recommendation
        if correlations:
            low_corr_pairs = [
                pair for pair, corr in correlations.items() if abs(corr) < 0.5
            ]
            if low_corr_pairs:
                report.append(
                    "• Consider combining strategies with low correlations for diversification:"
                )
                for pair in low_corr_pairs[:3]:  # Show top 3
                    strategies_pair = pair.replace("_vs_", " and ")
                    report.append(f"  - {strategies_pair}")

        return "\n".join(report)

    def create_ab_test(
        self,
        strategy_a_id: int,
        strategy_b_id: int,
        test_name: str,
        description: str,
        allocation_ratio: float = 0.5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        exchanges: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
    ) -> models.ABTest:
        """Create an A/B test between two strategies.

        Args:
            strategy_a_id: ID of strategy A
            strategy_b_id: ID of strategy B
            test_name: Name of the A/B test
            description: Description of the test
            allocation_ratio: Allocation ratio for strategy A (0.0 to 1.0)
            start_date: Start date for the test
            end_date: End date for the test
            exchanges: List of exchanges to test on
            symbols: List of symbols to test on

        Returns:
            models.ABTest: Created A/B test

        Raises:
            ValueError: If strategies not found or invalid parameters
        """
        # Validate strategies
        strategy_a = (
            self.db.query(models.Strategy)
            .filter(models.Strategy.id == strategy_a_id)
            .first()
        )
        strategy_b = (
            self.db.query(models.Strategy)
            .filter(models.Strategy.id == strategy_b_id)
            .first()
        )

        if not strategy_a:
            raise ValueError(f"Strategy A with ID {strategy_a_id} not found")
        if not strategy_b:
            raise ValueError(f"Strategy B with ID {strategy_b_id} not found")

        if not 0.0 <= allocation_ratio <= 1.0:
            raise ValueError("Allocation ratio must be between 0.0 and 1.0")

        # Create A/B test record
        ab_test = models.ABTest(
            name=test_name,
            description=description,
            strategy_a_id=strategy_a_id,
            strategy_b_id=strategy_b_id,
            allocation_ratio=allocation_ratio,
            start_date=start_date or datetime.now(),
            end_date=end_date,
            exchanges=exchanges or ["coinbase", "kraken", "binance"],
            symbols=symbols or ["USDC/USD"],
            status="created",
            created_at=datetime.now(),
        )

        self.db.add(ab_test)
        self.db.commit()
        self.db.refresh(ab_test)

        logger.info(
            f"Created A/B test '{test_name}' between strategies {strategy_a.name} and {strategy_b.name}"
        )
        return ab_test

    def run_ab_test(self, ab_test_id: int) -> Dict[str, Any]:
        """Run an A/B test.

        Args:
            ab_test_id: A/B test ID

        Returns:
            Dict[str, Any]: A/B test results

        Raises:
            ValueError: If A/B test not found
        """
        # Get A/B test
        ab_test = (
            self.db.query(models.ABTest).filter(models.ABTest.id == ab_test_id).first()
        )
        if not ab_test:
            raise ValueError(f"A/B test with ID {ab_test_id} not found")

        # Update status
        ab_test.status = "running"
        ab_test.started_at = datetime.now()
        self.db.commit()

        try:
            # Run comparison between the two strategies
            comparison_results = self.compare_strategies(
                strategy_ids=[ab_test.strategy_a_id, ab_test.strategy_b_id],
                start_date=ab_test.start_date,
                end_date=ab_test.end_date or datetime.now(),
                exchanges=ab_test.exchanges,
                symbols=ab_test.symbols,
            )

            # Calculate A/B test specific metrics
            ab_results = self._calculate_ab_test_results(
                comparison_results, ab_test.allocation_ratio
            )

            # Update A/B test with results
            ab_test.results = ab_results
            ab_test.status = "completed"
            ab_test.completed_at = datetime.now()
            self.db.commit()

            logger.info(f"Completed A/B test '{ab_test.name}'")
            return ab_results

        except Exception as e:
            # Update status on failure
            ab_test.status = "failed"
            ab_test.error_message = str(e)
            ab_test.completed_at = datetime.now()
            self.db.commit()
            raise

    def _calculate_ab_test_results(
        self, comparison_results: Dict[str, Any], allocation_ratio: float
    ) -> Dict[str, Any]:
        """Calculate A/B test specific results.

        Args:
            comparison_results: Strategy comparison results
            allocation_ratio: Allocation ratio for strategy A

        Returns:
            Dict[str, Any]: A/B test results
        """
        backtest_results = comparison_results["backtest_results"]
        strategy_names = list(backtest_results.keys())

        if len(strategy_names) != 2:
            raise ValueError("A/B test requires exactly 2 strategies")

        strategy_a_name = strategy_names[0]
        strategy_b_name = strategy_names[1]

        strategy_a_results = backtest_results[strategy_a_name]
        strategy_b_results = backtest_results[strategy_b_name]

        # Calculate blended performance
        a_return = strategy_a_results.get("metrics", {}).get("total_return", 0.0)
        b_return = strategy_b_results.get("metrics", {}).get("total_return", 0.0)

        blended_return = allocation_ratio * a_return + (1 - allocation_ratio) * b_return

        # Calculate statistical significance
        statistical_tests = comparison_results["comparison_metrics"].get(
            "statistical_tests", {}
        )

        # Determine winner
        winner = strategy_a_name if a_return > b_return else strategy_b_name
        confidence = (
            "high"
            if statistical_tests.get("t_test", {}).get("significant", False)
            else "low"
        )

        return {
            "strategy_a": {
                "name": strategy_a_name,
                "allocation": allocation_ratio,
                "performance": strategy_a_results.get("metrics", {}),
            },
            "strategy_b": {
                "name": strategy_b_name,
                "allocation": 1 - allocation_ratio,
                "performance": strategy_b_results.get("metrics", {}),
            },
            "blended_performance": {
                "total_return": blended_return,
                "allocation_ratio": allocation_ratio,
            },
            "winner": {
                "strategy": winner,
                "confidence": confidence,
                "margin": abs(a_return - b_return),
            },
            "statistical_tests": statistical_tests,
            "recommendation": self._generate_ab_test_recommendation(
                strategy_a_name,
                strategy_b_name,
                a_return,
                b_return,
                allocation_ratio,
                statistical_tests,
            ),
        }

    def _generate_ab_test_recommendation(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        a_return: float,
        b_return: float,
        allocation_ratio: float,
        statistical_tests: Dict[str, Any],
    ) -> str:
        """Generate A/B test recommendation.

        Args:
            strategy_a_name: Name of strategy A
            strategy_b_name: Name of strategy B
            a_return: Strategy A return
            b_return: Strategy B return
            allocation_ratio: Allocation ratio
            statistical_tests: Statistical test results

        Returns:
            str: Recommendation text
        """
        winner = strategy_a_name if a_return > b_return else strategy_b_name
        margin = abs(a_return - b_return)
        is_significant = statistical_tests.get("t_test", {}).get("significant", False)

        if is_significant:
            if margin > 0.05:  # 5% difference
                return f"Strong recommendation: Use {winner} exclusively. It significantly outperforms with a {margin:.2%} advantage."
            else:
                return f"Moderate recommendation: {winner} shows statistically significant but modest outperformance. Consider gradual transition."
        else:
            if margin < 0.02:  # 2% difference
                return f"No clear winner: Performance difference is minimal ({margin:.2%}). Consider maintaining current allocation or other factors."
            else:
                return f"Weak recommendation: {winner} shows better performance ({margin:.2%}) but not statistically significant. Monitor longer or increase sample size."

    def get_ab_test_results(self, ab_test_id: int) -> Optional[Dict[str, Any]]:
        """Get A/B test results.

        Args:
            ab_test_id: A/B test ID

        Returns:
            Optional[Dict[str, Any]]: A/B test results if available
        """
        ab_test = (
            self.db.query(models.ABTest).filter(models.ABTest.id == ab_test_id).first()
        )

        if not ab_test:
            return None

        return {
            "id": ab_test.id,
            "name": ab_test.name,
            "description": ab_test.description,
            "status": ab_test.status,
            "created_at": ab_test.created_at,
            "started_at": ab_test.started_at,
            "completed_at": ab_test.completed_at,
            "results": ab_test.results,
            "error_message": ab_test.error_message,
        }

    def list_ab_tests(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """List A/B tests.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List[Dict[str, Any]]: List of A/B tests
        """
        ab_tests = (
            self.db.query(models.ABTest)
            .order_by(models.ABTest.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

        return [
            {
                "id": test.id,
                "name": test.name,
                "description": test.description,
                "status": test.status,
                "created_at": test.created_at,
                "started_at": test.started_at,
                "completed_at": test.completed_at,
            }
            for test in ab_tests
        ]
