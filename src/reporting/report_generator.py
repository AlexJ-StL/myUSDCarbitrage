"""
Report generator module for creating automated reports on arbitrage opportunities and strategy performance.

This module provides functionality to generate HTML reports with detailed analysis
of arbitrage opportunities and strategy performance metrics.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import jinja2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy.orm import Session

from src.api.visualization.performance_visualization import PerformanceVisualization
from src.api.visualization.portfolio_analytics import PortfolioAnalytics
from src.api.visualization.risk_analysis import RiskAnalysis
from src.reporting.jinja_filters import setup_jinja_filters

# Configure logging
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Report generator for creating automated reports on arbitrage opportunities and strategy performance.

    This class provides methods to generate HTML reports with detailed analysis
    of arbitrage opportunities and strategy performance metrics.
    """

    def __init__(self):
        """Initialize the report generator with visualization components."""
        self.performance_viz = PerformanceVisualization()
        self.portfolio_analytics = PortfolioAnalytics()
        self.risk_analysis = RiskAnalysis()

        # Set up Jinja2 environment for templating
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("src/reporting/templates"),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

        # Set up custom filters
        setup_jinja_filters(self.template_env)

    def generate_strategy_performance_report(
        self,
        backtest_result: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None,
        include_sections: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a comprehensive HTML report for strategy performance.

        Args:
            backtest_result: Dictionary containing backtest results
            benchmark_data: Optional benchmark data for comparison
            include_sections: Optional list of sections to include in the report
                             (defaults to all sections)

        Returns:
            str: HTML report content
        """
        try:
            # Default sections to include all
            if include_sections is None:
                include_sections = [
                    "executive_summary",
                    "performance_metrics",
                    "equity_curve",
                    "drawdown_analysis",
                    "monthly_returns",
                    "trade_analysis",
                    "risk_metrics",
                ]

            # Generate visualizations
            visualizations = {}

            # Get performance dashboard
            if any(
                section in include_sections
                for section in [
                    "equity_curve",
                    "drawdown_analysis",
                    "monthly_returns",
                    "rolling_returns",
                ]
            ):
                performance_dashboard = (
                    self.performance_viz.create_performance_dashboard(
                        backtest_result, benchmark_data
                    )
                )
                visualizations.update(performance_dashboard)

            # Get portfolio analytics
            if any(
                section in include_sections
                for section in [
                    "portfolio_composition",
                    "position_history",
                    "exchange_exposure",
                    "portfolio_metrics",
                ]
            ):
                portfolio_dashboard = (
                    self.portfolio_analytics.create_portfolio_dashboard(backtest_result)
                )
                visualizations.update(portfolio_dashboard)

            # Extract metrics for executive summary
            metrics = backtest_result.get("metrics", {})

            # Prepare data for the template
            template_data = {
                "report_title": f"Strategy Performance Report: {backtest_result.get('strategy_name', 'Unknown Strategy')}",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "strategy_name": backtest_result.get(
                    "strategy_name", "Unknown Strategy"
                ),
                "backtest_period": f"{backtest_result.get('start_date', 'Unknown')} to {backtest_result.get('end_date', 'Unknown')}",
                "metrics": metrics,
                "visualizations": visualizations,
                "include_sections": include_sections,
                "has_benchmark": benchmark_data is not None,
                "benchmark_name": benchmark_data.get("name", "Benchmark")
                if benchmark_data
                else None,
            }

            # Render the template
            template = self.template_env.get_template(
                "strategy_performance_report.html"
            )
            html_report = template.render(**template_data)

            return html_report

        except Exception as e:
            logger.error(f"Error generating strategy performance report: {e}")
            # Return a simple error report
            return f"<html><body><h1>Error Generating Report</h1><p>{str(e)}</p></body></html>"

    def generate_arbitrage_report(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        exchanges: List[str],
        symbol: str = "USDC/USD",
        threshold: float = 0.001,
    ) -> str:
        """
        Generate an HTML report analyzing arbitrage opportunities.

        Args:
            db: Database session
            start_time: Start time for the analysis period
            end_time: End time for the analysis period
            exchanges: List of exchanges to compare
            symbol: Trading symbol to analyze
            threshold: Minimum percentage difference to be considered an opportunity

        Returns:
            str: HTML report content
        """
        try:
            # Query data from database
            data = self._query_arbitrage_data(
                db, start_time, end_time, exchanges, symbol
            )

            # Analyze opportunities
            opportunities = self._analyze_arbitrage_opportunities(
                data, exchanges, threshold
            )

            # Calculate metrics
            metrics = self._calculate_arbitrage_metrics(opportunities, data)

            # Generate visualizations
            visualizations = self._generate_arbitrage_visualizations(
                data, opportunities
            )

            # Prepare data for the template
            template_data = {
                "report_title": f"Arbitrage Opportunity Report: {symbol}",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "exchanges": exchanges,
                "period": f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
                "threshold": threshold * 100,  # Convert to percentage
                "metrics": metrics,
                "opportunities": opportunities[:10],  # Top 10 opportunities
                "visualizations": visualizations,
            }

            # Render the template
            template = self.template_env.get_template("arbitrage_report.html")
            html_report = template.render(**template_data)

            return html_report

        except Exception as e:
            logger.error(f"Error generating arbitrage report: {e}")
            # Return a simple error report
            return f"<html><body><h1>Error Generating Report</h1><p>{str(e)}</p></body></html>"

    def _query_arbitrage_data(
        self,
        db: Session,
        start_time: datetime,
        end_time: datetime,
        exchanges: List[str],
        symbol: str,
    ) -> pd.DataFrame:
        """
        Query market data from the database for arbitrage analysis.

        Args:
            db: Database session
            start_time: Start time for the query
            end_time: End time for the query
            exchanges: List of exchanges to query
            symbol: Trading symbol to query

        Returns:
            pd.DataFrame: DataFrame containing market data
        """
        # This is a placeholder implementation
        # In a real implementation, this would query the database

        # Example SQL query:
        # SELECT exchange, timestamp, close
        # FROM market_data
        # WHERE exchange IN ('exchange1', 'exchange2', ...)
        # AND symbol = 'USDC/USD'
        # AND timestamp BETWEEN start_time AND end_time
        # ORDER BY timestamp

        # For now, generate some sample data
        timestamps = pd.date_range(start=start_time, end=end_time, freq="1H")
        data = []

        for exchange in exchanges:
            base_price = 1.0  # USDC/USD typically trades around 1.0

            for ts in timestamps:
                # Add some random variation to prices
                price = base_price + np.random.normal(0, 0.001)

                data.append({"exchange": exchange, "timestamp": ts, "close": price})

        return pd.DataFrame(data)

    def _analyze_arbitrage_opportunities(
        self, data: pd.DataFrame, exchanges: List[str], threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Analyze data to identify arbitrage opportunities.

        Args:
            data: DataFrame containing market data
            exchanges: List of exchanges to compare
            threshold: Minimum percentage difference to be considered an opportunity

        Returns:
            List[Dict[str, Any]]: List of identified opportunities
        """
        opportunities = []

        # Pivot data to have exchanges as columns
        pivot_data = data.pivot_table(
            index="timestamp", columns="exchange", values="close"
        )

        # Iterate through each timestamp
        for timestamp, row in pivot_data.iterrows():
            # Check each pair of exchanges
            for i, exchange1 in enumerate(exchanges):
                for exchange2 in exchanges[i + 1 :]:
                    if pd.isna(row[exchange1]) or pd.isna(row[exchange2]):
                        continue

                    price1 = row[exchange1]
                    price2 = row[exchange2]

                    # Calculate price difference
                    diff = abs(price1 - price2)
                    pct_diff = diff / min(price1, price2)

                    # Check if difference exceeds threshold
                    if pct_diff >= threshold:
                        # Determine buy and sell exchanges
                        buy_exchange = exchange1 if price1 < price2 else exchange2
                        sell_exchange = exchange2 if price1 < price2 else exchange1
                        buy_price = min(price1, price2)
                        sell_price = max(price1, price2)

                        opportunities.append({
                            "timestamp": timestamp,
                            "buy_exchange": buy_exchange,
                            "sell_exchange": sell_exchange,
                            "buy_price": buy_price,
                            "sell_price": sell_price,
                            "diff": diff,
                            "pct_diff": pct_diff * 100,  # Convert to percentage
                            "profit_potential": diff,  # Simplified, would need to account for fees
                        })

        # Sort opportunities by profit potential
        opportunities.sort(key=lambda x: x["profit_potential"], reverse=True)

        return opportunities

    def _calculate_arbitrage_metrics(
        self, opportunities: List[Dict[str, Any]], data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate metrics for arbitrage opportunities.

        Args:
            opportunities: List of identified opportunities
            data: DataFrame containing market data

        Returns:
            Dict[str, Any]: Dictionary of metrics
        """
        if not opportunities:
            return {
                "total_opportunities": 0,
                "avg_profit_potential": 0,
                "max_profit_potential": 0,
                "total_profit_potential": 0,
                "opportunity_frequency": 0,
            }

        # Calculate metrics
        total_opportunities = len(opportunities)
        avg_profit_potential = (
            sum(op["profit_potential"] for op in opportunities) / total_opportunities
        )
        max_profit_potential = max(op["profit_potential"] for op in opportunities)
        total_profit_potential = sum(op["profit_potential"] for op in opportunities)

        # Calculate opportunity frequency (opportunities per day)
        total_days = (
            data["timestamp"].max() - data["timestamp"].min()
        ).total_seconds() / (24 * 3600)
        opportunity_frequency = (
            total_opportunities / total_days if total_days > 0 else 0
        )

        # Exchange pair metrics
        exchange_pairs = {}
        for op in opportunities:
            pair = f"{op['buy_exchange']}-{op['sell_exchange']}"
            if pair not in exchange_pairs:
                exchange_pairs[pair] = {"count": 0, "total_profit": 0, "avg_profit": 0}

            exchange_pairs[pair]["count"] += 1
            exchange_pairs[pair]["total_profit"] += op["profit_potential"]

        # Calculate average profit for each pair
        for pair in exchange_pairs:
            exchange_pairs[pair]["avg_profit"] = (
                exchange_pairs[pair]["total_profit"] / exchange_pairs[pair]["count"]
            )

        # Sort exchange pairs by total profit
        top_pairs = sorted(
            exchange_pairs.items(), key=lambda x: x[1]["total_profit"], reverse=True
        )[:5]  # Top 5 pairs

        return {
            "total_opportunities": total_opportunities,
            "avg_profit_potential": avg_profit_potential,
            "max_profit_potential": max_profit_potential,
            "total_profit_potential": total_profit_potential,
            "opportunity_frequency": opportunity_frequency,
            "top_exchange_pairs": top_pairs,
        }

    def _generate_arbitrage_visualizations(
        self, data: pd.DataFrame, opportunities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate visualizations for arbitrage report.

        Args:
            data: DataFrame containing market data
            opportunities: List of identified opportunities

        Returns:
            Dict[str, Any]: Dictionary of visualization figures
        """
        visualizations = {}

        # Price comparison chart
        visualizations["price_comparison"] = self._create_price_comparison_chart(data)

        # Opportunity distribution chart
        visualizations["opportunity_distribution"] = (
            self._create_opportunity_distribution_chart(opportunities)
        )

        # Profit potential over time chart
        visualizations["profit_potential"] = self._create_profit_potential_chart(
            opportunities
        )

        return visualizations

    def _create_price_comparison_chart(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create price comparison chart for different exchanges.

        Args:
            data: DataFrame containing market data

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Pivot data to have exchanges as columns
            pivot_data = data.pivot_table(
                index="timestamp", columns="exchange", values="close"
            )

            # Create figure
            fig = go.Figure()

            # Add line for each exchange
            for exchange in pivot_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=pivot_data.index,
                        y=pivot_data[exchange],
                        mode="lines",
                        name=exchange,
                        hovertemplate=f"<b>{exchange}</b><br>Time: %{{x}}<br>Price: %{{y:.6f}}<extra></extra>",
                    )
                )

            # Update layout
            fig.update_layout(
                title="Price Comparison Across Exchanges",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                height=500,
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating price comparison chart: {e}")
            return {"error": str(e)}

    def _create_opportunity_distribution_chart(
        self, opportunities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create chart showing distribution of arbitrage opportunities.

        Args:
            opportunities: List of identified opportunities

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            if not opportunities:
                return {"error": "No opportunities to visualize"}

            # Convert to DataFrame
            df = pd.DataFrame(opportunities)

            # Group by exchange pairs
            df["exchange_pair"] = df.apply(
                lambda x: f"{x['buy_exchange']} → {x['sell_exchange']}", axis=1
            )

            pair_counts = df["exchange_pair"].value_counts().reset_index()
            pair_counts.columns = ["exchange_pair", "count"]

            # Create figure
            fig = go.Figure()

            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=pair_counts["exchange_pair"],
                    y=pair_counts["count"],
                    text=pair_counts["count"],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Opportunities: %{y}<extra></extra>",
                )
            )

            # Update layout
            fig.update_layout(
                title="Arbitrage Opportunities by Exchange Pair",
                xaxis_title="Exchange Pair",
                yaxis_title="Number of Opportunities",
                margin=dict(l=40, r=40, t=60, b=40),
                height=400,
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating opportunity distribution chart: {e}")
            return {"error": str(e)}

    def _create_profit_potential_chart(
        self, opportunities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create chart showing profit potential over time.

        Args:
            opportunities: List of identified opportunities

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            if not opportunities:
                return {"error": "No opportunities to visualize"}

            # Convert to DataFrame
            df = pd.DataFrame(opportunities)

            # Create figure
            fig = go.Figure()

            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["profit_potential"],
                    mode="markers",
                    marker=dict(
                        size=df["pct_diff"] * 2,  # Size based on percentage difference
                        color=df["pct_diff"],
                        colorscale="Viridis",
                        colorbar=dict(title="Difference (%)"),
                        showscale=True,
                    ),
                    hovertemplate=(
                        "<b>Time:</b> %{x}<br>"
                        "<b>Profit Potential:</b> $%{y:.6f}<br>"
                        "<b>Buy:</b> %{customdata[0]} at $%{customdata[1]:.6f}<br>"
                        "<b>Sell:</b> %{customdata[2]} at $%{customdata[3]:.6f}<br>"
                        "<b>Difference:</b> %{customdata[4]:.4f}%<extra></extra>"
                    ),
                    customdata=np.column_stack((
                        df["buy_exchange"],
                        df["buy_price"],
                        df["sell_exchange"],
                        df["sell_price"],
                        df["pct_diff"],
                    )),
                )
            )

            # Update layout
            fig.update_layout(
                title="Arbitrage Profit Potential Over Time",
                xaxis_title="Time",
                yaxis_title="Profit Potential (USD)",
                margin=dict(l=40, r=40, t=60, b=40),
                height=500,
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating profit potential chart: {e}")
            return {"error": str(e)}


# Create a singleton instance for use in the API
report_generator = ReportGenerator()


def generate_arbitrage_report(
    db: Session,
    start_time: datetime,
    end_time: datetime,
    exchanges: List[str],
    symbol: str = "USDC/USD",
    threshold: float = 0.001,
) -> str:
    """
    Generate an HTML report analyzing arbitrage opportunities.

    Args:
        db: Database session
        start_time: Start time for the analysis period
        end_time: End time for the analysis period
        exchanges: List of exchanges to compare
        symbol: Trading symbol to analyze
        threshold: Minimum percentage difference to be considered an opportunity

    Returns:
        str: HTML report content
    """
    return report_generator.generate_arbitrage_report(
        db, start_time, end_time, exchanges, symbol, threshold
    )


def generate_strategy_performance_report(
    backtest_result: Dict[str, Any],
    benchmark_data: Optional[Dict[str, Any]] = None,
    include_sections: Optional[List[str]] = None,
) -> str:
    """
    Generate a comprehensive HTML report for strategy performance.

    Args:
        backtest_result: Dictionary containing backtest results
        benchmark_data: Optional benchmark data for comparison
        include_sections: Optional list of sections to include in the report
                         (defaults to all sections)

    Returns:
        str: HTML report content
    """
    return report_generator.generate_strategy_performance_report(
        backtest_result, benchmark_data, include_sections
    )
