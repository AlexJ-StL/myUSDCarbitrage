"""Comparative analysis tools for strategy benchmarking."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logger = logging.getLogger(__name__)


class StrategyComparison:
    """Comparative analysis tools for strategy benchmarking."""

    def __init__(self):
        """Initialize strategy comparison tools."""
        pass

    def create_equity_comparison_chart(
        self,
        strategy_results: List[Dict[str, Any]],
        benchmark_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create equity curve comparison chart.

        Args:
            strategy_results: List of strategy results
            benchmark_results: Optional list of benchmark results

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Create figure
            fig = go.Figure()

            # Process strategy results
            for strategy in strategy_results:
                if "equity_curve" in strategy and strategy["equity_curve"]:
                    df = pd.DataFrame(strategy["equity_curve"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp")

                    # Normalize to 100 at start
                    initial_equity = df["equity"].iloc[0]
                    df["normalized_equity"] = df["equity"] / initial_equity * 100

                    # Add trace
                    fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["normalized_equity"],
                            mode="lines",
                            name=strategy["name"],
                            hovertemplate="<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>",
                        )
                    )

            # Process benchmark results if provided
            if benchmark_results:
                for benchmark in benchmark_results:
                    if "data" in benchmark and benchmark["data"]:
                        df = pd.DataFrame(benchmark["data"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.sort_values("timestamp")

                        # Normalize to 100 at start
                        initial_value = df["value"].iloc[0]
                        df["normalized_value"] = df["value"] / initial_value * 100

                        # Add trace
                        fig.add_trace(
                            go.Scatter(
                                x=df["timestamp"],
                                y=df["normalized_value"],
                                mode="lines",
                                name=benchmark["name"],
                                line=dict(dash="dash"),
                                hovertemplate="<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>",
                            )
                        )

            # Update layout
            fig.update_layout(
                title="Strategy Performance Comparison (Normalized to 100)",
                xaxis_title="Date",
                yaxis_title="Value (Normalized)",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                height=500,
            )

            # Add range selector
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ])
                ),
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating equity comparison chart: {e}")
            return {"error": str(e)}

    def create_drawdown_comparison_chart(
        self,
        strategy_results: List[Dict[str, Any]],
        benchmark_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create drawdown comparison chart.

        Args:
            strategy_results: List of strategy results
            benchmark_results: Optional list of benchmark results

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Create figure
            fig = go.Figure()

            # Process strategy results
            for strategy in strategy_results:
                if "equity_curve" in strategy and strategy["equity_curve"]:
                    df = pd.DataFrame(strategy["equity_curve"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp")

                    # Calculate drawdown
                    df["peak"] = df["equity"].cummax()
                    df["drawdown"] = (
                        df["equity"] / df["peak"] - 1
                    ) * 100  # Convert to percentage

                    # Add trace
                    fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["drawdown"],
                            mode="lines",
                            name=strategy["name"],
                            hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
                        )
                    )

            # Process benchmark results if provided
            if benchmark_results:
                for benchmark in benchmark_results:
                    if "data" in benchmark and benchmark["data"]:
                        df = pd.DataFrame(benchmark["data"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.sort_values("timestamp")

                        # Calculate drawdown
                        df["peak"] = df["value"].cummax()
                        df["drawdown"] = (
                            df["value"] / df["peak"] - 1
                        ) * 100  # Convert to percentage

                        # Add trace
                        fig.add_trace(
                            go.Scatter(
                                x=df["timestamp"],
                                y=df["drawdown"],
                                mode="lines",
                                name=benchmark["name"],
                                line=dict(dash="dash"),
                                hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
                            )
                        )

            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="solid",
                line_color="black",
                line_width=1,
            )

            # Update layout
            fig.update_layout(
                title="Drawdown Comparison",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                height=500,
            )

            # Add range selector
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ])
                ),
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating drawdown comparison chart: {e}")
            return {"error": str(e)}

    def create_rolling_returns_comparison(
        self,
        strategy_results: List[Dict[str, Any]],
        window_days: int = 30,
        benchmark_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create rolling returns comparison chart.

        Args:
            strategy_results: List of strategy results
            window_days: Rolling window size in days
            benchmark_results: Optional list of benchmark results

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Create figure
            fig = go.Figure()

            # Process strategy results
            for strategy in strategy_results:
                if "equity_curve" in strategy and strategy["equity_curve"]:
                    df = pd.DataFrame(strategy["equity_curve"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp")

                    # Calculate daily returns
                    df["daily_return"] = df["equity"].pct_change()

                    # Calculate rolling returns
                    df["rolling_return"] = (
                        (1 + df["daily_return"])
                        .rolling(window=window_days)
                        .apply(lambda x: x.prod() - 1, raw=True)
                        * 100  # Convert to percentage
                    )

                    # Add trace
                    fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["rolling_return"],
                            mode="lines",
                            name=strategy["name"],
                            hovertemplate="<b>Date:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>",
                        )
                    )

            # Process benchmark results if provided
            if benchmark_results:
                for benchmark in benchmark_results:
                    if "data" in benchmark and benchmark["data"]:
                        df = pd.DataFrame(benchmark["data"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.sort_values("timestamp")

                        # Calculate daily returns
                        df["daily_return"] = df["value"].pct_change()

                        # Calculate rolling returns
                        df["rolling_return"] = (
                            (1 + df["daily_return"])
                            .rolling(window=window_days)
                            .apply(lambda x: x.prod() - 1, raw=True)
                            * 100  # Convert to percentage
                        )

                        # Add trace
                        fig.add_trace(
                            go.Scatter(
                                x=df["timestamp"],
                                y=df["rolling_return"],
                                mode="lines",
                                name=benchmark["name"],
                                line=dict(dash="dash"),
                                hovertemplate="<b>Date:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>",
                            )
                        )

            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="solid",
                line_color="black",
                line_width=1,
            )

            # Update layout
            fig.update_layout(
                title=f"{window_days}-Day Rolling Returns Comparison",
                xaxis_title="Date",
                yaxis_title="Rolling Return (%)",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                height=500,
            )

            # Add range selector
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ])
                ),
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating rolling returns comparison chart: {e}")
            return {"error": str(e)}

    def create_metrics_comparison_table(
        self,
        strategy_results: List[Dict[str, Any]],
        benchmark_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create metrics comparison table.

        Args:
            strategy_results: List of strategy results
            benchmark_results: Optional list of benchmark results

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Define metrics to include
            metrics_to_include = [
                ("total_return", "Total Return (%)", 100),
                ("cagr", "CAGR (%)", 100),
                ("sharpe_ratio", "Sharpe Ratio", 1),
                ("sortino_ratio", "Sortino Ratio", 1),
                ("max_drawdown", "Max Drawdown (%)", 100),
                ("calmar_ratio", "Calmar Ratio", 1),
                ("volatility", "Volatility (%)", 100),
                ("win_rate", "Win Rate (%)", 100),
                ("profit_factor", "Profit Factor", 1),
                ("avg_trade", "Avg Trade (%)", 100),
            ]

            # Prepare data
            table_data = []

            # Add header row
            header = ["Metric"]
            for strategy in strategy_results:
                header.append(strategy["name"])

            if benchmark_results:
                for benchmark in benchmark_results:
                    header.append(benchmark["name"])

            table_data.append(header)

            # Add metric rows
            for metric_key, metric_name, scale in metrics_to_include:
                row = [metric_name]

                # Add strategy values
                for strategy in strategy_results:
                    metrics = strategy.get("metrics", {})
                    value = metrics.get(metric_key, None)

                    if value is not None:
                        if metric_key == "max_drawdown":
                            # Drawdown is negative, but display as positive percentage
                            row.append(f"{abs(value * scale):.2f}%")
                        elif scale == 100:
                            row.append(f"{value * scale:.2f}%")
                        else:
                            row.append(f"{value * scale:.2f}")
                    else:
                        row.append("N/A")

                # Add benchmark values
                if benchmark_results:
                    for benchmark in benchmark_results:
                        metrics = benchmark.get("metrics", {})
                        value = metrics.get(metric_key, None)

                        if value is not None:
                            if metric_key == "max_drawdown":
                                # Drawdown is negative, but display as positive percentage
                                row.append(f"{abs(value * scale):.2f}%")
                            elif scale == 100:
                                row.append(f"{value * scale:.2f}%")
                            else:
                                row.append(f"{value * scale:.2f}")
                        else:
                            row.append("N/A")

                table_data.append(row)

            # Create table
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=table_data[0],
                            fill_color="paleturquoise",
                            align="left",
                            font=dict(size=12),
                        ),
                        cells=dict(
                            values=[row for row in zip(*table_data[1:])],
                            fill_color="lavender",
                            align="right",
                            font=dict(size=11),
                        ),
                    )
                ]
            )

            # Update layout
            fig.update_layout(
                title="Strategy Performance Metrics Comparison",
                margin=dict(l=10, r=10, t=40, b=10),
                height=400,
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating metrics comparison table: {e}")
            return {"error": str(e)}

    def create_monthly_returns_heatmap_comparison(
        self, strategy_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create monthly returns heatmap comparison.

        Args:
            strategy_results: List of strategy results

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            if not strategy_results:
                return {"error": "No strategy results provided"}

            # Create subplots
            fig = make_subplots(
                rows=len(strategy_results),
                cols=1,
                subplot_titles=[strategy["name"] for strategy in strategy_results],
                vertical_spacing=0.1,
            )

            # Process each strategy
            for i, strategy in enumerate(strategy_results):
                if "equity_curve" in strategy and strategy["equity_curve"]:
                    df = pd.DataFrame(strategy["equity_curve"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")

                    # Calculate daily returns
                    df["daily_return"] = df["equity"].pct_change()

                    # Resample to get monthly returns
                    monthly_returns = (
                        df["daily_return"]
                        .resample("M")
                        .apply(lambda x: (1 + x).prod() - 1)
                    )

                    # Create a pivot table for the heatmap
                    monthly_pivot = pd.DataFrame(monthly_returns)
                    monthly_pivot["year"] = monthly_pivot.index.year
                    monthly_pivot["month"] = monthly_pivot.index.month
                    monthly_pivot = monthly_pivot.pivot_table(
                        index="year", columns="month", values="daily_return"
                    )

                    # Create month labels
                    month_labels = [
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ]

                    # Add heatmap
                    fig.add_trace(
                        go.Heatmap(
                            z=monthly_pivot.values * 100,  # Convert to percentage
                            x=month_labels,
                            y=monthly_pivot.index,
                            colorscale=[
                                [0, "rgb(165,0,38)"],  # Red for negative
                                [0.5, "rgb(255,255,255)"],  # White for zero
                                [1, "rgb(0,104,55)"],  # Green for positive
                            ],
                            zmid=0,  # Center the color scale at zero
                            text=np.round(monthly_pivot.values * 100, 2),
                            hovertemplate="<b>Year:</b> %{y}<br><b>Month:</b> %{x}<br><b>Return:</b> %{z:.2f}%<extra></extra>",
                            texttemplate="%{text:.2f}%",
                            showscale=i == 0,  # Only show colorbar for first heatmap
                        ),
                        row=i + 1,
                        col=1,
                    )

                    # Update y-axis
                    fig.update_yaxes(
                        title_text="Year",
                        autorange="reversed",  # Latest year at the top
                        row=i + 1,
                        col=1,
                    )

            # Update layout
            fig.update_layout(
                title="Monthly Returns Comparison (%)",
                xaxis_title="Month",
                height=300 * len(strategy_results),
                margin=dict(l=40, r=40, t=60, b=40),
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating monthly returns heatmap comparison: {e}")
            return {"error": str(e)}

    def create_statistical_significance_test(
        self,
        strategy_results: List[Dict[str, Any]],
        benchmark_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create statistical significance test visualization.

        Args:
            strategy_results: List of strategy results
            benchmark_results: Optional list of benchmark results

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            import scipy.stats as stats

            # Prepare data
            returns_data = {}

            # Process strategy results
            for strategy in strategy_results:
                if "equity_curve" in strategy and strategy["equity_curve"]:
                    df = pd.DataFrame(strategy["equity_curve"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                    df["return"] = df["equity"].pct_change().fillna(0)
                    returns_data[strategy["name"]] = df["return"]

            # Process benchmark results if provided
            if benchmark_results:
                for benchmark in benchmark_results:
                    if "data" in benchmark and benchmark["data"]:
                        df = pd.DataFrame(benchmark["data"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.set_index("timestamp")
                        df["return"] = df["value"].pct_change().fillna(0)
                        returns_data[benchmark["name"]] = df["return"]

            # Create statistical test results
            if len(returns_data) > 1:
                # Create a DataFrame with all returns
                returns_df = pd.DataFrame(returns_data)

                # Perform t-tests between each pair
                names = list(returns_data.keys())
                n = len(names)
                p_values = np.zeros((n, n))
                t_stats = np.zeros((n, n))

                for i in range(n):
                    for j in range(n):
                        if i != j:
                            # Perform t-test
                            t_stat, p_value = stats.ttest_ind(
                                returns_df[names[i]].dropna(),
                                returns_df[names[j]].dropna(),
                                equal_var=False,  # Welch's t-test
                            )
                            p_values[i, j] = p_value
                            t_stats[i, j] = t_stat

                # Create heatmap for p-values
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=("P-Values", "T-Statistics"),
                    horizontal_spacing=0.1,
                )

                # Add p-value heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=p_values,
                        x=names,
                        y=names,
                        colorscale="Viridis",
                        text=np.round(p_values, 4),
                        texttemplate="%{text:.4f}",
                        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>P-Value: %{z:.4f}<extra></extra>",
                        colorbar=dict(title="P-Value", x=0.45),
                    ),
                    row=1,
                    col=1,
                )

                # Add t-statistic heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=t_stats,
                        x=names,
                        y=names,
                        colorscale="RdBu",
                        zmid=0,
                        text=np.round(t_stats, 4),
                        texttemplate="%{text:.4f}",
                        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>T-Statistic: %{z:.4f}<extra></extra>",
                        colorbar=dict(title="T-Statistic", x=1.0),
                    ),
                    row=1,
                    col=2,
                )

                # Update layout
                fig.update_layout(
                    title="Statistical Significance Tests (Welch's t-test)",
                    height=500,
                    margin=dict(l=40, r=40, t=60, b=40),
                )

                # Add annotations for significance levels
                fig.add_annotation(
                    text="* p < 0.05: Significant difference",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                )

                fig.add_annotation(
                    text="** p < 0.01: Highly significant difference",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.2,
                    showarrow=False,
                )

                return fig.to_dict()
            else:
                return {"error": "Need at least two series for statistical comparison"}

        except Exception as e:
            logger.error(f"Error creating statistical significance test: {e}")
            return {"error": str(e)}

    def create_comparison_dashboard(
        self,
        strategy_results: List[Dict[str, Any]],
        benchmark_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive strategy comparison dashboard.

        Args:
            strategy_results: List of strategy results
            benchmark_results: Optional list of benchmark results

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of Plotly figures as JSON
        """
        try:
            dashboard = {}

            # Create charts
            dashboard["equity_comparison"] = self.create_equity_comparison_chart(
                strategy_results, benchmark_results
            )

            dashboard["drawdown_comparison"] = self.create_drawdown_comparison_chart(
                strategy_results, benchmark_results
            )

            dashboard["rolling_returns"] = self.create_rolling_returns_comparison(
                strategy_results, 30, benchmark_results
            )

            dashboard["metrics_table"] = self.create_metrics_comparison_table(
                strategy_results, benchmark_results
            )

            dashboard["monthly_returns"] = (
                self.create_monthly_returns_heatmap_comparison(strategy_results)
            )

            if len(strategy_results) > 1 or benchmark_results:
                dashboard["statistical_tests"] = (
                    self.create_statistical_significance_test(
                        strategy_results, benchmark_results
                    )
                )

            return dashboard

        except Exception as e:
            logger.error(f"Error creating comparison dashboard: {e}")
            return {"error": str(e)}
