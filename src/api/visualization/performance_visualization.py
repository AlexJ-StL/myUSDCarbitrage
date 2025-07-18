"""Performance visualization system for strategy backtesting results."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logger = logging.getLogger(__name__)


class PerformanceVisualization:
    """Performance visualization system for strategy backtesting results."""

    def __init__(self):
        """Initialize performance visualization system."""
        pass

    def create_equity_curve_chart(
        self,
        equity_curve_data: List[Dict[str, Any]],
        benchmark_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create interactive equity curve chart.

        Args:
            equity_curve_data: List of equity curve data points
            benchmark_data: Optional list of benchmark data points

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(equity_curve_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

            # Create figure
            fig = go.Figure()

            # Add equity curve
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["equity"],
                    mode="lines",
                    name="Strategy Equity",
                    line=dict(color="blue", width=2),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Equity:</b> $%{y:.2f}<extra></extra>",
                )
            )

            # Add benchmark if provided
            if benchmark_data:
                benchmark_df = pd.DataFrame(benchmark_data)
                benchmark_df["timestamp"] = pd.to_datetime(benchmark_df["timestamp"])
                benchmark_df = benchmark_df.set_index("timestamp")

                # Align benchmark with equity curve
                common_index = df.index.intersection(benchmark_df.index)
                if not common_index.empty:
                    benchmark_df = benchmark_df.loc[common_index]

                    fig.add_trace(
                        go.Scatter(
                            x=benchmark_df.index,
                            y=benchmark_df["value"],
                            mode="lines",
                            name="Benchmark",
                            line=dict(color="gray", width=2, dash="dash"),
                            hovertemplate="<b>Date:</b> %{x}<br><b>Value:</b> $%{y:.2f}<extra></extra>",
                        )
                    )

            # Calculate drawdown
            if "equity" in df.columns:
                peak = df["equity"].cummax()
                drawdown = (df["equity"] / peak) - 1

                # Add drawdown chart
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=drawdown * 100,  # Convert to percentage
                        mode="lines",
                        name="Drawdown %",
                        line=dict(color="red", width=1.5),
                        visible="legendonly",  # Hidden by default
                        hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
                    )
                )

            # Update layout
            fig.update_layout(
                title="Strategy Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=40, r=40, t=60, b=60),
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
            logger.error(f"Error creating equity curve chart: {e}")
            return {"error": str(e)}

    def create_drawdown_chart(
        self, equity_curve_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create interactive drawdown chart.

        Args:
            equity_curve_data: List of equity curve data points

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(equity_curve_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

            # Calculate drawdown
            peak = df["equity"].cummax()
            drawdown = (df["equity"] / peak) - 1

            # Create figure
            fig = go.Figure()

            # Add drawdown chart
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=drawdown * 100,  # Convert to percentage
                    mode="lines",
                    name="Drawdown %",
                    fill="tozeroy",
                    line=dict(color="red", width=2),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
                )
            )

            # Update layout
            fig.update_layout(
                title="Drawdown Analysis",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode="x unified",
                yaxis=dict(
                    tickformat=".2f",
                    ticksuffix="%",
                ),
                margin=dict(l=40, r=40, t=60, b=60),
                height=400,
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
            logger.error(f"Error creating drawdown chart: {e}")
            return {"error": str(e)}

    def create_monthly_returns_heatmap(
        self, equity_curve_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create monthly returns heatmap.

        Args:
            equity_curve_data: List of equity curve data points

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(equity_curve_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

            # Calculate daily returns
            df["daily_return"] = df["equity"].pct_change()

            # Resample to get monthly returns
            monthly_returns = (
                df["daily_return"].resample("M").apply(lambda x: (1 + x).prod() - 1)
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

            # Create figure
            fig = go.Figure(
                data=go.Heatmap(
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
                )
            )

            # Update layout
            fig.update_layout(
                title="Monthly Returns Heatmap (%)",
                xaxis_title="Month",
                yaxis_title="Year",
                yaxis=dict(autorange="reversed"),  # Latest year at the top
                margin=dict(l=40, r=40, t=60, b=60),
                height=400,
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating monthly returns heatmap: {e}")
            return {"error": str(e)}

    def create_rolling_returns_chart(
        self, equity_curve_data: List[Dict[str, Any]], window_days: int = 30
    ) -> Dict[str, Any]:
        """Create rolling returns chart.

        Args:
            equity_curve_data: List of equity curve data points
            window_days: Rolling window size in days

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(equity_curve_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

            # Calculate daily returns
            df["daily_return"] = df["equity"].pct_change()

            # Calculate rolling returns
            df["rolling_return"] = (
                (1 + df["daily_return"])
                .rolling(window=window_days)
                .apply(lambda x: x.prod() - 1, raw=True)
            )

            # Create figure
            fig = go.Figure()

            # Add rolling returns chart
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["rolling_return"] * 100,  # Convert to percentage
                    mode="lines",
                    name=f"{window_days}-Day Rolling Return",
                    line=dict(color="blue", width=2),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>",
                )
            )

            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                annotation_text="Break-even",
                annotation_position="bottom right",
            )

            # Update layout
            fig.update_layout(
                title=f"{window_days}-Day Rolling Returns",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                hovermode="x unified",
                yaxis=dict(
                    tickformat=".2f",
                    ticksuffix="%",
                ),
                margin=dict(l=40, r=40, t=60, b=60),
                height=400,
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating rolling returns chart: {e}")
            return {"error": str(e)}

    def create_trade_analysis_chart(
        self, trades_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create trade analysis chart.

        Args:
            trades_data: List of trade data points

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(trades_data)

            if df.empty:
                return {"error": "No trade data available"}

            # Ensure required columns exist
            required_columns = ["timestamp", "side", "price", "amount", "pnl"]
            for col in required_columns:
                if col not in df.columns:
                    return {"error": f"Missing required column: {col}"}

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Calculate cumulative PnL
            df["cumulative_pnl"] = df["pnl"].cumsum()

            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("Cumulative PnL", "Trade PnL Distribution"),
                row_heights=[0.7, 0.3],
            )

            # Add cumulative PnL trace
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["cumulative_pnl"],
                    mode="lines",
                    name="Cumulative PnL",
                    line=dict(color="blue", width=2),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Cumulative PnL:</b> $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Add individual trade PnL as bar chart
            colors = ["green" if pnl >= 0 else "red" for pnl in df["pnl"]]

            fig.add_trace(
                go.Bar(
                    x=df["timestamp"],
                    y=df["pnl"],
                    name="Trade PnL",
                    marker_color=colors,
                    hovertemplate="<b>Date:</b> %{x}<br><b>PnL:</b> $%{y:.2f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Update layout
            fig.update_layout(
                title="Trade Analysis",
                hovermode="x unified",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=40, r=40, t=60, b=60),
                height=600,
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Cumulative PnL ($)", row=1, col=1)
            fig.update_yaxes(title_text="Trade PnL ($)", row=2, col=1)

            # Add range selector to bottom subplot
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ])
                ),
                row=2,
                col=1,
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating trade analysis chart: {e}")
            return {"error": str(e)}

    def create_correlation_matrix(
        self,
        strategy_results: List[Dict[str, Any]],
        benchmark_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create correlation matrix visualization.

        Args:
            strategy_results: List of strategy results
            benchmark_results: Optional list of benchmark results

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Prepare data
            data_dict = {}

            # Process strategy results
            for strategy in strategy_results:
                if "equity_curve" in strategy and strategy["equity_curve"]:
                    df = pd.DataFrame(strategy["equity_curve"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                    df["return"] = df["equity"].pct_change().fillna(0)
                    data_dict[strategy["name"]] = df["return"]

            # Process benchmark results if provided
            if benchmark_results:
                for benchmark in benchmark_results:
                    if "data" in benchmark and benchmark["data"]:
                        df = pd.DataFrame(benchmark["data"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.set_index("timestamp")
                        df["return"] = df["value"].pct_change().fillna(0)
                        data_dict[benchmark["name"]] = df["return"]

            # Create correlation matrix
            if data_dict:
                # Combine all series into a DataFrame
                returns_df = pd.DataFrame(data_dict)

                # Calculate correlation matrix
                corr_matrix = returns_df.corr()

                # Create heatmap
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale="RdBu",
                        zmid=0,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate="%{text:.2f}",
                        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.4f}<extra></extra>",
                    )
                )

                # Update layout
                fig.update_layout(
                    title="Strategy Correlation Matrix",
                    height=500,
                    width=600,
                    margin=dict(l=40, r=40, t=60, b=60),
                )

                return fig.to_dict()
            else:
                return {"error": "No valid data for correlation matrix"}

        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            return {"error": str(e)}

    def create_risk_metrics_radar(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create risk metrics radar chart.

        Args:
            metrics: Dictionary of risk metrics

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Define metrics to include in radar chart
            radar_metrics = [
                ("sharpe_ratio", "Sharpe Ratio", 1),
                ("sortino_ratio", "Sortino Ratio", 1),
                ("calmar_ratio", "Calmar Ratio", 1),
                ("win_rate", "Win Rate", 100),  # Convert to percentage
                ("profit_factor", "Profit Factor", 1),
                (
                    "max_drawdown",
                    "Max Drawdown",
                    -100,
                ),  # Convert to percentage and invert
            ]

            # Extract values
            values = []
            labels = []

            for key, label, scale in radar_metrics:
                if key == "max_drawdown":
                    # Invert drawdown (less negative is better)
                    value = (
                        metrics.get(key, 0) * scale * -1
                    )  # Make positive for visualization
                else:
                    value = metrics.get(key, 0) * scale

                values.append(value)
                labels.append(label)

            # Create radar chart
            fig = go.Figure()

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill="toself",
                    name="Strategy Metrics",
                    line_color="blue",
                )
            )

            # Update layout
            fig.update_layout(
                title="Risk-Return Profile",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(values) * 1.1],  # Add 10% margin
                    )
                ),
                showlegend=False,
                height=500,
                width=500,
                margin=dict(l=40, r=40, t=60, b=60),
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating risk metrics radar chart: {e}")
            return {"error": str(e)}

    def create_performance_dashboard(
        self,
        backtest_result: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive performance dashboard.

        Args:
            backtest_result: Backtest result data
            benchmark_data: Optional benchmark data

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of Plotly figures as JSON
        """
        try:
            dashboard = {}

            # Extract data
            equity_curve_data = backtest_result.get("results", {}).get(
                "equity_curve", []
            )
            trades_data = backtest_result.get("results", {}).get("transactions", [])
            metrics = backtest_result.get("metrics", {})

            # Create charts
            if equity_curve_data:
                benchmark_curve = (
                    benchmark_data.get("data", []) if benchmark_data else None
                )

                dashboard["equity_curve"] = self.create_equity_curve_chart(
                    equity_curve_data, benchmark_curve
                )
                dashboard["drawdown"] = self.create_drawdown_chart(equity_curve_data)
                dashboard["monthly_returns"] = self.create_monthly_returns_heatmap(
                    equity_curve_data
                )
                dashboard["rolling_returns"] = self.create_rolling_returns_chart(
                    equity_curve_data
                )

            if trades_data:
                dashboard["trade_analysis"] = self.create_trade_analysis_chart(
                    trades_data
                )

            if metrics:
                dashboard["risk_metrics"] = self.create_risk_metrics_radar(metrics)

            return dashboard

        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            return {"error": str(e)}
