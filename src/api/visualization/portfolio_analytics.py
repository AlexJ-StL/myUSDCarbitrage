"""Portfolio analytics dashboard with drill-down capabilities."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logger = logging.getLogger(__name__)


class PortfolioAnalytics:
    """Portfolio analytics dashboard with drill-down capabilities."""

    def __init__(self):
        """Initialize portfolio analytics dashboard."""
        pass

    def create_portfolio_composition_chart(
        self, positions_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create portfolio composition chart.

        Args:
            positions_data: List of position data points

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(positions_data)

            if df.empty:
                return {"error": "No position data available"}

            # Ensure required columns exist
            required_columns = [
                "timestamp",
                "exchange",
                "symbol",
                "amount",
                "current_price",
            ]
            for col in required_columns:
                if col not in df.columns:
                    return {"error": f"Missing required column: {col}"}

            # Get the latest snapshot
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            latest_date = df["timestamp"].max()
            latest_positions = df[df["timestamp"] == latest_date]

            # Calculate position values
            latest_positions["position_value"] = (
                latest_positions["amount"] * latest_positions["current_price"]
            )

            # Group by exchange and symbol
            grouped = (
                latest_positions.groupby(["exchange", "symbol"])["position_value"]
                .sum()
                .reset_index()
            )

            # Create sunburst chart
            fig = go.Figure(
                go.Sunburst(
                    labels=grouped["symbol"].tolist() + grouped["exchange"].tolist(),
                    parents=grouped["exchange"].tolist()
                    + [""] * len(grouped["exchange"].unique()),
                    values=grouped["position_value"].tolist()
                    + [0] * len(grouped["exchange"].unique()),
                    branchvalues="total",
                    hovertemplate="<b>%{label}</b><br>Value: $%{value:.2f}<br>Percentage: %{percentRoot:.2%}<extra></extra>",
                )
            )

            # Update layout
            fig.update_layout(
                title="Portfolio Composition",
                margin=dict(l=0, r=0, t=40, b=0),
                height=500,
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating portfolio composition chart: {e}")
            return {"error": str(e)}

    def create_position_history_chart(
        self,
        positions_data: List[Dict[str, Any]],
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create position history chart with drill-down capability.

        Args:
            positions_data: List of position data points
            exchange: Optional exchange filter
            symbol: Optional symbol filter

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(positions_data)

            if df.empty:
                return {"error": "No position data available"}

            # Ensure required columns exist
            required_columns = [
                "timestamp",
                "exchange",
                "symbol",
                "amount",
                "current_price",
            ]
            for col in required_columns:
                if col not in df.columns:
                    return {"error": f"Missing required column: {col}"}

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["position_value"] = df["amount"] * df["current_price"]

            # Apply filters if provided
            if exchange:
                df = df[df["exchange"] == exchange]

            if symbol:
                df = df[df["symbol"] == symbol]

            if df.empty:
                return {"error": "No data available after applying filters"}

            # Group by timestamp and calculate total position value
            if exchange and symbol:
                # Detailed view for specific exchange and symbol
                grouped = df.sort_values("timestamp")
            else:
                # Aggregate view
                grouped = (
                    df.groupby(["timestamp", "exchange", "symbol"])["position_value"]
                    .sum()
                    .reset_index()
                )

            # Create figure
            fig = go.Figure()

            # Add position value traces
            for (exch, sym), group in grouped.groupby(["exchange", "symbol"]):
                fig.add_trace(
                    go.Scatter(
                        x=group["timestamp"],
                        y=group["position_value"],
                        mode="lines",
                        name=f"{exch} - {sym}",
                        hovertemplate="<b>Date:</b> %{x}<br><b>Value:</b> $%{y:.2f}<extra></extra>",
                    )
                )

            # Update layout
            title = "Position History"
            if exchange:
                title += f" - {exchange}"
                if symbol:
                    title += f" - {symbol}"

            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Position Value ($)",
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
            logger.error(f"Error creating position history chart: {e}")
            return {"error": str(e)}

    def create_exchange_exposure_chart(
        self, positions_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create exchange exposure chart.

        Args:
            positions_data: List of position data points

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(positions_data)

            if df.empty:
                return {"error": "No position data available"}

            # Ensure required columns exist
            required_columns = ["timestamp", "exchange", "amount", "current_price"]
            for col in required_columns:
                if col not in df.columns:
                    return {"error": f"Missing required column: {col}"}

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["position_value"] = df["amount"] * df["current_price"]

            # Group by timestamp and exchange
            grouped = (
                df.groupby(["timestamp", "exchange"])["position_value"]
                .sum()
                .reset_index()
            )

            # Pivot to get exchanges as columns
            pivot_df = grouped.pivot(
                index="timestamp", columns="exchange", values="position_value"
            ).fillna(0)

            # Calculate percentage
            pivot_df_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

            # Create figure
            fig = go.Figure()

            # Add area chart for each exchange
            for exchange in pivot_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=pivot_df.index,
                        y=pivot_df_pct[exchange],
                        mode="lines",
                        name=exchange,
                        stackgroup="one",
                        hovertemplate="<b>Date:</b> %{x}<br><b>"
                        + exchange
                        + ":</b> %{y:.2f}%<extra></extra>",
                    )
                )

            # Update layout
            fig.update_layout(
                title="Exchange Exposure Over Time",
                xaxis_title="Date",
                yaxis_title="Allocation (%)",
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
            logger.error(f"Error creating exchange exposure chart: {e}")
            return {"error": str(e)}

    def create_portfolio_metrics_chart(
        self, equity_curve_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create portfolio metrics chart with key performance indicators.

        Args:
            equity_curve_data: List of equity curve data points

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(equity_curve_data)

            if df.empty:
                return {"error": "No equity curve data available"}

            # Ensure required columns exist
            required_columns = ["timestamp", "equity"]
            for col in required_columns:
                if col not in df.columns:
                    return {"error": f"Missing required column: {col}"}

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Calculate daily returns
            df["daily_return"] = df["equity"].pct_change()

            # Calculate metrics
            initial_equity = df["equity"].iloc[0]
            final_equity = df["equity"].iloc[-1]
            total_return = (final_equity / initial_equity) - 1

            # Calculate drawdown
            df["peak"] = df["equity"].cummax()
            df["drawdown"] = (df["equity"] / df["peak"]) - 1
            max_drawdown = df["drawdown"].min()

            # Calculate volatility (annualized)
            volatility = df["daily_return"].std() * np.sqrt(252)

            # Calculate Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = (
                df["daily_return"].mean() / df["daily_return"].std() * np.sqrt(252)
                if df["daily_return"].std() > 0
                else 0
            )

            # Calculate win rate
            win_rate = (df["daily_return"] > 0).mean()

            # Create figure with subplots for KPIs
            fig = make_subplots(
                rows=2,
                cols=3,
                specs=[
                    [
                        {"type": "indicator"},
                        {"type": "indicator"},
                        {"type": "indicator"},
                    ],
                    [
                        {"type": "indicator"},
                        {"type": "indicator"},
                        {"type": "indicator"},
                    ],
                ],
                subplot_titles=(
                    "Total Return",
                    "Max Drawdown",
                    "Volatility",
                    "Sharpe Ratio",
                    "Win Rate",
                    "Final Equity",
                ),
            )

            # Add indicators
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=total_return * 100,  # Convert to percentage
                    number={"suffix": "%", "valueformat": ".2f"},
                    delta={"reference": 0, "valueformat": ".2f"},
                    title={"text": "Total Return"},
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=max_drawdown * 100,  # Convert to percentage
                    number={"suffix": "%", "valueformat": ".2f"},
                    delta={
                        "reference": 0,
                        "valueformat": ".2f",
                        "increasing": {"color": "red"},
                        "decreasing": {"color": "green"},
                    },
                    title={"text": "Max Drawdown"},
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=volatility * 100,  # Convert to percentage
                    number={"suffix": "%", "valueformat": ".2f"},
                    title={"text": "Annualized Volatility"},
                ),
                row=1,
                col=3,
            )

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=sharpe_ratio,
                    number={"valueformat": ".2f"},
                    title={"text": "Sharpe Ratio"},
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=win_rate * 100,  # Convert to percentage
                    number={"suffix": "%", "valueformat": ".2f"},
                    title={"text": "Win Rate"},
                ),
                row=2,
                col=2,
            )

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=final_equity,
                    number={"prefix": "$", "valueformat": ",.2f"},
                    title={"text": "Final Equity"},
                ),
                row=2,
                col=3,
            )

            # Update layout
            fig.update_layout(
                title="Portfolio Performance Metrics",
                height=500,
                margin=dict(l=40, r=40, t=60, b=40),
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating portfolio metrics chart: {e}")
            return {"error": str(e)}

    def create_portfolio_dashboard(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive portfolio analytics dashboard.

        Args:
            backtest_result: Backtest result data

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of Plotly figures as JSON
        """
        try:
            dashboard = {}

            # Extract data
            equity_curve_data = backtest_result.get("results", {}).get(
                "equity_curve", []
            )
            positions_data = backtest_result.get("results", {}).get("positions", [])

            # Create charts
            if positions_data:
                dashboard["portfolio_composition"] = (
                    self.create_portfolio_composition_chart(positions_data)
                )
                dashboard["position_history"] = self.create_position_history_chart(
                    positions_data
                )
                dashboard["exchange_exposure"] = self.create_exchange_exposure_chart(
                    positions_data
                )

            if equity_curve_data:
                dashboard["portfolio_metrics"] = self.create_portfolio_metrics_chart(
                    equity_curve_data
                )

            return dashboard

        except Exception as e:
            logger.error(f"Error creating portfolio dashboard: {e}")
            return {"error": str(e)}
