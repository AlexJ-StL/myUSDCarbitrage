"""Risk analysis visualizations including drawdown charts and correlation matrices."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .risk_management import RiskManagementAnalytics

# Configure logging
logger = logging.getLogger(__name__)


class RiskAnalysis:
    """Risk analysis visualizations including drawdown charts and correlation matrices."""

    def __init__(self):
        """Initialize risk analysis visualizations."""
        self.risk_mgmt = RiskManagementAnalytics()

    def create_drawdown_analysis_chart(
        self, equity_curve_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create detailed drawdown analysis chart.

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

            # Calculate drawdown
            df["peak"] = df["equity"].cummax()
            df["drawdown"] = (df["equity"] / df["peak"]) - 1

            # Identify drawdown periods (drawdown < -5%)
            threshold = -0.05
            df["is_drawdown"] = df["drawdown"] < threshold

            # Create drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_idx = None

            for idx, row in df.iterrows():
                if not in_drawdown and row["is_drawdown"]:
                    # Start of drawdown period
                    in_drawdown = True
                    start_idx = idx
                elif in_drawdown and not row["is_drawdown"]:
                    # End of drawdown period
                    in_drawdown = False
                    drawdown_periods.append({
                        "start_idx": start_idx,
                        "end_idx": idx,
                        "start_date": df.loc[start_idx, "timestamp"],
                        "end_date": row["timestamp"],
                        "duration": (
                            row["timestamp"] - df.loc[start_idx, "timestamp"]
                        ).days,
                        "max_drawdown": df.loc[start_idx:idx, "drawdown"].min(),
                        "recovery": (row["equity"] / df.loc[start_idx, "equity"]) - 1,
                    })
                    start_idx = None

            # Check if still in drawdown at the end
            if in_drawdown:
                drawdown_periods.append({
                    "start_idx": start_idx,
                    "end_idx": df.index[-1],
                    "start_date": df.loc[start_idx, "timestamp"],
                    "end_date": df.iloc[-1]["timestamp"],
                    "duration": (
                        df.iloc[-1]["timestamp"] - df.loc[start_idx, "timestamp"]
                    ).days,
                    "max_drawdown": df.loc[start_idx:, "drawdown"].min(),
                    "recovery": (df.iloc[-1]["equity"] / df.loc[start_idx, "equity"])
                    - 1,
                })

            # Create figure with subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(
                    "Equity Curve with Drawdown Periods",
                    "Drawdown Percentage",
                ),
                row_heights=[0.6, 0.4],
            )

            # Add equity curve
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["equity"],
                    mode="lines",
                    name="Equity",
                    line=dict(color="blue", width=2),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Equity:</b> $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Add peak equity
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["peak"],
                    mode="lines",
                    name="Peak Equity",
                    line=dict(color="green", width=1, dash="dash"),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Peak Equity:</b> $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Highlight drawdown periods
            for i, period in enumerate(drawdown_periods):
                fig.add_trace(
                    go.Scatter(
                        x=[period["start_date"], period["end_date"]],
                        y=[
                            df.loc[period["start_idx"], "equity"],
                            df.loc[period["end_idx"], "equity"],
                        ],
                        mode="markers",
                        name=f"Drawdown {i + 1}",
                        marker=dict(color="red", size=8),
                        hovertemplate=(
                            f"<b>Drawdown {i + 1}</b><br>"
                            f"Start: %{{x}}<br>"
                            f"End: {period['end_date']}<br>"
                            f"Duration: {period['duration']} days<br>"
                            f"Max Drawdown: {period['max_drawdown'] * 100:.2f}%<br>"
                            f"Recovery: {period['recovery'] * 100:.2f}%<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

                # Add shaded area for drawdown period
                fig.add_trace(
                    go.Scatter(
                        x=[
                            period["start_date"],
                            period["start_date"],
                            period["end_date"],
                            period["end_date"],
                        ],
                        y=[
                            df["equity"].min() * 0.95,
                            df["equity"].max() * 1.05,
                            df["equity"].max() * 1.05,
                            df["equity"].min() * 0.95,
                        ],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        line=dict(color="rgba(255, 0, 0, 0)"),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

            # Add drawdown percentage
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["drawdown"] * 100,  # Convert to percentage
                    mode="lines",
                    name="Drawdown %",
                    line=dict(color="red", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Add threshold line
            fig.add_trace(
                go.Scatter(
                    x=[df["timestamp"].min(), df["timestamp"].max()],
                    y=[threshold * 100, threshold * 100],  # Convert to percentage
                    mode="lines",
                    name="Threshold (-5%)",
                    line=dict(color="orange", width=1, dash="dash"),
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

            # Update layout
            fig.update_layout(
                title="Drawdown Analysis",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                height=700,
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

            # Add range selector
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
            logger.error(f"Error creating drawdown analysis chart: {e}")
            return {"error": str(e)}

    def create_underwater_chart(
        self, equity_curve_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create underwater chart showing drawdown periods.

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

            # Calculate drawdown
            df["peak"] = df["equity"].cummax()
            df["drawdown"] = (df["equity"] / df["peak"]) - 1

            # Create figure
            fig = go.Figure()

            # Add underwater chart
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["drawdown"] * 100,  # Convert to percentage
                    mode="lines",
                    name="Drawdown",
                    fill="tozeroy",
                    fillcolor="rgba(255, 0, 0, 0.3)",
                    line=dict(color="red", width=2),
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

            # Add threshold lines
            thresholds = [-5, -10, -20]
            colors = ["orange", "red", "darkred"]

            for threshold, color in zip(thresholds, colors):
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color=color,
                    line_width=1,
                    annotation_text=f"{threshold}%",
                    annotation_position="left",
                )

            # Update layout
            fig.update_layout(
                title="Underwater Chart (Drawdown Periods)",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode="x unified",
                yaxis=dict(
                    tickformat=".1f",
                    ticksuffix="%",
                ),
                margin=dict(l=40, r=40, t=60, b=40),
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
            logger.error(f"Error creating underwater chart: {e}")
            return {"error": str(e)}

    def create_volatility_chart(
        self, equity_curve_data: List[Dict[str, Any]], window_days: int = 30
    ) -> Dict[str, Any]:
        """Create rolling volatility chart.

        Args:
            equity_curve_data: List of equity curve data points
            window_days: Rolling window size in days

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

            # Calculate rolling volatility (annualized)
            df["rolling_volatility"] = df["daily_return"].rolling(
                window=window_days
            ).std() * np.sqrt(252)

            # Create figure
            fig = go.Figure()

            # Add rolling volatility
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["rolling_volatility"] * 100,  # Convert to percentage
                    mode="lines",
                    name=f"{window_days}-Day Rolling Volatility",
                    line=dict(color="purple", width=2),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Volatility:</b> %{y:.2f}%<extra></extra>",
                )
            )

            # Calculate average volatility
            avg_volatility = (
                df["rolling_volatility"].mean() * 100
            )  # Convert to percentage

            # Add average volatility line
            fig.add_hline(
                y=avg_volatility,
                line_dash="dash",
                line_color="gray",
                line_width=1,
                annotation_text=f"Avg: {avg_volatility:.2f}%",
                annotation_position="right",
            )

            # Update layout
            fig.update_layout(
                title=f"{window_days}-Day Rolling Volatility (Annualized)",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                hovermode="x unified",
                yaxis=dict(
                    tickformat=".2f",
                    ticksuffix="%",
                ),
                margin=dict(l=40, r=40, t=60, b=40),
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
            logger.error(f"Error creating volatility chart: {e}")
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
                    margin=dict(l=40, r=40, t=60, b=40),
                )

                return fig.to_dict()
            else:
                return {"error": "No valid data for correlation matrix"}

        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            return {"error": str(e)}

    def create_risk_return_scatter(
        self,
        strategy_results: List[Dict[str, Any]],
        benchmark_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create risk-return scatter plot.

        Args:
            strategy_results: List of strategy results
            benchmark_results: Optional list of benchmark results

        Returns:
            Dict[str, Any]: Plotly figure as JSON
        """
        try:
            # Prepare data
            risk_return_data = []

            # Process strategy results
            for strategy in strategy_results:
                if (
                    "equity_curve" in strategy
                    and strategy["equity_curve"]
                    and "metrics" in strategy
                ):
                    # Extract metrics
                    metrics = strategy["metrics"]

                    risk_return_data.append({
                        "name": strategy["name"],
                        "return": metrics.get("total_return", 0)
                        * 100,  # Convert to percentage
                        "risk": metrics.get("annualized_volatility", 0)
                        * 100,  # Convert to percentage
                        "sharpe": metrics.get("sharpe_ratio", 0),
                        "max_drawdown": metrics.get("max_drawdown", 0)
                        * 100,  # Convert to percentage
                        "type": "strategy",
                    })

            # Process benchmark results if provided
            if benchmark_results:
                for benchmark in benchmark_results:
                    if (
                        "data" in benchmark
                        and benchmark["data"]
                        and "metrics" in benchmark
                    ):
                        # Extract metrics
                        metrics = benchmark["metrics"]

                        risk_return_data.append({
                            "name": benchmark["name"],
                            "return": metrics.get("total_return", 0)
                            * 100,  # Convert to percentage
                            "risk": metrics.get("annualized_volatility", 0)
                            * 100,  # Convert to percentage
                            "sharpe": metrics.get("sharpe_ratio", 0),
                            "max_drawdown": metrics.get("max_drawdown", 0)
                            * 100,  # Convert to percentage
                            "type": "benchmark",
                        })

            # Create scatter plot
            if risk_return_data:
                df = pd.DataFrame(risk_return_data)

                # Create figure
                fig = go.Figure()

                # Add strategies
                strategies = df[df["type"] == "strategy"]
                if not strategies.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=strategies["risk"],
                            y=strategies["return"],
                            mode="markers+text",
                            name="Strategies",
                            marker=dict(
                                size=strategies["sharpe"].abs() * 5
                                + 10,  # Size based on Sharpe ratio
                                color=strategies[
                                    "max_drawdown"
                                ],  # Color based on max drawdown
                                colorscale="RdYlGn_r",  # Red for large drawdowns, green for small
                                colorbar=dict(title="Max Drawdown (%)"),
                                showscale=True,
                            ),
                            text=strategies["name"],
                            textposition="top center",
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                "Return: %{y:.2f}%<br>"
                                "Risk: %{x:.2f}%<br>"
                                "Sharpe: %{marker.size:.2f}<br>"
                                "Max Drawdown: %{marker.color:.2f}%<extra></extra>"
                            ),
                        )
                    )

                # Add benchmarks
                benchmarks = df[df["type"] == "benchmark"]
                if not benchmarks.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=benchmarks["risk"],
                            y=benchmarks["return"],
                            mode="markers+text",
                            name="Benchmarks",
                            marker=dict(
                                size=15,
                                symbol="diamond",
                                color="black",
                                line=dict(color="white", width=1),
                            ),
                            text=benchmarks["name"],
                            textposition="top center",
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                "Return: %{y:.2f}%<br>"
                                "Risk: %{x:.2f}%<br>"
                                "Sharpe: %{marker.size:.2f}<br>"
                                "Max Drawdown: %{marker.color:.2f}%<extra></extra>"
                            ),
                        )
                    )

                # Add Sharpe ratio lines
                max_risk = df["risk"].max() * 1.2
                sharpe_ratios = [0.5, 1.0, 1.5, 2.0, 2.5]

                for sharpe in sharpe_ratios:
                    # y = sharpe * x (assuming risk-free rate = 0)
                    x_values = np.linspace(0, max_risk, 100)
                    y_values = sharpe * x_values

                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode="lines",
                            name=f"Sharpe = {sharpe}",
                            line=dict(color="gray", width=1, dash="dot"),
                            hoverinfo="skip",
                        )
                    )

                # Update layout
                fig.update_layout(
                    title="Risk-Return Analysis",
                    xaxis_title="Risk (Annualized Volatility %)",
                    yaxis_title="Return (%)",
                    hovermode="closest",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                    margin=dict(l=40, r=40, t=60, b=40),
                    height=600,
                    width=800,
                )

                return fig.to_dict()
            else:
                return {"error": "No valid data for risk-return scatter plot"}

        except Exception as e:
            logger.error(f"Error creating risk-return scatter plot: {e}")
            return {"error": str(e)}

    def calculate_var_cvar(
        self, returns: List[float], confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).

        Args:
            returns: List of portfolio returns
            confidence_levels: List of confidence levels

        Returns:
            Dict containing VaR and CVaR for each confidence level
        """
        return self.risk_mgmt.calculate_var_cvar(returns, confidence_levels)

    def calculate_portfolio_risk_attribution(
        self, positions: List[Dict[str, Any]], returns_data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Calculate portfolio risk attribution analysis.

        Args:
            positions: List of position data with weights
            returns_data: Dictionary of asset returns data

        Returns:
            Dict containing risk attribution metrics
        """
        return self.risk_mgmt.calculate_portfolio_risk_attribution(
            positions, returns_data
        )

    def perform_stress_testing(
        self,
        equity_curve_data: List[Dict[str, Any]],
        scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Perform stress testing with historical scenarios.

        Args:
            equity_curve_data: Portfolio equity curve data
            scenarios: Optional custom stress scenarios

        Returns:
            Dict containing stress test results
        """
        return self.risk_mgmt.perform_stress_testing(equity_curve_data, scenarios)

    def detect_market_regimes(
        self, equity_curve_data: List[Dict[str, Any]], n_regimes: int = 3
    ) -> Dict[str, Any]:
        """Detect market regimes using clustering analysis.

        Args:
            equity_curve_data: Portfolio equity curve data
            n_regimes: Number of market regimes to detect

        Returns:
            Dict containing regime analysis results
        """
        return self.risk_mgmt.detect_market_regimes(equity_curve_data, n_regimes)

    def create_var_cvar_chart(
        self, returns: List[float], confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, Any]:
        """Create VaR and CVaR visualization chart.

        Args:
            returns: List of portfolio returns
            confidence_levels: List of confidence levels

        Returns:
            Dict containing Plotly figure as JSON
        """
        return self.risk_mgmt.create_var_cvar_chart(returns, confidence_levels)

    def create_stress_test_chart(
        self, stress_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create stress testing visualization chart.

        Args:
            stress_results: Results from stress testing analysis

        Returns:
            Dict containing Plotly figure as JSON
        """
        return self.risk_mgmt.create_stress_test_chart(stress_results)

    def create_regime_analysis_chart(
        self, regime_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create market regime analysis visualization chart.

        Args:
            regime_results: Results from regime detection analysis

        Returns:
            Dict containing Plotly figure as JSON
        """
        return self.risk_mgmt.create_regime_analysis_chart(regime_results)

    def create_risk_attribution_chart(
        self, attribution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create portfolio risk attribution visualization chart.

        Args:
            attribution_results: Results from risk attribution analysis

        Returns:
            Dict containing Plotly figure as JSON
        """
        return self.risk_mgmt.create_risk_attribution_chart(attribution_results)

    def create_risk_dashboard(
        self,
        backtest_result: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive risk analysis dashboard.

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

            # Create charts
            if equity_curve_data:
                dashboard["drawdown_analysis"] = self.create_drawdown_analysis_chart(
                    equity_curve_data
                )
                dashboard["underwater"] = self.create_underwater_chart(
                    equity_curve_data
                )
                dashboard["volatility"] = self.create_volatility_chart(
                    equity_curve_data
                )

                # Calculate returns for advanced risk analytics
                df = pd.DataFrame(equity_curve_data)
                if not df.empty and "equity" in df.columns:
                    df["returns"] = df["equity"].pct_change().fillna(0)
                    returns = df["returns"].tolist()

                    # Advanced risk management analytics
                    dashboard["var_cvar"] = self.create_var_cvar_chart(returns)

                    # Stress testing
                    stress_results = self.perform_stress_testing(equity_curve_data)
                    if "error" not in stress_results:
                        dashboard["stress_test"] = self.create_stress_test_chart(
                            stress_results
                        )

                    # Regime analysis
                    regime_results = self.detect_market_regimes(equity_curve_data)
                    if "error" not in regime_results:
                        dashboard["regime_analysis"] = (
                            self.create_regime_analysis_chart(regime_results)
                        )

                # Portfolio risk attribution (if position data available)
                positions_data = backtest_result.get("results", {}).get("positions", [])
                if positions_data:
                    # Extract returns data for risk attribution
                    returns_data = {}
                    for pos in positions_data:
                        if "symbol" in pos and "returns" in pos:
                            returns_data[pos["symbol"]] = pos["returns"]

                    if returns_data:
                        attribution_results = self.calculate_portfolio_risk_attribution(
                            positions_data, returns_data
                        )
                        if "error" not in attribution_results:
                            dashboard["risk_attribution"] = (
                                self.create_risk_attribution_chart(attribution_results)
                            )

                # For correlation and risk-return, we need multiple strategies
                if benchmark_data:
                    dashboard["correlation"] = self.create_correlation_matrix(
                        [backtest_result], [benchmark_data]
                    )
                    dashboard["risk_return"] = self.create_risk_return_scatter(
                        [backtest_result], [benchmark_data]
                    )

            return dashboard

        except Exception as e:
            logger.error(f"Error creating risk dashboard: {e}")
            return {"error": str(e)}
