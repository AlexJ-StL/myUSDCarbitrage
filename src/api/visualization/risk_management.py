"""Advanced risk management analytics including VaR, CVaR, stress testing, and regime detection."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)


class RiskManagementAnalytics:
    """Advanced risk management analytics for portfolio analysis."""

    def __init__(self):
        """Initialize risk management analytics."""
        pass

    def calculate_var_cvar(
        self, returns: List[float], confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).

        Args:
            returns: List of portfolio returns
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])

        Returns:
            Dict containing VaR and CVaR for each confidence level
        """
        try:
            if not returns or len(returns) < 10:
                return {"error": "Insufficient return data for VaR/CVaR calculation"}

            returns_array = np.array(returns)
            results = {}

            for confidence_level in confidence_levels:
                # Historical VaR (percentile method)
                var_historical = np.percentile(
                    returns_array, (1 - confidence_level) * 100
                )

                # Parametric VaR (assuming normal distribution)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                var_parametric = mean_return + std_return * stats.norm.ppf(
                    1 - confidence_level
                )

                # CVaR (Expected Shortfall) - average of returns below VaR
                cvar_historical = np.mean(
                    returns_array[returns_array <= var_historical]
                )

                # Monte Carlo VaR (using fitted distribution)
                # Fit normal distribution and simulate
                simulated_returns = np.random.normal(mean_return, std_return, 10000)
                var_monte_carlo = np.percentile(
                    simulated_returns, (1 - confidence_level) * 100
                )
                cvar_monte_carlo = np.mean(
                    simulated_returns[simulated_returns <= var_monte_carlo]
                )

                results[f"{confidence_level:.0%}"] = {
                    "var_historical": float(var_historical),
                    "var_parametric": float(var_parametric),
                    "var_monte_carlo": float(var_monte_carlo),
                    "cvar_historical": float(cvar_historical),
                    "cvar_monte_carlo": float(cvar_monte_carlo),
                }

            return results

        except Exception as e:
            logger.error(f"Error calculating VaR/CVaR: {e}")
            return {"error": str(e)}

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
        try:
            if not positions or not returns_data:
                return {"error": "Insufficient data for risk attribution"}

            # Create returns matrix
            assets = list(returns_data.keys())
            returns_matrix = pd.DataFrame(returns_data)

            if returns_matrix.empty:
                return {"error": "No valid returns data"}

            # Calculate covariance matrix
            cov_matrix = returns_matrix.cov().values

            # Extract weights from positions
            weights = []
            position_names = []

            for pos in positions:
                if "weight" in pos and "symbol" in pos:
                    weights.append(pos["weight"])
                    position_names.append(pos["symbol"])
                elif "amount" in pos and "value" in pos:
                    # Calculate weight from amount and value
                    total_value = sum(p.get("value", 0) for p in positions)
                    weight = pos["value"] / total_value if total_value > 0 else 0
                    weights.append(weight)
                    position_names.append(
                        pos.get("symbol", f"Asset_{len(position_names)}")
                    )

            if not weights:
                return {"error": "No valid position weights found"}

            weights = np.array(weights)

            # Ensure weights sum to 1
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)

            # Calculate portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Calculate marginal contribution to risk (MCR)
            mcr = np.dot(cov_matrix, weights) / portfolio_volatility

            # Calculate component contribution to risk (CCR)
            ccr = weights * mcr

            # Calculate percentage contribution to risk
            risk_contribution_pct = ccr / portfolio_volatility

            # Calculate diversification ratio
            individual_volatilities = np.sqrt(np.diag(cov_matrix))
            weighted_avg_volatility = np.dot(weights, individual_volatilities)
            diversification_ratio = weighted_avg_volatility / portfolio_volatility

            results = {
                "portfolio_volatility": float(portfolio_volatility),
                "diversification_ratio": float(diversification_ratio),
                "risk_attribution": [],
            }

            for i, asset in enumerate(position_names):
                results["risk_attribution"].append({
                    "asset": asset,
                    "weight": float(weights[i]) if i < len(weights) else 0.0,
                    "individual_volatility": float(individual_volatilities[i])
                    if i < len(individual_volatilities)
                    else 0.0,
                    "marginal_contribution": float(mcr[i]) if i < len(mcr) else 0.0,
                    "component_contribution": float(ccr[i]) if i < len(ccr) else 0.0,
                    "risk_contribution_pct": float(risk_contribution_pct[i])
                    if i < len(risk_contribution_pct)
                    else 0.0,
                })

            return results

        except Exception as e:
            logger.error(f"Error calculating portfolio risk attribution: {e}")
            return {"error": str(e)}

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
        try:
            if not equity_curve_data:
                return {"error": "No equity curve data provided"}

            df = pd.DataFrame(equity_curve_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            df["returns"] = df["equity"].pct_change().fillna(0)

            # Default historical stress scenarios
            if scenarios is None:
                scenarios = [
                    {
                        "name": "2008 Financial Crisis",
                        "description": "Severe market downturn scenario",
                        "shock_magnitude": -0.30,  # 30% decline
                        "duration_days": 180,
                        "recovery_days": 365,
                    },
                    {
                        "name": "COVID-19 Market Crash",
                        "description": "Rapid market decline and recovery",
                        "shock_magnitude": -0.35,  # 35% decline
                        "duration_days": 30,
                        "recovery_days": 90,
                    },
                    {
                        "name": "Flash Crash",
                        "description": "Sudden severe market drop",
                        "shock_magnitude": -0.20,  # 20% decline
                        "duration_days": 1,
                        "recovery_days": 30,
                    },
                    {
                        "name": "Prolonged Bear Market",
                        "description": "Extended period of negative returns",
                        "shock_magnitude": -0.50,  # 50% decline
                        "duration_days": 730,  # 2 years
                        "recovery_days": 1095,  # 3 years
                    },
                ]

            stress_results = []
            current_equity = df["equity"].iloc[-1]

            for scenario in scenarios:
                # Apply stress scenario
                shock_magnitude = scenario["shock_magnitude"]
                duration_days = scenario["duration_days"]
                recovery_days = scenario.get("recovery_days", duration_days)

                # Calculate stressed equity
                stressed_equity = current_equity * (1 + shock_magnitude)

                # Calculate maximum drawdown during stress
                max_drawdown = abs(shock_magnitude)

                # Estimate recovery time based on historical volatility
                historical_volatility = df["returns"].std() * np.sqrt(252)  # Annualized

                # Calculate stress metrics
                stress_var_95 = np.percentile(df["returns"], 5) * np.sqrt(duration_days)
                stress_var_99 = np.percentile(df["returns"], 1) * np.sqrt(duration_days)

                stress_results.append({
                    "scenario_name": scenario["name"],
                    "description": scenario["description"],
                    "shock_magnitude": shock_magnitude,
                    "duration_days": duration_days,
                    "recovery_days": recovery_days,
                    "current_equity": float(current_equity),
                    "stressed_equity": float(stressed_equity),
                    "equity_loss": float(current_equity - stressed_equity),
                    "max_drawdown": float(max_drawdown),
                    "stress_var_95": float(stress_var_95),
                    "stress_var_99": float(stress_var_99),
                    "estimated_recovery_time_days": float(recovery_days),
                    "probability_of_scenario": self._estimate_scenario_probability(
                        df["returns"], shock_magnitude, duration_days
                    ),
                })

            return {
                "stress_test_date": datetime.now().isoformat(),
                "portfolio_current_value": float(current_equity),
                "scenarios": stress_results,
                "summary": {
                    "worst_case_loss": float(
                        max([s["equity_loss"] for s in stress_results])
                    ),
                    "worst_case_drawdown": float(
                        max([s["max_drawdown"] for s in stress_results])
                    ),
                    "average_recovery_time": float(
                        np.mean([
                            s["estimated_recovery_time_days"] for s in stress_results
                        ])
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error performing stress testing: {e}")
            return {"error": str(e)}

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
        try:
            if not equity_curve_data:
                return {"error": "No equity curve data provided"}

            df = pd.DataFrame(equity_curve_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            df["returns"] = df["equity"].pct_change().fillna(0)

            # Calculate rolling statistics for regime detection
            window = min(30, len(df) // 4)  # Adaptive window size
            df["rolling_mean"] = df["returns"].rolling(window=window).mean()
            df["rolling_std"] = df["returns"].rolling(window=window).std()
            df["rolling_skew"] = df["returns"].rolling(window=window).skew()
            df["rolling_kurt"] = df["returns"].rolling(window=window).kurt()

            # Remove NaN values
            df_clean = df.dropna()

            if len(df_clean) < n_regimes * 10:  # Need sufficient data points
                return {"error": "Insufficient data for regime detection"}

            # Prepare features for clustering
            features = ["rolling_mean", "rolling_std", "rolling_skew", "rolling_kurt"]
            X = df_clean[features].values

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            df_clean = df_clean.copy()  # Avoid SettingWithCopyWarning
            df_clean["regime"] = kmeans.fit_predict(X_scaled)

            # Analyze each regime
            regime_analysis = []
            regime_names = ["Bull Market", "Bear Market", "Sideways Market"][:n_regimes]

            for i in range(n_regimes):
                regime_data = df_clean[df_clean["regime"] == i]

                if len(regime_data) > 0:
                    regime_analysis.append({
                        "regime_id": i,
                        "regime_name": regime_names[i]
                        if i < len(regime_names)
                        else f"Regime {i + 1}",
                        "periods": len(regime_data),
                        "percentage_of_time": float(
                            len(regime_data) / len(df_clean) * 100
                        ),
                        "avg_return": float(regime_data["returns"].mean()),
                        "volatility": float(regime_data["returns"].std()),
                        "sharpe_ratio": float(
                            regime_data["returns"].mean() / regime_data["returns"].std()
                            if regime_data["returns"].std() > 0
                            else 0
                        ),
                        "max_drawdown": float(
                            (
                                regime_data["equity"] / regime_data["equity"].cummax()
                                - 1
                            ).min()
                        ),
                        "start_date": regime_data["timestamp"].min().isoformat(),
                        "end_date": regime_data["timestamp"].max().isoformat(),
                    })

            # Calculate regime transition probabilities
            transitions = self._calculate_regime_transitions(df_clean["regime"].values)

            return {
                "analysis_date": datetime.now().isoformat(),
                "n_regimes": n_regimes,
                "total_periods": len(df_clean),
                "regimes": regime_analysis,
                "transition_matrix": transitions,
                "current_regime": int(df_clean["regime"].iloc[-1]),
                "regime_stability": float(
                    np.mean(np.diag(transitions))
                ),  # Average of diagonal elements
                "regime_data": df_clean[["timestamp", "returns", "regime"]].to_dict(
                    "records"
                ),
            }

        except Exception as e:
            logger.error(f"Error detecting market regimes: {e}")
            return {"error": str(e)}

    def _estimate_scenario_probability(
        self, returns: pd.Series, shock_magnitude: float, duration_days: int
    ) -> float:
        """Estimate the probability of a stress scenario occurring."""
        try:
            # Calculate rolling returns over the duration period
            rolling_returns = returns.rolling(window=duration_days).sum()

            # Count occurrences of returns worse than shock magnitude
            worse_outcomes = (rolling_returns <= shock_magnitude).sum()
            total_periods = len(rolling_returns.dropna())

            if total_periods == 0:
                return 0.0

            probability = worse_outcomes / total_periods
            return float(min(probability, 1.0))  # Cap at 100%

        except Exception:
            return 0.05  # Default 5% probability if calculation fails

    def _calculate_regime_transitions(self, regimes: np.ndarray) -> List[List[float]]:
        """Calculate regime transition probability matrix."""
        try:
            n_regimes = len(np.unique(regimes))
            transition_matrix = np.zeros((n_regimes, n_regimes))

            for i in range(len(regimes) - 1):
                current_regime = int(regimes[i])
                next_regime = int(regimes[i + 1])
                transition_matrix[current_regime, next_regime] += 1

            # Normalize to get probabilities
            for i in range(n_regimes):
                row_sum = transition_matrix[i].sum()
                if row_sum > 0:
                    transition_matrix[i] = transition_matrix[i] / row_sum

            return transition_matrix.tolist()

        except Exception:
            # Return uniform transition matrix as fallback
            n_regimes = len(np.unique(regimes))
            uniform_prob = 1.0 / n_regimes
            return [[uniform_prob] * n_regimes for _ in range(n_regimes)]

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
        try:
            var_cvar_results = self.calculate_var_cvar(returns, confidence_levels)

            if "error" in var_cvar_results:
                return var_cvar_results

            # Create histogram of returns
            fig = go.Figure()

            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name="Return Distribution",
                    opacity=0.7,
                    marker_color="lightblue",
                )
            )

            # Add VaR and CVaR lines for each confidence level
            colors = ["red", "darkred", "orange"]

            for i, confidence_level in enumerate(confidence_levels):
                level_key = f"{confidence_level:.0%}"
                if level_key in var_cvar_results:
                    var_hist = var_cvar_results[level_key]["var_historical"]
                    cvar_hist = var_cvar_results[level_key]["cvar_historical"]

                    # Add VaR line
                    fig.add_vline(
                        x=var_hist,
                        line_dash="dash",
                        line_color=colors[i % len(colors)],
                        annotation_text=f"VaR {level_key}: {var_hist:.3f}",
                        annotation_position="top",
                    )

                    # Add CVaR line
                    fig.add_vline(
                        x=cvar_hist,
                        line_dash="solid",
                        line_color=colors[i % len(colors)],
                        annotation_text=f"CVaR {level_key}: {cvar_hist:.3f}",
                        annotation_position="bottom",
                    )

            fig.update_layout(
                title="Value at Risk (VaR) and Conditional VaR (CVaR) Analysis",
                xaxis_title="Returns",
                yaxis_title="Frequency",
                showlegend=True,
                height=500,
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating VaR/CVaR chart: {e}")
            return {"error": str(e)}

    def create_stress_test_chart(
        self, stress_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create stress testing visualization chart.

        Args:
            stress_results: Results from stress testing analysis

        Returns:
            Dict containing Plotly figure as JSON
        """
        try:
            if "error" in stress_results or "scenarios" not in stress_results:
                return {"error": "Invalid stress test results"}

            scenarios = stress_results["scenarios"]

            # Create subplot figure
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Equity Loss by Scenario",
                    "Maximum Drawdown by Scenario",
                    "Recovery Time by Scenario",
                    "Scenario Probabilities",
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}],
                ],
            )

            scenario_names = [s["scenario_name"] for s in scenarios]
            equity_losses = [s["equity_loss"] for s in scenarios]
            max_drawdowns = [
                s["max_drawdown"] * 100 for s in scenarios
            ]  # Convert to percentage
            recovery_times = [s["estimated_recovery_time_days"] for s in scenarios]
            probabilities = [
                s["probability_of_scenario"] * 100 for s in scenarios
            ]  # Convert to percentage

            # Equity Loss
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=equity_losses,
                    name="Equity Loss ($)",
                    marker_color="red",
                ),
                row=1,
                col=1,
            )

            # Maximum Drawdown
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=max_drawdowns,
                    name="Max Drawdown (%)",
                    marker_color="orange",
                ),
                row=1,
                col=2,
            )

            # Recovery Time
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=recovery_times,
                    name="Recovery Time (Days)",
                    marker_color="blue",
                ),
                row=2,
                col=1,
            )

            # Scenario Probabilities
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=probabilities,
                    name="Probability (%)",
                    marker_color="green",
                ),
                row=2,
                col=2,
            )

            fig.update_layout(
                title="Stress Testing Analysis Results", height=800, showlegend=False
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Loss ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
            fig.update_yaxes(title_text="Days", row=2, col=1)
            fig.update_yaxes(title_text="Probability (%)", row=2, col=2)

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating stress test chart: {e}")
            return {"error": str(e)}

    def create_regime_analysis_chart(
        self, regime_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create market regime analysis visualization chart.

        Args:
            regime_results: Results from regime detection analysis

        Returns:
            Dict containing Plotly figure as JSON
        """
        try:
            if "error" in regime_results or "regime_data" not in regime_results:
                return {"error": "Invalid regime analysis results"}

            regime_data = pd.DataFrame(regime_results["regime_data"])
            regime_data["timestamp"] = pd.to_datetime(regime_data["timestamp"])

            # Create subplot figure
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(
                    "Returns Over Time with Regime Classification",
                    "Regime Timeline",
                    "Regime Statistics",
                ),
                row_heights=[0.4, 0.3, 0.3],
            )

            # Color map for regimes
            regime_colors = ["blue", "red", "green", "orange", "purple"]

            # Plot returns with regime coloring
            for regime_id in regime_data["regime"].unique():
                regime_subset = regime_data[regime_data["regime"] == regime_id]
                regime_info = next(
                    (
                        r
                        for r in regime_results["regimes"]
                        if r["regime_id"] == regime_id
                    ),
                    None,
                )
                regime_name = (
                    regime_info["regime_name"] if regime_info else f"Regime {regime_id}"
                )

                fig.add_trace(
                    go.Scatter(
                        x=regime_subset["timestamp"],
                        y=regime_subset["returns"],
                        mode="markers",
                        name=regime_name,
                        marker=dict(
                            color=regime_colors[regime_id % len(regime_colors)], size=4
                        ),
                    ),
                    row=1,
                    col=1,
                )

            # Create regime timeline
            fig.add_trace(
                go.Scatter(
                    x=regime_data["timestamp"],
                    y=regime_data["regime"],
                    mode="lines+markers",
                    name="Regime Timeline",
                    line=dict(width=2),
                    marker=dict(size=3),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            # Create regime statistics bar chart
            if "regimes" in regime_results:
                regimes = regime_results["regimes"]
                regime_names = [r["regime_name"] for r in regimes]
                avg_returns = [
                    r["avg_return"] * 100 for r in regimes
                ]  # Convert to percentage
                volatilities = [
                    r["volatility"] * 100 for r in regimes
                ]  # Convert to percentage

                fig.add_trace(
                    go.Bar(
                        x=regime_names,
                        y=avg_returns,
                        name="Avg Return (%)",
                        marker_color="lightblue",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )

            fig.update_layout(
                title="Market Regime Analysis", height=900, hovermode="x unified"
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Returns", row=1, col=1)
            fig.update_yaxes(title_text="Regime ID", row=2, col=1)
            fig.update_yaxes(title_text="Avg Return (%)", row=3, col=1)
            fig.update_xaxes(title_text="Date", row=3, col=1)

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating regime analysis chart: {e}")
            return {"error": str(e)}

    def create_risk_attribution_chart(
        self, attribution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create portfolio risk attribution visualization chart.

        Args:
            attribution_results: Results from risk attribution analysis

        Returns:
            Dict containing Plotly figure as JSON
        """
        try:
            if (
                "error" in attribution_results
                or "risk_attribution" not in attribution_results
            ):
                return {"error": "Invalid risk attribution results"}

            attribution_data = attribution_results["risk_attribution"]

            # Create subplot figure
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Portfolio Weights",
                    "Risk Contribution (%)",
                    "Individual vs Portfolio Volatility",
                    "Marginal Risk Contribution",
                ),
                specs=[
                    [{"type": "pie"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}],
                ],
            )

            assets = [item["asset"] for item in attribution_data]
            weights = [
                item["weight"] * 100 for item in attribution_data
            ]  # Convert to percentage
            risk_contributions = [
                item["risk_contribution_pct"] * 100 for item in attribution_data
            ]
            individual_vols = [
                item["individual_volatility"] * 100 for item in attribution_data
            ]
            marginal_contributions = [
                item["marginal_contribution"] for item in attribution_data
            ]

            # Portfolio weights pie chart
            fig.add_trace(
                go.Pie(labels=assets, values=weights, name="Weights"), row=1, col=1
            )

            # Risk contribution bar chart
            fig.add_trace(
                go.Bar(
                    x=assets,
                    y=risk_contributions,
                    name="Risk Contribution (%)",
                    marker_color="red",
                ),
                row=1,
                col=2,
            )

            # Individual volatility comparison
            portfolio_vol = attribution_results.get("portfolio_volatility", 0) * 100
            fig.add_trace(
                go.Bar(
                    x=assets,
                    y=individual_vols,
                    name="Individual Volatility (%)",
                    marker_color="blue",
                ),
                row=2,
                col=1,
            )

            # Add portfolio volatility line
            fig.add_hline(
                y=portfolio_vol,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Portfolio Vol: {portfolio_vol:.2f}%",
                row=2,
                col=1,
            )

            # Marginal risk contribution
            fig.add_trace(
                go.Bar(
                    x=assets,
                    y=marginal_contributions,
                    name="Marginal Contribution",
                    marker_color="orange",
                ),
                row=2,
                col=2,
            )

            fig.update_layout(
                title="Portfolio Risk Attribution Analysis",
                height=800,
                showlegend=False,
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Risk Contribution (%)", row=1, col=2)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            fig.update_yaxes(title_text="Marginal Contribution", row=2, col=2)

            return fig.to_dict()

        except Exception as e:
            logger.error(f"Error creating risk attribution chart: {e}")
            return {"error": str(e)}
