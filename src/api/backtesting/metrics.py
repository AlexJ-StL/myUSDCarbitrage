"""Performance metrics calculator for backtesting."""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("backtesting.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class PerformanceMetrics:
    """Performance metrics calculator for backtesting."""

    def __init__(self, risk_free_rate: float = 0.0):
        """Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 0.0)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
    ) -> Dict[str, Union[float, Dict]]:
        """Calculate comprehensive performance metrics.

        Args:
            equity_curve: Equity curve as pandas Series with datetime index
            trades: DataFrame of trades (optional)
            benchmark: Benchmark returns as pandas Series with same index as equity_curve (optional)

        Returns:
            Dict[str, Union[float, Dict]]: Performance metrics
        """
        # Ensure equity curve is sorted by date
        equity_curve = equity_curve.sort_index()

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Calculate benchmark returns if provided
        benchmark_returns = None
        if benchmark is not None:
            benchmark = benchmark.sort_index()
            benchmark_returns = benchmark.pct_change().dropna()

            # Align benchmark with equity curve
            common_index = returns.index.intersection(benchmark_returns.index)
            returns = returns.loc[common_index]
            benchmark_returns = benchmark_returns.loc[common_index]

        # Calculate basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Calculate drawdown metrics
        drawdown_info = self.calculate_drawdown_metrics(equity_curve)

        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(returns)

        # Calculate return metrics
        return_metrics = self.calculate_return_metrics(
            equity_curve, returns, risk_metrics["annualized_volatility"]
        )

        # Calculate trade metrics if trades provided
        trade_metrics = {}
        if trades is not None:
            trade_metrics = self.calculate_trade_metrics(trades)

        # Calculate benchmark comparison if benchmark provided
        benchmark_metrics = {}
        if benchmark_returns is not None:
            benchmark_metrics = self.calculate_benchmark_metrics(
                returns, benchmark_returns
            )

        # Combine all metrics
        metrics = {
            "total_return": total_return,
            "drawdown": drawdown_info,
            "risk": risk_metrics,
            "returns": return_metrics,
            "trades": trade_metrics,
            "benchmark": benchmark_metrics,
        }

        return metrics

    def calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate drawdown metrics.

        Args:
            equity_curve: Equity curve as pandas Series with datetime index

        Returns:
            Dict[str, float]: Drawdown metrics
        """
        # Calculate drawdown series
        peak = equity_curve.cummax()
        drawdown = (equity_curve / peak) - 1

        # Calculate maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_idx = drawdown.idxmin()

        # Find the peak before the max drawdown
        peak_idx = equity_curve[:max_drawdown_idx].idxmax()

        # Find the recovery after the max drawdown
        recovery_idx = None
        if max_drawdown_idx < equity_curve.index[-1]:
            # Find where equity exceeds the previous peak
            recovery_series = equity_curve[max_drawdown_idx:] >= peak[max_drawdown_idx]
            if recovery_series.any():
                recovery_idx = recovery_series[recovery_series].index[0]

        # Calculate drawdown duration
        if peak_idx and max_drawdown_idx:
            drawdown_duration = (max_drawdown_idx - peak_idx).days
        else:
            drawdown_duration = 0

        # Calculate recovery duration
        if recovery_idx and max_drawdown_idx:
            recovery_duration = (recovery_idx - max_drawdown_idx).days
        else:
            recovery_duration = None

        # Calculate underwater periods (drawdowns > 5%)
        underwater_periods = []
        threshold = -0.05  # 5% drawdown threshold

        in_drawdown = False
        start_idx = None

        for idx, dd in drawdown.items():
            if not in_drawdown and dd < threshold:
                # Start of underwater period
                in_drawdown = True
                start_idx = idx
            elif in_drawdown and dd >= threshold:
                # End of underwater period
                in_drawdown = False
                underwater_periods.append({
                    "start": start_idx,
                    "end": idx,
                    "duration": (idx - start_idx).days,
                    "max_drawdown": drawdown[start_idx:idx].min(),
                })
                start_idx = None

        # Check if still in drawdown at the end
        if in_drawdown:
            underwater_periods.append({
                "start": start_idx,
                "end": drawdown.index[-1],
                "duration": (drawdown.index[-1] - start_idx).days,
                "max_drawdown": drawdown[start_idx:].min(),
            })

        # Calculate average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

        # Calculate Ulcer Index (square root of the mean of squared drawdowns)
        ulcer_index = (
            np.sqrt(np.mean(np.square(drawdown[drawdown < 0])))
            if (drawdown < 0).any()
            else 0
        )

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_date": max_drawdown_idx,
            "peak_date": peak_idx,
            "recovery_date": recovery_idx,
            "drawdown_duration": drawdown_duration,
            "recovery_duration": recovery_duration,
            "underwater_periods": underwater_periods,
            "avg_drawdown": avg_drawdown,
            "ulcer_index": ulcer_index,
        }

    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics.

        Args:
            returns: Returns series

        Returns:
            Dict[str, float]: Risk metrics
        """
        # Calculate volatility (annualized)
        trading_days = 252  # Assuming daily returns
        if len(returns) > 1:
            if (
                returns.index.freq == "D"
                or (returns.index[1] - returns.index[0]).days >= 1
            ):
                trading_days = 252
            elif (
                returns.index.freq == "H"
                or (returns.index[1] - returns.index[0]).seconds >= 3600
            ):
                trading_days = 252 * 24
            elif (
                returns.index.freq == "min"
                or (returns.index[1] - returns.index[0]).seconds >= 60
            ):
                trading_days = 252 * 24 * 60

        volatility = returns.std() * np.sqrt(trading_days)

        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(trading_days)
            if len(downside_returns) > 0
            else 0
        )

        # Calculate semi-deviation (only returns below mean)
        below_mean_returns = returns[returns < returns.mean()]
        semi_deviation = (
            below_mean_returns.std() * np.sqrt(trading_days)
            if len(below_mean_returns) > 0
            else 0
        )

        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)  # 95% VaR
        var_99 = np.percentile(returns, 1)  # 99% VaR

        # Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        cvar_95 = (
            returns[returns <= var_95].mean()
            if len(returns[returns <= var_95]) > 0
            else var_95
        )
        cvar_99 = (
            returns[returns <= var_99].mean()
            if len(returns[returns <= var_99]) > 0
            else var_99
        )

        # Calculate skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Calculate maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        return {
            "annualized_volatility": volatility,
            "downside_deviation": downside_deviation,
            "semi_deviation": semi_deviation,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "max_consecutive_losses": max_consecutive_losses,
        }

    def calculate_return_metrics(
        self, equity_curve: pd.Series, returns: pd.Series, volatility: float
    ) -> Dict[str, float]:
        """Calculate return metrics.

        Args:
            equity_curve: Equity curve as pandas Series with datetime index
            returns: Returns series
            volatility: Annualized volatility

        Returns:
            Dict[str, float]: Return metrics
        """
        # Calculate annualized return
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25

        if years > 0:
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            cagr = (1 + total_return) ** (1 / years) - 1
        else:
            cagr = 0

        # Calculate Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily excess returns
        sharpe_ratio = (
            excess_returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0
            else 0
        )

        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = (
            excess_returns.mean() / downside_returns.std() * np.sqrt(252)
            if len(downside_returns) > 0 and downside_returns.std() > 0
            else 0
        )

        # Calculate Calmar ratio
        drawdown_info = self.calculate_drawdown_metrics(equity_curve)
        max_drawdown = drawdown_info["max_drawdown"]
        calmar_ratio = abs(cagr / max_drawdown) if max_drawdown < 0 else float("inf")

        # Calculate MAR ratio (CAGR / Max Drawdown)
        mar_ratio = abs(cagr / max_drawdown) if max_drawdown < 0 else float("inf")

        # Calculate Omega ratio
        threshold = 0  # Target return threshold
        upside_returns = returns[returns > threshold]
        downside_returns = returns[returns <= threshold]

        if len(downside_returns) > 0 and abs(downside_returns.sum()) > 0:
            omega_ratio = upside_returns.sum() / abs(downside_returns.sum())
        else:
            omega_ratio = float("inf")

        # Calculate gain-to-pain ratio
        if len(downside_returns) > 0 and abs(downside_returns.sum()) > 0:
            gain_to_pain = returns.sum() / abs(downside_returns.sum())
        else:
            gain_to_pain = float("inf")

        # Calculate Kestner ratio (K-ratio)
        log_returns = np.log(1 + returns)
        cumulative_log_returns = log_returns.cumsum()

        if len(cumulative_log_returns) > 1:
            # Linear regression of cumulative log returns
            x = np.arange(len(cumulative_log_returns))
            slope, _, r_value, _, _ = stats.linregress(x, cumulative_log_returns)
            kestner_ratio = r_value * np.sqrt(len(cumulative_log_returns))
        else:
            kestner_ratio = 0

        # Calculate information ratio (if benchmark provided)
        information_ratio = (
            None  # Will be calculated in benchmark metrics if benchmark provided
        )

        # Calculate compound growth metrics
        compound_returns = (1 + returns).cumprod()
        peak_equity = compound_returns.cummax()
        drawdown_pct = (compound_returns / peak_equity) - 1

        # Calculate average monthly returns
        if returns.index.freq == "D" or (returns.index[1] - returns.index[0]).days >= 1:
            # Convert daily returns to monthly
            monthly_returns = (1 + returns).resample("M").prod() - 1
        else:
            # For higher frequency data, first convert to daily then to monthly
            daily_returns = (1 + returns).resample("D").prod() - 1
            monthly_returns = (1 + daily_returns).resample("M").prod() - 1

        avg_monthly_return = monthly_returns.mean()
        monthly_volatility = monthly_returns.std()

        return {
            "cagr": cagr,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "mar_ratio": mar_ratio,
            "omega_ratio": omega_ratio,
            "gain_to_pain": gain_to_pain,
            "kestner_ratio": kestner_ratio,
            "avg_monthly_return": avg_monthly_return,
            "monthly_volatility": monthly_volatility,
        }

    def calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade metrics.

        Args:
            trades: DataFrame of trades with columns: entry_price, exit_price, entry_time, exit_time, pnl, etc.

        Returns:
            Dict[str, float]: Trade metrics
        """
        if len(trades) == 0:
            return {}

        # Calculate win rate
        winning_trades = trades[trades["pnl"] > 0]
        losing_trades = trades[trades["pnl"] <= 0]

        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0

        # Calculate average trade metrics
        avg_trade_pnl = trades["pnl"].mean()
        avg_winning_trade = (
            winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
        )
        avg_losing_trade = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0

        # Calculate profit factor
        gross_profit = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Calculate expectancy
        expectancy = win_rate * avg_winning_trade + (1 - win_rate) * avg_losing_trade

        # Calculate average holding time
        if "entry_time" in trades.columns and "exit_time" in trades.columns:
            holding_times = (
                trades["exit_time"] - trades["entry_time"]
            ).dt.total_seconds() / 3600  # Hours
            avg_holding_time = holding_times.mean()
            max_holding_time = holding_times.max()
            min_holding_time = holding_times.min()
        else:
            avg_holding_time = None
            max_holding_time = None
            min_holding_time = None

        # Calculate maximum consecutive wins/losses
        consecutive_wins = 0
        max_consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0

        for pnl in trades["pnl"]:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

        # Calculate largest win/loss
        largest_win = winning_trades["pnl"].max() if len(winning_trades) > 0 else 0
        largest_loss = losing_trades["pnl"].min() if len(losing_trades) > 0 else 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_trade_pnl": avg_trade_pnl,
            "avg_winning_trade": avg_winning_trade,
            "avg_losing_trade": avg_losing_trade,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_holding_time": avg_holding_time,
            "max_holding_time": max_holding_time,
            "min_holding_time": min_holding_time,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
        }

    def calculate_benchmark_metrics(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate benchmark comparison metrics.

        Args:
            returns: Strategy returns series
            benchmark_returns: Benchmark returns series

        Returns:
            Dict[str, float]: Benchmark metrics
        """
        # Calculate alpha and beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Calculate alpha (Jensen's alpha)
        risk_free_daily = self.risk_free_rate / 252
        alpha = (
            returns.mean()
            - risk_free_daily
            - beta * (benchmark_returns.mean() - risk_free_daily)
        ) * 252

        # Calculate information ratio
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = (
            (returns.mean() - benchmark_returns.mean()) * 252 / tracking_error
            if tracking_error > 0
            else 0
        )

        # Calculate up/down capture ratios
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0

        if up_market.any():
            up_capture = returns[up_market].mean() / benchmark_returns[up_market].mean()
        else:
            up_capture = 0

        if down_market.any():
            down_capture = (
                returns[down_market].mean() / benchmark_returns[down_market].mean()
            )
        else:
            down_capture = 0

        # Calculate correlation
        correlation = returns.corr(benchmark_returns)

        # Calculate R-squared
        r_squared = correlation**2

        # Calculate outperformance
        outperformance = (1 + returns).prod() / (1 + benchmark_returns).prod() - 1

        return {
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "up_capture": up_capture,
            "down_capture": down_capture,
            "correlation": correlation,
            "r_squared": r_squared,
            "outperformance": outperformance,
            "tracking_error": tracking_error,
        }

    def calculate_regime_metrics(
        self, returns: pd.Series, regime_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics across different market regimes.

        Args:
            returns: Strategy returns series
            regime_data: DataFrame with regime classifications

        Returns:
            Dict[str, Dict[str, float]]: Metrics by regime
        """
        regime_metrics = {}

        # Ensure indices are aligned
        common_index = returns.index.intersection(regime_data.index)
        returns = returns.loc[common_index]
        regime_data = regime_data.loc[common_index]

        # Calculate metrics for each regime
        for regime in regime_data["regime"].unique():
            regime_returns = returns[regime_data["regime"] == regime]

            if len(regime_returns) > 0:
                # Calculate basic metrics for this regime
                total_return = (1 + regime_returns).prod() - 1
                volatility = regime_returns.std() * np.sqrt(252)
                sharpe = (
                    regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                    if regime_returns.std() > 0
                    else 0
                )

                # Calculate win rate in this regime
                win_rate = (regime_returns > 0).mean()

                regime_metrics[regime] = {
                    "total_return": total_return,
                    "annualized_return": total_return * (252 / len(regime_returns)),
                    "volatility": volatility,
                    "sharpe_ratio": sharpe,
                    "win_rate": win_rate,
                    "num_periods": len(regime_returns),
                }

        return regime_metrics
