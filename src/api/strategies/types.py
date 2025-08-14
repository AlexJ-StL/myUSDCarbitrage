# Copyright 2025 USDC Arbitrage Application
"""Advanced strategy types for USDC arbitrage application."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..backtesting import OrderSide, Portfolio

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("strategies.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseStrategy(ABC):
    """Base class for all strategy types."""

    def __init__(self, parameters: Dict[str, Any]):
        """Initialize strategy with parameters.

        Args:
            parameters: Strategy parameters
        """
        self.parameters = parameters
        self.name = "BaseStrategy"
        self.description = "Base strategy class"

    @abstractmethod
    def generate_signals(
        self,
        timestamp: datetime,
        portfolio: Portfolio,
        market_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate trading signals.

        Args:
            timestamp: Current timestamp
            portfolio: Portfolio object
            market_data: Market data for all exchanges and symbols

        Returns:
            List[Dict[str, Any]]: List of trading signals
        """
        pass

    def __call__(
        self,
        timestamp: datetime,
        portfolio: Portfolio,
        market_data: Dict[str, Dict[str, Any]],
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Make strategy callable for backtesting engine.

        Args:
            timestamp: Current timestamp
            portfolio: Portfolio object
            market_data: Market data for all exchanges and symbols
            params: Strategy parameters (will override self.parameters)

        Returns:
            List[Dict[str, Any]]: List of trading signals
        """
        # Override parameters if provided
        if params:
            merged_params = self.parameters.copy()
            merged_params.update(params)
            self.parameters = merged_params

        return self.generate_signals(timestamp, portfolio, market_data)

    def get_callable(self) -> callable:
        """Get callable function for backtesting engine.

        Returns:
            callable: Strategy function
        """
        return self.__call__


class ArbitrageStrategy(BaseStrategy):
    """Arbitrage strategy that exploits price differences between exchanges."""

    def __init__(self, parameters: Dict[str, Any]):
        """Initialize arbitrage strategy.

        Args:
            parameters: Strategy parameters
        """
        super().__init__(parameters)
        self.name = "ArbitrageStrategy"
        self.description = "Exploits price differences between exchanges"

        # Set default parameters if not provided
        self.parameters.setdefault("min_spread", 0.001)  # 0.1% minimum spread
        self.parameters.setdefault("max_position", 1000.0)  # Maximum position size
        self.parameters.setdefault("max_positions", 5)  # Maximum number of positions
        self.parameters.setdefault("exchanges", [])  # Exchanges to consider

    def generate_signals(
        self,
        timestamp: datetime,
        portfolio: Portfolio,
        market_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate arbitrage signals.

        Args:
            timestamp: Current timestamp
            portfolio: Portfolio object
            market_data: Market data for all exchanges and symbols

        Returns:
            List[Dict[str, Any]]: List of trading signals
        """
        signals = []
        min_spread = self.parameters["min_spread"]
        max_position = self.parameters["max_position"]
        max_positions = self.parameters["max_positions"]
        allowed_exchanges = self.parameters["exchanges"]

        # Extract symbols from market data
        symbols = set()
        for key in market_data:
            exchange, symbol = key.split(":")
            symbols.add(symbol)

        # Check each symbol for arbitrage opportunities
        for symbol in symbols:
            # Collect prices for this symbol across exchanges
            exchange_prices = {}
            for key, data in market_data.items():
                exchange, sym = key.split(":")
                if sym == symbol and (
                    not allowed_exchanges or exchange in allowed_exchanges
                ):
                    exchange_prices[exchange] = data["close"]

            if len(exchange_prices) < 2:
                # Need at least 2 exchanges for arbitrage
                continue

            # Find lowest and highest prices
            lowest_exchange = min(exchange_prices, key=lambda x: exchange_prices[x])
            highest_exchange = max(exchange_prices, key=lambda x: exchange_prices[x])

            lowest_price = exchange_prices[lowest_exchange]
            highest_price = exchange_prices[highest_exchange]

            # Calculate spread
            spread = (highest_price - lowest_price) / lowest_price

            # Check if spread is large enough
            if spread > min_spread:
                # Check current positions
                lowest_position = portfolio.get_position(lowest_exchange, symbol)
                highest_position = portfolio.get_position(highest_exchange, symbol)

                # Calculate position sizes
                buy_amount = max_position / lowest_price
                sell_amount = max_position / highest_price

                if lowest_position:
                    # Adjust buy amount if we already have a position
                    buy_amount = max(0, buy_amount - lowest_position.amount)

                if highest_position:
                    # Adjust sell amount if we already have a position
                    sell_amount = min(highest_position.amount, sell_amount)

                # Check if we have too many positions
                current_positions = len(portfolio.positions)
                if current_positions >= max_positions:
                    # Skip if we already have maximum positions
                    continue

                # Generate signals
                if buy_amount > 0:
                    signals.append({
                        "exchange": lowest_exchange,
                        "symbol": symbol,
                        "side": OrderSide.BUY,
                        "amount": buy_amount,
                        "reason": f"Arbitrage: Buy at {lowest_exchange} (${lowest_price:.4f})",
                    })

                if sell_amount > 0:
                    signals.append({
                        "exchange": highest_exchange,
                        "symbol": symbol,
                        "side": OrderSide.SELL,
                        "amount": sell_amount,
                        "reason": f"Arbitrage: Sell at {highest_exchange} (${highest_price:.4f})",
                    })

        return signals


class TrendFollowingStrategy(BaseStrategy):
    """Trend following strategy that follows price momentum."""

    def __init__(self, parameters: dict[str, Any]) -> None:
        """Initialize trend following strategy.

        Args:
            parameters: Strategy parameters
        """
        super().__init__(parameters)
        self.name = "TrendFollowingStrategy"
        self.description = "Follows price momentum using moving averages"

        # Set default parameters if not provided
        self.parameters.setdefault("fast_ma", 10)  # Fast moving average period
        self.parameters.setdefault("slow_ma", 30)  # Slow moving average period
        self.parameters.setdefault(
            "position_size", 0.1
        )  # Position size as fraction of portfolio
        self.parameters.setdefault("stop_loss", 0.05)  # Stop loss percentage
        self.parameters.setdefault("take_profit", 0.1)  # Take profit percentage

        # Initialize state
        self.historical_data: dict[str, list[dict[str, Any]]] = {}

    def generate_signals(
        self,
        timestamp: datetime,
        portfolio: Portfolio,
        market_data: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate trend following signals.

        Args:
            timestamp: Current timestamp
            portfolio: Portfolio object
            market_data: Market data for all exchanges and symbols

        Returns:
            List of trading signals
        """
        signals = []
        fast_ma = self.parameters["fast_ma"]
        slow_ma = self.parameters["slow_ma"]
        position_size = self.parameters["position_size"]
        stop_loss = self.parameters["stop_loss"]
        take_profit = self.parameters["take_profit"]

        # Update historical data
        for key, data in market_data.items():
            if key not in self.historical_data:
                self.historical_data[key] = []

            # Add current data point
            self.historical_data[key].append({
                "timestamp": timestamp,
                "close": data["close"],
            })

            # Keep only necessary history
            max_history = max(fast_ma, slow_ma) + 10
            if len(self.historical_data[key]) > max_history:
                self.historical_data[key] = self.historical_data[key][-max_history:]

        # Generate signals for each market
        for key, history in self.historical_data.items():
            if len(history) < slow_ma + 1:
                # Not enough data yet
                continue

            exchange, symbol = key.split(":")

            # Calculate moving averages
            closes = [h["close"] for h in history]
            fast_ma_value = sum(closes[-fast_ma:]) / fast_ma
            slow_ma_value = sum(closes[-slow_ma:]) / slow_ma

            # Previous moving averages
            prev_closes = closes[:-1]
            prev_fast_ma = sum(prev_closes[-fast_ma:]) / fast_ma
            prev_slow_ma = sum(prev_closes[-slow_ma:]) / slow_ma

            # Current position
            position = portfolio.get_position(exchange, symbol)
            current_price = closes[-1]

            # Check for crossovers
            if fast_ma_value > slow_ma_value and prev_fast_ma <= prev_slow_ma:
                # Bullish crossover - buy signal
                if not position or position.amount <= 0:
                    # Calculate position size based on portfolio equity
                    equity = portfolio.calculate_equity()
                    amount = (equity * position_size) / current_price

                    signals.append({
                        "exchange": exchange,
                        "symbol": symbol,
                        "side": OrderSide.BUY,
                        "amount": amount,
                        "reason": f"Trend: Bullish crossover (Fast MA: {fast_ma_value:.4f}, Slow MA: {slow_ma_value:.4f})",
                    })

            elif fast_ma_value < slow_ma_value and prev_fast_ma >= prev_slow_ma:
                # Bearish crossover - sell signal
                if position and position.amount > 0:
                    signals.append({
                        "exchange": exchange,
                        "symbol": symbol,
                        "side": OrderSide.SELL,
                        "amount": position.amount,
                        "reason": f"Trend: Bearish crossover (Fast MA: {fast_ma_value:.4f}, Slow MA: {slow_ma_value:.4f})",
                    })

            # Check stop loss and take profit for existing positions
            if position and position.amount > 0:
                # Calculate profit/loss percentage
                pnl_pct = (current_price - position.entry_price) / position.entry_price

                if pnl_pct <= -stop_loss:
                    # Stop loss triggered
                    signals.append({
                        "exchange": exchange,
                        "symbol": symbol,
                        "side": OrderSide.SELL,
                        "amount": position.amount,
                        "reason": f"Trend: Stop loss triggered ({pnl_pct:.2%})",
                    })
                elif pnl_pct >= take_profit:
                    # Take profit triggered
                    signals.append({
                        "exchange": exchange,
                        "symbol": symbol,
                        "side": OrderSide.SELL,
                        "amount": position.amount,
                        "reason": f"Trend: Take profit triggered ({pnl_pct:.2%})",
                    })

        return signals


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy that trades price reversals to the mean."""

    def __init__(self, parameters: dict[str, Any]) -> None:
        """Initialize mean reversion strategy.

        Args:
            parameters: Strategy parameters
        """
        super().__init__(parameters)
        self.name = "MeanReversionStrategy"
        self.description = "Trades price reversals to the mean"

        # Set default parameters if not provided
        self.parameters.setdefault("lookback", 20)  # Lookback period
        self.parameters.setdefault(
            "entry_threshold", 2.0
        )  # Entry threshold in standard deviations
        self.parameters.setdefault(
            "exit_threshold", 0.5
        )  # Exit threshold in standard deviations
        self.parameters.setdefault(
            "position_size", 0.1
        )  # Position size as fraction of portfolio
        self.parameters.setdefault("max_positions", 5)  # Maximum number of positions

        # Initialize state
        self.historical_data: dict[str, list[dict[str, Any]]] = {}

    def generate_signals(
        self,
        timestamp: datetime,
        portfolio: Portfolio,
        market_data: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate mean reversion signals.

        Args:
            timestamp: Current timestamp
            portfolio: Portfolio object
            market_data: Market data for all exchanges and symbols

        Returns:
            List of trading signals
        """
        signals = []
        lookback = self.parameters["lookback"]
        entry_threshold = self.parameters["entry_threshold"]
        exit_threshold = self.parameters["exit_threshold"]
        position_size = self.parameters["position_size"]
        max_positions = self.parameters["max_positions"]

        # Update historical data
        for key, data in market_data.items():
            if key not in self.historical_data:
                self.historical_data[key] = []

            # Add current data point
            self.historical_data[key].append({
                "timestamp": timestamp,
                "close": data["close"],
            })

            # Keep only necessary history
            max_history = lookback + 10
            if len(self.historical_data[key]) > max_history:
                self.historical_data[key] = self.historical_data[key][-max_history:]

        # Generate signals for each market
        for key, history in self.historical_data.items():
            if len(history) < lookback + 1:
                # Not enough data yet
                continue

            exchange, symbol = key.split(":")

            # Calculate mean and standard deviation
            closes = [
                h["close"] for h in history[-lookback - 1 : -1]
            ]  # Exclude current price
            mean = sum(closes) / len(closes)
            std_dev = np.std(closes)

            if std_dev == 0:
                # Avoid division by zero
                continue

            # Current price and z-score
            current_price = history[-1]["close"]
            z_score = (current_price - mean) / std_dev

            # Current position
            position = portfolio.get_position(exchange, symbol)

            # Check for entry and exit signals
            if z_score <= -entry_threshold:
                # Price is significantly below mean - buy signal
                if not position or position.amount <= 0:
                    # Check if we have too many positions
                    current_positions = len(portfolio.positions)
                    if current_positions >= max_positions:
                        # Skip if we already have maximum positions
                        continue

                    # Calculate position size based on portfolio equity
                    equity = portfolio.calculate_equity()
                    amount = (equity * position_size) / current_price

                    signals.append({
                        "exchange": exchange,
                        "symbol": symbol,
                        "side": OrderSide.BUY,
                        "amount": amount,
                        "reason": f"Mean Reversion: Buy signal (z-score: {z_score:.2f})",
                    })

            elif z_score >= entry_threshold:
                # Price is significantly above mean - sell signal
                if not position or position.amount >= 0:
                    # Check if we have too many positions
                    current_positions = len(portfolio.positions)
                    if current_positions >= max_positions:
                        # Skip if we already have maximum positions
                        continue

                    # Calculate position size based on portfolio equity
                    equity = portfolio.calculate_equity()
                    amount = (equity * position_size) / current_price

                    signals.append({
                        "exchange": exchange,
                        "symbol": symbol,
                        "side": OrderSide.SELL,
                        "amount": amount,
                        "reason": f"Mean Reversion: Sell signal (z-score: {z_score:.2f})",
                    })

            # Check for exit signals
            if position:
                if position.amount > 0 and z_score >= exit_threshold:
                    # Long position and price has reverted to mean - exit
                    signals.append({
                        "exchange": exchange,
                        "symbol": symbol,
                        "side": OrderSide.SELL,
                        "amount": position.amount,
                        "reason": f"Mean Reversion: Exit long position (z-score: {z_score:.2f})",
                    })
                elif position.amount < 0 and z_score <= -exit_threshold:
                    # Short position and price has reverted to mean - exit
                    signals.append({
                        "exchange": exchange,
                        "symbol": symbol,
                        "side": OrderSide.BUY,
                        "amount": abs(position.amount),
                        "reason": f"Mean Reversion: Exit short position (z-score: {z_score:.2f})",
                    })

        return signals


# Strategy factory for creating strategy instances
STRATEGY_TYPES = {
    "arbitrage": ArbitrageStrategy,
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def create_strategy(strategy_type: str, parameters: Dict[str, Any]) -> BaseStrategy:
    """Create a strategy instance.

    Args:
        strategy_type: Type of strategy to create
        parameters: Strategy parameters

    Returns:
        BaseStrategy: Strategy instance

    Raises:
        ValueError: If strategy type is not supported
    """
    if strategy_type not in STRATEGY_TYPES:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")

    strategy_class = STRATEGY_TYPES[strategy_type]
    return strategy_class(parameters)


def get_strategy_template(strategy_type: str) -> Dict[str, Any]:
    """Get default parameters template for a strategy type.

    Args:
        strategy_type: Type of strategy

    Returns:
        Dict[str, Any]: Default parameters template

    Raises:
        ValueError: If strategy type is not supported
    """
    if strategy_type not in STRATEGY_TYPES:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")

    # Create temporary instance to get default parameters
    temp_strategy = STRATEGY_TYPES[strategy_type]({})
    return temp_strategy.parameters.copy()
