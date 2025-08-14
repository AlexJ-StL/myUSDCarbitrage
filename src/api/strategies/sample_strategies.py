"""Sample strategies for USDC arbitrage backtesting."""

from datetime import datetime
from typing import Any, Dict, List

from ..backtesting import OrderSide, Portfolio


def simple_arbitrage_strategy(
    timestamp: datetime,
    portfolio: Portfolio,
    market_data: Dict[str, Dict[str, Any]],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Simple arbitrage strategy that buys on the exchange with lowest price and sells on the highest.

    Args:
        timestamp: Current timestamp
        portfolio: Portfolio object
        market_data: Market data for all exchanges and symbols
        params: Strategy parameters

    Returns:
        List[Dict[str, Any]]: List of trading signals
    """
    signals = []

    # Extract parameters
    min_spread = params.get("min_spread", 0.001)  # Minimum spread to trade (0.1%)
    max_position = params.get("max_position", 1000)  # Maximum position size

    # Find lowest and highest prices across exchanges for each symbol
    symbols = set()
    for key in market_data:
        _, symbol = key.split(":")
        symbols.add(symbol)

    for symbol in symbols:
        # Collect prices for this symbol across exchanges
        exchange_prices = {}
        for key, data in market_data.items():
            exchange, sym = key.split(":")
            if sym == symbol:
                exchange_prices[exchange] = data["close"]

        if len(exchange_prices) < 2:
            # Need at least 2 exchanges to do arbitrage
            continue

        # Find lowest and highest prices
        lowest_exchange = min(exchange_prices, key=exchange_prices.get)
        highest_exchange = max(exchange_prices, key=exchange_prices.get)

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

            # Generate signals
            if buy_amount > 0:
                signals.append({
                    "exchange": lowest_exchange,
                    "symbol": symbol,
                    "side": OrderSide.BUY,
                    "amount": buy_amount,
                })

            if sell_amount > 0:
                signals.append({
                    "exchange": highest_exchange,
                    "symbol": symbol,
                    "side": OrderSide.SELL,
                    "amount": sell_amount,
                })

    return signals


def mean_reversion_strategy(
    timestamp: datetime,
    portfolio: Portfolio,
    market_data: Dict[str, Dict[str, Any]],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Mean reversion strategy that buys when price is below moving average and sells when above.

    Args:
        timestamp: Current timestamp
        portfolio: Portfolio object
        market_data: Market data for all exchanges and symbols
        params: Strategy parameters

    Returns:
        List[Dict[str, Any]]: List of trading signals
    """
    signals = []

    # Extract parameters
    lookback = params.get("lookback", 20)  # Lookback period for moving average
    entry_threshold = params.get("entry_threshold", 0.01)  # Entry threshold (1%)
    exit_threshold = params.get("exit_threshold", 0.005)  # Exit threshold (0.5%)

    # We need historical data for this strategy, which is not available in this simple example
    # In a real implementation, we would fetch historical data from the database
    # For now, we'll just return an empty list

    return signals


def trend_following_strategy(
    timestamp: datetime,
    portfolio: Portfolio,
    market_data: Dict[str, Dict[str, Any]],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Trend following strategy that buys when price is above moving average and sells when below.

    Args:
        timestamp: Current timestamp
        portfolio: Portfolio object
        market_data: Market data for all exchanges and symbols
        params: Strategy parameters

    Returns:
        List[Dict[str, Any]]: List of trading signals
    """
    signals = []

    # Extract parameters
    lookback = params.get("lookback", 20)  # Lookback period for moving average
    entry_threshold = params.get("entry_threshold", 0.01)  # Entry threshold (1%)
    exit_threshold = params.get("exit_threshold", 0.005)  # Exit threshold (0.5%)

    # We need historical data for this strategy, which is not available in this simple example
    # In a real implementation, we would fetch historical data from the database
    # For now, we'll just return an empty list

    return signals
