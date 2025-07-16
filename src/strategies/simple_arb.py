"""Simple arbitrage strategy implementation."""

from typing import Any

import pandas as pd


def strategy_simple_arb(
    df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    initial_capital: float,
) -> dict[str, Any]:
    """
    Execute a simple arbitrage strategy based on price thresholds.

    Args:
        df: DataFrame with OHLCV data including 'timestamp' and 'close' columns
        buy_threshold: Price level below which to buy USDC
        sell_threshold: Price level above which to sell USDC
        initial_capital: Starting capital amount

    Returns:
        Dictionary containing trades and portfolio value history
    """
    position = 0.0
    cash = initial_capital
    trades = []
    portfolio = [{"date": df.iloc[0]["timestamp"], "value": float(cash)}]

    for _, row in df.iterrows():
        price = row["close"]

        if price <= buy_threshold and cash > 0:
            position = cash / price
            cash = 0.0
            trades.append(
                {
                    "type": "buy",
                    "datetime": row["timestamp"],
                    "price": price,
                    "position": position,
                }
            )

        elif price >= sell_threshold and position > 0:
            cash = position * price
            position = 0.0
            trades.append(
                {
                    "type": "sell",
                    "datetime": row["timestamp"],
                    "price": price,
                    "position": position,
                }
            )

        portfolio_value = cash + (position * price)
        portfolio.append({"date": row["timestamp"], "value": portfolio_value})

    return {"trades": trades, "portfolio": portfolio}
