import pandas as pd


def strategy_simple_arb(
    df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    initial_capital: float,
):
    position = 0.0
    cash = initial_capital
    trades = []
    portfolio = [{"date": df.iloc[0]["timestamp"], "value": float(cash)}]

    for i, row in df.iterrows():
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
