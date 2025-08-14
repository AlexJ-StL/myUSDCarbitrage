"""Core backtesting engine for USDC arbitrage strategies."""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..database import DBConnector

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("backtesting.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class OrderType(Enum):
    """Order types for backtesting."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides for backtesting."""

    BUY = "buy"
    SELL = "sell"


class PositionSizing(Enum):
    """Position sizing strategies."""

    FIXED = "fixed"  # Fixed position size
    PERCENT = "percent"  # Percentage of portfolio
    KELLY = "kelly"  # Kelly criterion
    VOLATILITY = "volatility"  # Volatility-based sizing
    RISK_PARITY = "risk_parity"  # Risk parity allocation


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequencies."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    THRESHOLD = "threshold"  # Rebalance when allocation drifts beyond threshold


class Order:
    """Order representation for backtesting."""

    def __init__(
        self,
        exchange: str,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Initialize order.

        Args:
            exchange: Exchange identifier
            symbol: Trading pair symbol
            order_type: Order type (market, limit, etc.)
            side: Order side (buy or sell)
            amount: Order amount
            price: Order price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            timestamp: Order timestamp
        """
        self.exchange = exchange
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.amount = amount
        self.price = price
        self.stop_price = stop_price
        self.timestamp = timestamp or datetime.now()
        self.executed_price: Optional[float] = None
        self.executed_amount: float = 0.0
        self.fee: float = 0.0
        self.status = "open"
        self.execution_time: Optional[datetime] = None
        self.slippage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "order_type": self.order_type.value,
            "side": self.side.value,
            "amount": self.amount,
            "price": self.price,
            "stop_price": self.stop_price,
            "timestamp": self.timestamp.isoformat(),
            "executed_price": self.executed_price,
            "executed_amount": self.executed_amount,
            "fee": self.fee,
            "status": self.status,
            "execution_time": self.execution_time.isoformat()
            if self.execution_time
            else None,
            "slippage": self.slippage,
        }


class Position:
    """Position representation for backtesting."""

    def __init__(
        self,
        exchange: str,
        symbol: str,
        amount: float,
        entry_price: float,
        entry_time: datetime,
    ):
        """Initialize position.

        Args:
            exchange: Exchange identifier
            symbol: Trading pair symbol
            amount: Position amount (positive for long, negative for short)
            entry_price: Average entry price
            entry_time: Position entry time
        """
        self.exchange = exchange
        self.symbol = symbol
        self.amount = amount
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.fees_paid = 0.0

    def update_price(self, price: float) -> None:
        """Update position with new price.

        Args:
            price: New price
        """
        self.current_price = price
        self.unrealized_pnl = self.calculate_unrealized_pnl()

    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized profit and loss.

        Returns:
            float: Unrealized PnL
        """
        if self.amount == 0:
            return 0.0
        return self.amount * (self.current_price - self.entry_price)

    def close(self, price: float, amount: float, fee: float) -> float:
        """Close position partially or fully.

        Args:
            price: Closing price
            amount: Amount to close (positive value)
            fee: Fee paid for closing

        Returns:
            float: Realized PnL
        """
        if amount > abs(self.amount):
            amount = abs(self.amount)

        # Calculate realized PnL
        if self.amount > 0:  # Long position
            realized_pnl = amount * (price - self.entry_price) - fee
        else:  # Short position
            realized_pnl = amount * (self.entry_price - price) - fee

        # Update position
        if self.amount > 0:  # Long position
            self.amount -= amount
        else:  # Short position
            self.amount += amount

        self.realized_pnl += realized_pnl
        self.fees_paid += fee

        return realized_pnl

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
        }


class Portfolio:
    """Portfolio representation for backtesting."""

    def __init__(self, initial_balance: float = 10000.0):
        """Initialize portfolio.

        Args:
            initial_balance: Initial cash balance
        """
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions: Dict[str, Position] = {}  # Key: exchange:symbol
        self.closed_positions: List[Dict[str, Any]] = []
        self.orders: List[Order] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.transactions: List[Dict[str, Any]] = []

    def get_position_key(self, exchange: str, symbol: str) -> str:
        """Get position key.

        Args:
            exchange: Exchange identifier
            symbol: Trading pair symbol

        Returns:
            str: Position key
        """
        return f"{exchange}:{symbol}"

    def get_position(self, exchange: str, symbol: str) -> Optional[Position]:
        """Get position by exchange and symbol.

        Args:
            exchange: Exchange identifier
            symbol: Trading pair symbol

        Returns:
            Optional[Position]: Position if exists, None otherwise
        """
        key = self.get_position_key(exchange, symbol)
        return self.positions.get(key)

    def add_position(self, position: Position) -> None:
        """Add position to portfolio.

        Args:
            position: Position to add
        """
        key = self.get_position_key(position.exchange, position.symbol)
        self.positions[key] = position

    def update_position(
        self, exchange: str, symbol: str, price: float, timestamp: datetime
    ) -> None:
        """Update position with new price.

        Args:
            exchange: Exchange identifier
            symbol: Trading pair symbol
            price: New price
            timestamp: Update timestamp
        """
        position = self.get_position(exchange, symbol)
        if position:
            position.update_price(price)

        # Update equity curve
        equity = self.calculate_equity()
        self.equity_curve.append((timestamp, equity))

    def calculate_equity(self) -> float:
        """Calculate total portfolio equity.

        Returns:
            float: Total equity
        """
        positions_value = sum(
            position.amount * position.current_price
            for position in self.positions.values()
        )
        return self.cash + positions_value

    def add_transaction(
        self,
        exchange: str,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
        fee: float,
        timestamp: datetime,
    ) -> None:
        """Add transaction to portfolio history.

        Args:
            exchange: Exchange identifier
            symbol: Trading pair symbol
            side: Transaction side (buy or sell)
            amount: Transaction amount
            price: Transaction price
            fee: Transaction fee
            timestamp: Transaction timestamp
        """
        self.transactions.append({
            "exchange": exchange,
            "symbol": symbol,
            "side": side.value,
            "amount": amount,
            "price": price,
            "fee": fee,
            "timestamp": timestamp.isoformat(),
            "value": amount * price,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary."""
        return {
            "initial_balance": self.initial_balance,
            "cash": self.cash,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "closed_positions": self.closed_positions,
            "orders": [order.to_dict() for order in self.orders],
            "equity_curve": [
                {"timestamp": ts.isoformat(), "equity": eq}
                for ts, eq in self.equity_curve
            ],
            "transactions": self.transactions,
        }


class ExchangeFeeModel:
    """Exchange fee model for realistic transaction cost modeling."""

    def __init__(
        self,
        exchange: str,
        maker_fee: float = 0.001,  # 0.1% default
        taker_fee: float = 0.002,  # 0.2% default
        fixed_fee: float = 0.0,
        min_fee: float = 0.0,
        max_fee: Optional[float] = None,
        fee_currency: str = "USD",
        tiered_fees: Optional[List[Tuple[float, float]]] = None,
    ):
        """Initialize exchange fee model.

        Args:
            exchange: Exchange identifier
            maker_fee: Maker fee rate (default: 0.1%)
            taker_fee: Taker fee rate (default: 0.2%)
            fixed_fee: Fixed fee per trade
            min_fee: Minimum fee per trade
            max_fee: Maximum fee per trade
            fee_currency: Fee currency
            tiered_fees: Tiered fee structure [(volume_threshold, fee_rate), ...]
        """
        self.exchange = exchange
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.fixed_fee = fixed_fee
        self.min_fee = min_fee
        self.max_fee = max_fee
        self.fee_currency = fee_currency
        self.tiered_fees = tiered_fees or []

    def calculate_fee(
        self, order_type: OrderType, side: OrderSide, amount: float, price: float
    ) -> float:
        """Calculate fee for a trade.

        Args:
            order_type: Order type
            side: Order side
            amount: Order amount
            price: Order price

        Returns:
            float: Fee amount
        """
        trade_value = amount * price

        # Determine if maker or taker fee applies
        if order_type == OrderType.LIMIT:
            fee_rate = self.maker_fee
        else:
            fee_rate = self.taker_fee

        # Apply tiered fees if applicable
        if self.tiered_fees:
            for volume_threshold, tier_fee_rate in sorted(
                self.tiered_fees, key=lambda x: x[0], reverse=True
            ):
                if trade_value >= volume_threshold:
                    fee_rate = tier_fee_rate
                    break

        # Calculate fee
        fee = trade_value * fee_rate + self.fixed_fee

        # Apply min/max constraints
        if fee < self.min_fee:
            fee = self.min_fee
        if self.max_fee is not None and fee > self.max_fee:
            fee = self.max_fee

        return fee


class SlippageModel:
    """Slippage model for realistic trade execution simulation."""

    def __init__(
        self,
        base_slippage: float = 0.0005,  # 0.05% default
        volume_impact: float = 0.1,
        volatility_impact: float = 0.2,
        random_factor: float = 0.2,
        max_slippage: float = 0.01,  # 1% max slippage
    ):
        """Initialize slippage model.

        Args:
            base_slippage: Base slippage rate (default: 0.05%)
            volume_impact: Impact of volume on slippage (0-1)
            volatility_impact: Impact of volatility on slippage (0-1)
            random_factor: Random factor for slippage variation (0-1)
            max_slippage: Maximum slippage rate
        """
        self.base_slippage = base_slippage
        self.volume_impact = volume_impact
        self.volatility_impact = volatility_impact
        self.random_factor = random_factor
        self.max_slippage = max_slippage

    def calculate_slippage(
        self,
        side: OrderSide,
        amount: float,
        price: float,
        market_data: pd.DataFrame,
        timestamp: datetime,
    ) -> float:
        """Calculate slippage for a trade.

        Args:
            side: Order side
            amount: Order amount
            price: Order price
            market_data: Market data for slippage calculation
            timestamp: Order timestamp

        Returns:
            float: Executed price with slippage
        """
        # Find the closest data point to the timestamp
        market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
        closest_idx = (market_data["timestamp"] - timestamp).abs().idxmin()
        closest_data = market_data.iloc[closest_idx]

        # Calculate volume-based slippage
        volume = closest_data.get("volume", 0)
        if volume > 0:
            volume_ratio = min(amount / volume, 1.0)
        else:
            volume_ratio = 1.0
        volume_slippage = self.base_slippage * (1 + self.volume_impact * volume_ratio)

        # Calculate volatility-based slippage
        if len(market_data) >= 10:
            # Use rolling window for volatility calculation
            window = min(10, len(market_data))
            volatility = (
                market_data["close"].rolling(window=window).std().iloc[closest_idx]
            )
            normalized_volatility = volatility / price
            volatility_slippage = self.base_slippage * (
                1 + self.volatility_impact * normalized_volatility * 100
            )
        else:
            volatility_slippage = self.base_slippage

        # Combine slippage factors
        slippage_rate = (volume_slippage + volatility_slippage) / 2

        # Add random variation
        if self.random_factor > 0:
            random_variation = np.random.uniform(
                -self.random_factor, self.random_factor
            )
            slippage_rate *= 1 + random_variation

        # Apply maximum slippage constraint
        slippage_rate = min(slippage_rate, self.max_slippage)

        # Apply slippage based on order side
        if side == OrderSide.BUY:
            # Buy orders get executed at a higher price
            executed_price = price * (1 + slippage_rate)
        else:
            # Sell orders get executed at a lower price
            executed_price = price * (1 - slippage_rate)

        return executed_price


class PositionSizer:
    """Position sizing strategies for risk management."""

    def __init__(
        self,
        strategy: PositionSizing = PositionSizing.PERCENT,
        percent_size: float = 0.02,  # 2% of portfolio by default
        fixed_size: float = 1000.0,  # Fixed position size
        max_size: Optional[float] = None,  # Maximum position size
        kelly_fraction: float = 0.5,  # Kelly criterion fraction
        volatility_target: float = 0.01,  # 1% daily volatility target
        risk_target: float = 0.01,  # 1% risk per trade
    ):
        """Initialize position sizer.

        Args:
            strategy: Position sizing strategy
            percent_size: Percentage of portfolio for each position
            fixed_size: Fixed position size
            max_size: Maximum position size
            kelly_fraction: Fraction of Kelly criterion to use
            volatility_target: Target volatility for volatility-based sizing
            risk_target: Target risk per trade
        """
        self.strategy = strategy
        self.percent_size = percent_size
        self.fixed_size = fixed_size
        self.max_size = max_size
        self.kelly_fraction = kelly_fraction
        self.volatility_target = volatility_target
        self.risk_target = risk_target

    def calculate_position_size(
        self,
        portfolio: Portfolio,
        exchange: str,
        symbol: str,
        price: float,
        market_data: pd.DataFrame,
        win_rate: float = 0.5,
        profit_loss_ratio: float = 1.5,
    ) -> float:
        """Calculate position size based on strategy.

        Args:
            portfolio: Portfolio
            exchange: Exchange identifier
            symbol: Trading pair symbol
            price: Current price
            market_data: Market data for calculations
            win_rate: Historical win rate (for Kelly criterion)
            profit_loss_ratio: Profit/loss ratio (for Kelly criterion)

        Returns:
            float: Position size in base currency
        """
        equity = portfolio.calculate_equity()

        if self.strategy == PositionSizing.FIXED:
            size = self.fixed_size
        elif self.strategy == PositionSizing.PERCENT:
            size = equity * self.percent_size
        elif self.strategy == PositionSizing.KELLY:
            # Kelly criterion: f* = (bp - q) / b
            # where p = win rate, q = 1-p, b = profit/loss ratio
            p = win_rate
            q = 1 - p
            b = profit_loss_ratio
            kelly = (b * p - q) / b
            # Apply fraction to avoid full Kelly (too aggressive)
            kelly = max(0, kelly * self.kelly_fraction)
            size = equity * kelly
        elif self.strategy == PositionSizing.VOLATILITY:
            # Volatility-based position sizing
            if len(market_data) >= 20:
                # Calculate historical volatility
                returns = market_data["close"].pct_change().dropna()
                volatility = returns.std()
                # Size inversely proportional to volatility
                if volatility > 0:
                    size = equity * self.volatility_target / volatility
                else:
                    size = equity * self.percent_size
            else:
                # Not enough data, fall back to percent sizing
                size = equity * self.percent_size
        elif self.strategy == PositionSizing.RISK_PARITY:
            # Risk parity allocation
            # For simplicity, we'll use a fixed risk target
            size = equity * self.risk_target
        else:
            # Default to percent sizing
            size = equity * self.percent_size

        # Convert to position size in base currency
        position_size = size / price

        # Apply maximum size constraint if specified
        if self.max_size is not None and position_size * price > self.max_size:
            position_size = self.max_size / price

        return position_size


class PortfolioRebalancer:
    """Portfolio rebalancing logic with configurable frequencies."""

    def __init__(
        self,
        frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        threshold: float = 0.05,  # 5% threshold for threshold-based rebalancing
        target_weights: Optional[Dict[str, float]] = None,  # Target portfolio weights
    ):
        """Initialize portfolio rebalancer.

        Args:
            frequency: Rebalancing frequency
            threshold: Threshold for threshold-based rebalancing
            target_weights: Target portfolio weights (exchange:symbol -> weight)
        """
        self.frequency = frequency
        self.threshold = threshold
        self.target_weights = target_weights or {}
        self.last_rebalance_time: Optional[datetime] = None

    def should_rebalance(self, portfolio: Portfolio, current_time: datetime) -> bool:
        """Check if portfolio should be rebalanced.

        Args:
            portfolio: Portfolio
            current_time: Current time

        Returns:
            bool: True if rebalance is needed, False otherwise
        """
        if not self.last_rebalance_time:
            # First rebalance
            return True

        if self.frequency == RebalanceFrequency.THRESHOLD:
            # Check if any position weight deviates from target by more than threshold
            current_weights = self._calculate_current_weights(portfolio)
            for key, target_weight in self.target_weights.items():
                current_weight = current_weights.get(key, 0.0)
                if abs(current_weight - target_weight) > self.threshold:
                    return True
            return False
        elif self.frequency == RebalanceFrequency.DAILY:
            return (current_time - self.last_rebalance_time).days >= 1
        elif self.frequency == RebalanceFrequency.WEEKLY:
            return (current_time - self.last_rebalance_time).days >= 7
        elif self.frequency == RebalanceFrequency.MONTHLY:
            # Approximate month as 30 days
            return (current_time - self.last_rebalance_time).days >= 30
        elif self.frequency == RebalanceFrequency.QUARTERLY:
            # Approximate quarter as 90 days
            return (current_time - self.last_rebalance_time).days >= 90
        else:
            return False

    def _calculate_current_weights(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate current portfolio weights.

        Args:
            portfolio: Portfolio

        Returns:
            Dict[str, float]: Current weights (exchange:symbol -> weight)
        """
        total_equity = portfolio.calculate_equity()
        if total_equity == 0:
            return {}

        weights = {}
        for key, position in portfolio.positions.items():
            position_value = position.amount * position.current_price
            weights[key] = position_value / total_equity

        return weights

    def generate_rebalance_orders(
        self, portfolio: Portfolio, current_time: datetime
    ) -> List[Order]:
        """Generate orders to rebalance portfolio.

        Args:
            portfolio: Portfolio
            current_time: Current time

        Returns:
            List[Order]: Rebalance orders
        """
        if not self.target_weights:
            return []

        orders = []
        total_equity = portfolio.calculate_equity()
        current_weights = self._calculate_current_weights(portfolio)

        for key, target_weight in self.target_weights.items():
            current_weight = current_weights.get(key, 0.0)
            if (
                abs(current_weight - target_weight) > 0.001
            ):  # Small threshold to avoid unnecessary trades
                # Parse exchange and symbol from key
                exchange, symbol = key.split(":")

                # Get current position
                position = portfolio.get_position(exchange, symbol)

                # Calculate target position value
                target_value = total_equity * target_weight

                if position:
                    # Position exists, adjust it
                    current_value = position.amount * position.current_price
                    value_diff = target_value - current_value

                    if abs(value_diff) > 0.01:  # Avoid tiny adjustments
                        # Calculate amount to buy or sell
                        amount = abs(value_diff / position.current_price)

                        if value_diff > 0:
                            # Need to buy more
                            orders.append(
                                Order(
                                    exchange=exchange,
                                    symbol=symbol,
                                    order_type=OrderType.MARKET,
                                    side=OrderSide.BUY,
                                    amount=amount,
                                    timestamp=current_time,
                                )
                            )
                        else:
                            # Need to sell some
                            orders.append(
                                Order(
                                    exchange=exchange,
                                    symbol=symbol,
                                    order_type=OrderType.MARKET,
                                    side=OrderSide.SELL,
                                    amount=amount,
                                    timestamp=current_time,
                                )
                            )
                else:
                    # Position doesn't exist, create it if target weight > 0
                    if target_weight > 0:
                        # Need to get current price
                        # In a real implementation, this would come from market data
                        # For now, we'll assume a placeholder price
                        price = 1.0  # Placeholder

                        # Calculate amount to buy
                        amount = target_value / price

                        orders.append(
                            Order(
                                exchange=exchange,
                                symbol=symbol,
                                order_type=OrderType.MARKET,
                                side=OrderSide.BUY,
                                amount=amount,
                                timestamp=current_time,
                            )
                        )

        # Update last rebalance time
        self.last_rebalance_time = current_time

        return orders


class BacktestEngine:
    """Core backtesting engine for USDC arbitrage strategies."""

    def __init__(
        self,
        db_connection_string: str,
        initial_balance: float = 10000.0,
        fee_models: Optional[Dict[str, ExchangeFeeModel]] = None,
        slippage_model: Optional[SlippageModel] = None,
        position_sizer: Optional[PositionSizer] = None,
        rebalancer: Optional[PortfolioRebalancer] = None,
    ):
        """Initialize backtesting engine.

        Args:
            db_connection_string: Database connection string
            initial_balance: Initial portfolio balance
            fee_models: Exchange fee models
            slippage_model: Slippage model
            position_sizer: Position sizer
            rebalancer: Portfolio rebalancer
        """
        self.db = DBConnector(db_connection_string)
        self.portfolio = Portfolio(initial_balance=initial_balance)

        # Initialize models with defaults if not provided
        self.fee_models = fee_models or {
            "coinbase": ExchangeFeeModel("coinbase", maker_fee=0.005, taker_fee=0.005),
            "kraken": ExchangeFeeModel("kraken", maker_fee=0.0016, taker_fee=0.0026),
            "binance": ExchangeFeeModel("binance", maker_fee=0.001, taker_fee=0.001),
            "bitfinex": ExchangeFeeModel("bitfinex", maker_fee=0.001, taker_fee=0.002),
            "bitstamp": ExchangeFeeModel("bitstamp", maker_fee=0.002, taker_fee=0.005),
        }

        self.slippage_model = slippage_model or SlippageModel()
        self.position_sizer = position_sizer or PositionSizer()
        self.rebalancer = rebalancer or PortfolioRebalancer()

        # Market data cache
        self.market_data_cache: Dict[str, pd.DataFrame] = {}

    def get_market_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Get market data for backtesting.

        Args:
            exchange: Exchange identifier
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h', '1d')
            start_date: Start date
            end_date: End date

        Returns:
            pd.DataFrame: Market data
        """
        cache_key = f"{exchange}:{symbol}:{timeframe}:{start_date.isoformat()}:{end_date.isoformat()}"

        if cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]

        try:
            data = self.db.get_ohlcv_data_range(
                exchange, symbol, timeframe, start_date, end_date
            )
            self.market_data_cache[cache_key] = data
            return data
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return pd.DataFrame()

    def execute_order(
        self, order: Order, market_data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Execute order in backtest.

        Args:
            order: Order to execute
            market_data: Market data for execution

        Returns:
            Tuple[bool, str]: Success flag and message
        """
        if not market_data.empty:
            # Find the closest data point to the order timestamp
            market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
            closest_idx = (market_data["timestamp"] - order.timestamp).abs().idxmin()
            closest_data = market_data.iloc[closest_idx]

            # Get execution price based on order type
            if order.order_type == OrderType.MARKET:
                # Market orders execute at current price with slippage
                base_price = closest_data["close"]
                executed_price = self.slippage_model.calculate_slippage(
                    order.side, order.amount, base_price, market_data, order.timestamp
                )
                slippage = abs(executed_price - base_price) / base_price
            elif order.order_type == OrderType.LIMIT:
                # Limit orders execute at limit price if reached
                if order.price is None:
                    return False, "Limit order requires price"

                if order.side == OrderSide.BUY and closest_data["low"] <= order.price:
                    executed_price = order.price
                    slippage = 0.0
                elif (
                    order.side == OrderSide.SELL and closest_data["high"] >= order.price
                ):
                    executed_price = order.price
                    slippage = 0.0
                else:
                    return False, "Limit price not reached"
            else:
                # Other order types not implemented yet
                return False, f"Order type {order.order_type} not implemented"

            # Calculate fee
            fee_model = self.fee_models.get(
                order.exchange,
                ExchangeFeeModel(order.exchange),  # Default fee model
            )
            fee = fee_model.calculate_fee(
                order.order_type, order.side, order.amount, executed_price
            )

            # Update order with execution details
            order.executed_price = executed_price
            order.executed_amount = order.amount
            order.fee = fee
            order.status = "filled"
            order.execution_time = order.timestamp
            order.slippage = slippage

            # Update portfolio
            position_key = self.portfolio.get_position_key(order.exchange, order.symbol)
            position = self.portfolio.positions.get(position_key)

            if order.side == OrderSide.BUY:
                # Buy order
                order_cost = order.amount * executed_price + fee

                if order_cost > self.portfolio.cash:
                    return False, "Insufficient funds"

                self.portfolio.cash -= order_cost

                if position:
                    # Update existing position
                    new_amount = position.amount + order.amount
                    new_cost = (
                        position.amount * position.entry_price
                        + order.amount * executed_price
                    )
                    position.amount = new_amount
                    position.entry_price = new_cost / new_amount
                else:
                    # Create new position
                    new_position = Position(
                        exchange=order.exchange,
                        symbol=order.symbol,
                        amount=order.amount,
                        entry_price=executed_price,
                        entry_time=order.timestamp,
                    )
                    self.portfolio.add_position(new_position)
            else:
                # Sell order
                if not position or position.amount < order.amount:
                    return False, "Insufficient position"

                # Close position partially or fully
                realized_pnl = position.close(executed_price, order.amount, fee)

                # Update cash
                self.portfolio.cash += order.amount * executed_price - fee

                # If position is closed completely, move to closed positions
                if position.amount == 0:
                    position_dict = position.to_dict()
                    position_dict["exit_price"] = executed_price
                    position_dict["exit_time"] = order.timestamp.isoformat()
                    position_dict["realized_pnl"] = realized_pnl
                    self.portfolio.closed_positions.append(position_dict)
                    del self.portfolio.positions[position_key]

            # Record transaction
            self.portfolio.add_transaction(
                exchange=order.exchange,
                symbol=order.symbol,
                side=order.side,
                amount=order.amount,
                price=executed_price,
                fee=fee,
                timestamp=order.timestamp,
            )

            # Add order to portfolio history
            self.portfolio.orders.append(order)

            return True, "Order executed successfully"
        else:
            return False, "No market data available for execution"

    def run_backtest(
        self,
        strategy_func: callable,
        exchanges: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run backtest for a strategy.

        Args:
            strategy_func: Strategy function that generates signals
            exchanges: List of exchanges to include
            symbols: List of symbols to include
            timeframe: Timeframe for backtesting
            start_date: Start date
            end_date: End date
            strategy_params: Strategy parameters

        Returns:
            Dict[str, Any]: Backtest results
        """
        logger.info(
            f"Starting backtest from {start_date} to {end_date} "
            f"for {len(exchanges)} exchanges and {len(symbols)} symbols"
        )

        # Initialize portfolio
        self.portfolio = Portfolio(initial_balance=self.portfolio.initial_balance)

        # Get market data for all exchange/symbol combinations
        all_market_data = {}
        for exchange in exchanges:
            for symbol in symbols:
                data = self.get_market_data(
                    exchange, symbol, timeframe, start_date, end_date
                )
                if not data.empty:
                    all_market_data[f"{exchange}:{symbol}"] = data

        if not all_market_data:
            logger.error("No market data available for backtest")
            return {"error": "No market data available"}

        # Combine all timestamps from all data sources
        all_timestamps = set()
        for data in all_market_data.values():
            all_timestamps.update(data["timestamp"].tolist())

        # Sort timestamps chronologically
        sorted_timestamps = sorted(all_timestamps)

        # Run backtest for each timestamp
        for timestamp in sorted_timestamps:
            # Prepare current market data snapshot
            current_data = {}
            for key, data in all_market_data.items():
                exchange, symbol = key.split(":")
                # Find closest data point to current timestamp
                closest_idx = (data["timestamp"] - timestamp).abs().idxmin()
                current_data[key] = data.iloc[closest_idx].to_dict()

                # Update portfolio positions with current prices
                position = self.portfolio.get_position(exchange, symbol)
                if position:
                    position.update_price(current_data[key]["close"])

            # Check if rebalancing is needed
            if self.rebalancer and self.rebalancer.should_rebalance(
                self.portfolio, timestamp
            ):
                rebalance_orders = self.rebalancer.generate_rebalance_orders(
                    self.portfolio, timestamp
                )

                # Execute rebalance orders
                for order in rebalance_orders:
                    market_data = all_market_data.get(
                        f"{order.exchange}:{order.symbol}", pd.DataFrame()
                    )
                    self.execute_order(order, market_data)

            # Call strategy function to generate signals
            signals = strategy_func(
                timestamp=timestamp,
                portfolio=self.portfolio,
                market_data=current_data,
                params=strategy_params or {},
            )

            # Process signals
            for signal in signals:
                exchange = signal.get("exchange")
                symbol = signal.get("symbol")
                side = signal.get("side")

                if not exchange or not symbol or not side:
                    continue

                # Convert side string to enum
                if isinstance(side, str):
                    side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

                # Calculate position size
                market_data = all_market_data.get(
                    f"{exchange}:{symbol}", pd.DataFrame()
                )
                if market_data.empty:
                    continue

                # Find closest data point to current timestamp
                closest_idx = (market_data["timestamp"] - timestamp).abs().idxmin()
                price = market_data.iloc[closest_idx]["close"]

                # Use position sizer to determine amount
                amount = signal.get("amount")
                if amount is None:
                    amount = self.position_sizer.calculate_position_size(
                        portfolio=self.portfolio,
                        exchange=exchange,
                        symbol=symbol,
                        price=price,
                        market_data=market_data,
                    )

                # Create order
                order = Order(
                    exchange=exchange,
                    symbol=symbol,
                    order_type=OrderType.MARKET,  # Default to market orders
                    side=side,
                    amount=amount,
                    timestamp=timestamp,
                )

                # Execute order
                success, message = self.execute_order(order, market_data)
                if not success:
                    logger.warning(f"Order execution failed: {message}")

            # Update portfolio equity curve
            equity = self.portfolio.calculate_equity()
            self.portfolio.equity_curve.append((timestamp, equity))

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()

        # Prepare results
        results = {
            "portfolio": self.portfolio.to_dict(),
            "metrics": metrics,
            "parameters": strategy_params or {},
        }

        logger.info(
            f"Backtest completed with final equity: {self.portfolio.calculate_equity():.2f}"
        )

        return results

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for the backtest.

        Returns:
            Dict[str, Any]: Performance metrics
        """
        if not self.portfolio.equity_curve:
            return {"error": "No equity data available"}

        # Extract equity values and timestamps
        timestamps = [ts for ts, _ in self.portfolio.equity_curve]
        equity_values = [eq for _, eq in self.portfolio.equity_curve]

        # Convert to pandas Series for calculations
        equity_series = pd.Series(equity_values, index=timestamps)

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        # Calculate basic metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1

        # Calculate drawdown
        drawdown_series = equity_series / equity_series.cummax() - 1
        max_drawdown = drawdown_series.min()

        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0

        # Calculate Sortino ratio (downside risk only)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = float("inf")  # No negative returns

        # Calculate CAGR (Compound Annual Growth Rate)
        if len(timestamps) > 1:
            years = (timestamps[-1] - timestamps[0]).days / 365.25
            if years > 0:
                cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (
                    1 / years
                ) - 1
            else:
                cagr = 0
        else:
            cagr = 0

        # Calculate win rate
        if self.portfolio.transactions:
            # Group transactions by symbol and calculate PnL
            trades = []
            for i in range(len(self.portfolio.transactions) - 1):
                curr_tx = self.portfolio.transactions[i]
                next_tx = self.portfolio.transactions[i + 1]

                # Check if this is a round-trip trade
                if (
                    curr_tx["exchange"] == next_tx["exchange"]
                    and curr_tx["symbol"] == next_tx["symbol"]
                    and curr_tx["side"] != next_tx["side"]
                ):
                    # Calculate PnL
                    if curr_tx["side"] == "buy" and next_tx["side"] == "sell":
                        entry_price = curr_tx["price"]
                        exit_price = next_tx["price"]
                        amount = min(curr_tx["amount"], next_tx["amount"])
                        pnl = (
                            (exit_price - entry_price) * amount
                            - curr_tx["fee"]
                            - next_tx["fee"]
                        )
                    else:
                        entry_price = curr_tx["price"]
                        exit_price = next_tx["price"]
                        amount = min(curr_tx["amount"], next_tx["amount"])
                        pnl = (
                            (entry_price - exit_price) * amount
                            - curr_tx["fee"]
                            - next_tx["fee"]
                        )

                    trades.append({"pnl": pnl})

            if trades:
                win_count = sum(1 for trade in trades if trade["pnl"] > 0)
                win_rate = win_count / len(trades)
            else:
                win_rate = 0
        else:
            win_rate = 0

        # Convert transactions to trades DataFrame
        trades_df = self._convert_transactions_to_trades()

        # Use the performance metrics calculator
        from .metrics import PerformanceMetrics

        metrics_calculator = PerformanceMetrics(risk_free_rate=0.0)

        # Calculate comprehensive metrics
        metrics = metrics_calculator.calculate_metrics(
            equity_curve=equity_series,
            trades=trades_df,
        )

        # Extract key metrics for backward compatibility
        key_metrics = {
            "total_return": metrics["total_return"],
            "max_drawdown": metrics["drawdown"]["max_drawdown"],
            "sharpe_ratio": metrics["returns"]["sharpe_ratio"],
            "sortino_ratio": metrics["returns"]["sortino_ratio"],
            "calmar_ratio": metrics["returns"]["calmar_ratio"],
            "cagr": metrics["returns"]["cagr"],
            "win_rate": metrics["trades"].get("win_rate", 0),
            "final_equity": equity_series.iloc[-1],
            "initial_equity": equity_series.iloc[0],
            "trade_count": metrics["trades"].get("total_trades", 0),
            "annual_volatility": metrics["risk"]["annualized_volatility"],
            "var_95": metrics["risk"]["var_95"],
            "cvar_95": metrics["risk"]["cvar_95"],
            "max_drawdown_duration": metrics["drawdown"]["drawdown_duration"],
        }

        # Add detailed metrics
        key_metrics["detailed"] = metrics

        return key_metrics

    def _convert_transactions_to_trades(self) -> pd.DataFrame:
        """Convert transactions to trades DataFrame.

        Returns:
            pd.DataFrame: Trades DataFrame
        """
        if not self.portfolio.transactions:
            return pd.DataFrame()

        # Group transactions by symbol and calculate PnL
        trades = []
        for i in range(len(self.portfolio.transactions) - 1):
            curr_tx = self.portfolio.transactions[i]
            next_tx = self.portfolio.transactions[i + 1]

            # Check if this is a round-trip trade
            if (
                curr_tx["exchange"] == next_tx["exchange"]
                and curr_tx["symbol"] == next_tx["symbol"]
                and curr_tx["side"] != next_tx["side"]
            ):
                # Calculate PnL
                if curr_tx["side"] == "buy" and next_tx["side"] == "sell":
                    entry_price = curr_tx["price"]
                    exit_price = next_tx["price"]
                    amount = min(curr_tx["amount"], next_tx["amount"])
                    pnl = (
                        (exit_price - entry_price) * amount
                        - curr_tx["fee"]
                        - next_tx["fee"]
                    )

                    trades.append({
                        "exchange": curr_tx["exchange"],
                        "symbol": curr_tx["symbol"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "amount": amount,
                        "entry_time": datetime.fromisoformat(curr_tx["timestamp"]),
                        "exit_time": datetime.fromisoformat(next_tx["timestamp"]),
                        "pnl": pnl,
                        "entry_fee": curr_tx["fee"],
                        "exit_fee": next_tx["fee"],
                        "direction": "long",
                    })
                else:
                    entry_price = curr_tx["price"]
                    exit_price = next_tx["price"]
                    amount = min(curr_tx["amount"], next_tx["amount"])
                    pnl = (
                        (entry_price - exit_price) * amount
                        - curr_tx["fee"]
                        - next_tx["fee"]
                    )

                    trades.append({
                        "exchange": curr_tx["exchange"],
                        "symbol": curr_tx["symbol"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "amount": amount,
                        "entry_time": datetime.fromisoformat(curr_tx["timestamp"]),
                        "exit_time": datetime.fromisoformat(next_tx["timestamp"]),
                        "pnl": pnl,
                        "entry_fee": curr_tx["fee"],
                        "exit_fee": next_tx["fee"],
                        "direction": "short",
                    })

        return pd.DataFrame(trades)
