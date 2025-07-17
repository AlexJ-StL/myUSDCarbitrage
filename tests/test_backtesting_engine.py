"""Tests for the backtesting engine."""

import os
import sys
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add src directory to Python path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from api.backtesting import (
    BacktestEngine,
    ExchangeFeeModel,
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    PortfolioRebalancer,
    Position,
    PositionSizer,
    PositionSizing,
    RebalanceFrequency,
    SlippageModel,
)


class TestBacktestingComponents(unittest.TestCase):
    """Test individual components of the backtesting engine."""

    def test_order_creation(self):
        """Test order creation and to_dict method."""
        order = Order(
            exchange="coinbase",
            symbol="USDC/USD",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=100.0,
            price=1.0,
            timestamp=datetime(2023, 1, 1, tzinfo=UTC),
        )

        self.assertEqual(order.exchange, "coinbase")
        self.assertEqual(order.symbol, "USDC/USD")
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.amount, 100.0)
        self.assertEqual(order.price, 1.0)

        # Test to_dict method
        order_dict = order.to_dict()
        self.assertEqual(order_dict["exchange"], "coinbase")
        self.assertEqual(order_dict["symbol"], "USDC/USD")
        self.assertEqual(order_dict["order_type"], "market")
        self.assertEqual(order_dict["side"], "buy")
        self.assertEqual(order_dict["amount"], 100.0)
        self.assertEqual(order_dict["price"], 1.0)

    def test_position_creation_and_update(self):
        """Test position creation and update methods."""
        position = Position(
            exchange="coinbase",
            symbol="USDC/USD",
            amount=100.0,
            entry_price=1.0,
            entry_time=datetime(2023, 1, 1, tzinfo=UTC),
        )

        self.assertEqual(position.exchange, "coinbase")
        self.assertEqual(position.symbol, "USDC/USD")
        self.assertEqual(position.amount, 100.0)
        self.assertEqual(position.entry_price, 1.0)
        self.assertEqual(position.unrealized_pnl, 0.0)

        # Test update_price method
        position.update_price(1.1)
        self.assertEqual(position.current_price, 1.1)
        self.assertEqual(position.unrealized_pnl, 10.0)  # 100 * (1.1 - 1.0)

        # Test close method
        realized_pnl = position.close(1.2, 50.0, 1.0)
        self.assertEqual(position.amount, 50.0)
        self.assertEqual(realized_pnl, 9.0)  # 50 * (1.2 - 1.0) - 1.0
        self.assertEqual(position.realized_pnl, 9.0)
        self.assertEqual(position.fees_paid, 1.0)

    def test_portfolio_management(self):
        """Test portfolio management functionality."""
        portfolio = Portfolio(initial_balance=10000.0)

        # Test initial state
        self.assertEqual(portfolio.cash, 10000.0)
        self.assertEqual(len(portfolio.positions), 0)
        self.assertEqual(len(portfolio.equity_curve), 0)

        # Add a position
        position = Position(
            exchange="coinbase",
            symbol="USDC/USD",
            amount=100.0,
            entry_price=1.0,
            entry_time=datetime(2023, 1, 1, tzinfo=UTC),
        )
        portfolio.add_position(position)

        # Test position retrieval
        retrieved_position = portfolio.get_position("coinbase", "USDC/USD")
        self.assertEqual(retrieved_position, position)

        # Test position update
        portfolio.update_position(
            "coinbase", "USDC/USD", 1.1, datetime(2023, 1, 2, tzinfo=UTC)
        )
        self.assertEqual(retrieved_position.current_price, 1.1)
        self.assertEqual(retrieved_position.unrealized_pnl, 10.0)

        # Test equity calculation
        equity = portfolio.calculate_equity()
        self.assertEqual(equity, 10000.0 + 10.0)  # Cash + unrealized PnL

        # Test equity curve
        self.assertEqual(len(portfolio.equity_curve), 1)
        self.assertEqual(portfolio.equity_curve[0][1], 10010.0)

    def test_exchange_fee_model(self):
        """Test exchange fee model calculations."""
        # Test basic fee calculation
        fee_model = ExchangeFeeModel(
            exchange="coinbase",
            maker_fee=0.001,
            taker_fee=0.002,
        )

        # Test maker fee
        maker_fee = fee_model.calculate_fee(OrderType.LIMIT, OrderSide.BUY, 100.0, 1.0)
        self.assertEqual(maker_fee, 0.1)  # 100 * 1.0 * 0.001

        # Test taker fee
        taker_fee = fee_model.calculate_fee(OrderType.MARKET, OrderSide.BUY, 100.0, 1.0)
        self.assertEqual(taker_fee, 0.2)  # 100 * 1.0 * 0.002

        # Test tiered fees
        tiered_fee_model = ExchangeFeeModel(
            exchange="binance",
            maker_fee=0.001,
            taker_fee=0.002,
            tiered_fees=[
                (1000, 0.0009),  # >= $1000: 0.09%
                (10000, 0.0007),  # >= $10000: 0.07%
                (100000, 0.0005),  # >= $100000: 0.05%
            ],
        )

        # Test tier 1
        tier1_fee = tiered_fee_model.calculate_fee(
            OrderType.MARKET, OrderSide.BUY, 500.0, 1.0
        )
        self.assertEqual(tier1_fee, 1.0)  # 500 * 1.0 * 0.002

        # Test tier 2
        tier2_fee = tiered_fee_model.calculate_fee(
            OrderType.MARKET, OrderSide.BUY, 2000.0, 1.0
        )
        self.assertEqual(tier2_fee, 1.8)  # 2000 * 1.0 * 0.0009

        # Test tier 3
        tier3_fee = tiered_fee_model.calculate_fee(
            OrderType.MARKET, OrderSide.BUY, 20000.0, 1.0
        )
        self.assertEqual(tier3_fee, 14.0)  # 20000 * 1.0 * 0.0007

    def test_slippage_model(self):
        """Test slippage model calculations."""
        # Create a simple slippage model
        slippage_model = SlippageModel(
            base_slippage=0.001,
            volume_impact=0.5,
            volatility_impact=0.5,
            random_factor=0.0,  # Disable randomness for testing
        )

        # Create test market data
        market_data = pd.DataFrame({
            "timestamp": [
                datetime(2023, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(20)
            ],
            "open": [1.0] * 20,
            "high": [1.1] * 20,
            "low": [0.9] * 20,
            "close": [1.0 + 0.01 * i for i in range(20)],
            "volume": [100.0 + 10.0 * i for i in range(20)],
        })

        # Test buy slippage
        buy_price = slippage_model.calculate_slippage(
            OrderSide.BUY,
            50.0,
            1.0,
            market_data,
            datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
        )
        # Expected slippage: base_slippage * (1 + volume_impact * volume_ratio)
        # volume_ratio = 50 / 110 = 0.455
        # slippage_rate = 0.001 * (1 + 0.5 * 0.455) = 0.001 * 1.2275 = 0.0012275
        # buy_price = 1.0 * (1 + 0.0012275) = 1.0012275
        self.assertAlmostEqual(buy_price, 1.0012275, places=6)

        # Test sell slippage
        sell_price = slippage_model.calculate_slippage(
            OrderSide.SELL,
            50.0,
            1.0,
            market_data,
            datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
        )
        # Expected slippage: base_slippage * (1 + volume_impact * volume_ratio)
        # volume_ratio = 50 / 110 = 0.455
        # slippage_rate = 0.001 * (1 + 0.5 * 0.455) = 0.001 * 1.2275 = 0.0012275
        # sell_price = 1.0 * (1 - 0.0012275) = 0.9987725
        self.assertAlmostEqual(sell_price, 0.9987725, places=6)

    def test_position_sizer(self):
        """Test position sizing strategies."""
        # Create a portfolio
        portfolio = Portfolio(initial_balance=10000.0)

        # Create test market data
        market_data = pd.DataFrame({
            "timestamp": [
                datetime(2023, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(20)
            ],
            "open": [1.0] * 20,
            "high": [1.1] * 20,
            "low": [0.9] * 20,
            "close": [1.0 + 0.01 * i for i in range(20)],
            "volume": [100.0 + 10.0 * i for i in range(20)],
        })

        # Test fixed position sizing
        fixed_sizer = PositionSizer(
            strategy=PositionSizing.FIXED,
            fixed_size=1000.0,
        )
        fixed_size = fixed_sizer.calculate_position_size(
            portfolio, "coinbase", "USDC/USD", 1.0, market_data
        )
        self.assertEqual(fixed_size, 1000.0)

        # Test percent position sizing
        percent_sizer = PositionSizer(
            strategy=PositionSizing.PERCENT,
            percent_size=0.02,
        )
        percent_size = percent_sizer.calculate_position_size(
            portfolio, "coinbase", "USDC/USD", 1.0, market_data
        )
        self.assertEqual(percent_size, 10000.0 * 0.02)

        # Test Kelly criterion
        kelly_sizer = PositionSizer(
            strategy=PositionSizing.KELLY,
            kelly_fraction=0.5,
        )
        kelly_size = kelly_sizer.calculate_position_size(
            portfolio,
            "coinbase",
            "USDC/USD",
            1.0,
            market_data,
            win_rate=0.6,
            profit_loss_ratio=2.0,
        )
        # Kelly formula: f* = (bp - q) / b
        # p = 0.6, q = 0.4, b = 2.0
        # f* = (2.0 * 0.6 - 0.4) / 2.0 = 0.4
        # With fraction = 0.5: f = 0.4 * 0.5 = 0.2
        # Position size = 10000.0 * 0.2 = 2000.0
        self.assertEqual(kelly_size, 2000.0)

    def test_portfolio_rebalancer(self):
        """Test portfolio rebalancing logic."""
        # Create a portfolio
        portfolio = Portfolio(initial_balance=10000.0)

        # Add positions
        portfolio.add_position(
            Position(
                exchange="coinbase",
                symbol="USDC/USD",
                amount=1000.0,
                entry_price=1.0,
                entry_time=datetime(2023, 1, 1, tzinfo=UTC),
            )
        )
        portfolio.add_position(
            Position(
                exchange="kraken",
                symbol="USDC/USD",
                amount=2000.0,
                entry_price=1.0,
                entry_time=datetime(2023, 1, 1, tzinfo=UTC),
            )
        )

        # Update prices
        portfolio.update_position(
            "coinbase", "USDC/USD", 1.1, datetime(2023, 1, 2, tzinfo=UTC)
        )
        portfolio.update_position(
            "kraken", "USDC/USD", 0.9, datetime(2023, 1, 2, tzinfo=UTC)
        )

        # Create rebalancer with target weights
        rebalancer = PortfolioRebalancer(
            frequency=RebalanceFrequency.THRESHOLD,
            threshold=0.05,
            target_weights={
                "coinbase:USDC/USD": 0.4,
                "kraken:USDC/USD": 0.4,
            },
        )

        # Test should_rebalance
        self.assertTrue(
            rebalancer.should_rebalance(portfolio, datetime(2023, 1, 2, tzinfo=UTC))
        )

        # Test generate_rebalance_orders
        orders = rebalancer.generate_rebalance_orders(
            portfolio, datetime(2023, 1, 2, tzinfo=UTC)
        )

        # Calculate expected orders
        # Total equity = 10000 (cash) + 1000 * 1.1 (coinbase) + 2000 * 0.9 (kraken) = 12900
        # Target coinbase value = 12900 * 0.4 = 5160
        # Current coinbase value = 1000 * 1.1 = 1100
        # Need to buy 5160 - 1100 = 4060 worth of coinbase
        # Target kraken value = 12900 * 0.4 = 5160
        # Current kraken value = 2000 * 0.9 = 1800
        # Need to buy 5160 - 1800 = 3360 worth of kraken

        # Find coinbase order
        coinbase_order = next((o for o in orders if o.exchange == "coinbase"), None)
        self.assertIsNotNone(coinbase_order)
        self.assertEqual(coinbase_order.side, OrderSide.BUY)
        self.assertAlmostEqual(coinbase_order.amount, 4060 / 1.1, places=2)

        # Find kraken order
        kraken_order = next((o for o in orders if o.exchange == "kraken"), None)
        self.assertIsNotNone(kraken_order)
        self.assertEqual(kraken_order.side, OrderSide.BUY)
        self.assertAlmostEqual(kraken_order.amount, 3360 / 0.9, places=2)


@patch("api.backtesting.engine.DBConnector")
class TestBacktestEngine(unittest.TestCase):
    """Test the backtesting engine."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample market data
        self.market_data = pd.DataFrame({
            "timestamp": [
                datetime(2023, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(24)
            ],
            "open": [1.0] * 24,
            "high": [1.1] * 24,
            "low": [0.9] * 24,
            "close": [1.0 + 0.01 * (i % 12) for i in range(24)],
            "volume": [100.0 + 10.0 * i for i in range(24)],
        })

    def test_engine_initialization(self, mock_db):
        """Test backtesting engine initialization."""
        engine = BacktestEngine("dummy_connection")

        # Check that components are initialized
        self.assertIsNotNone(engine.portfolio)
        self.assertIsNotNone(engine.fee_models)
        self.assertIsNotNone(engine.slippage_model)
        self.assertIsNotNone(engine.position_sizer)
        self.assertIsNotNone(engine.rebalancer)

        # Check default values
        self.assertEqual(engine.portfolio.initial_balance, 10000.0)
        self.assertEqual(engine.portfolio.cash, 10000.0)

    def test_get_market_data(self, mock_db):
        """Test market data retrieval."""
        # Mock the database connector
        mock_db.return_value.get_ohlcv_data_range.return_value = self.market_data

        engine = BacktestEngine("dummy_connection")

        # Test data retrieval
        data = engine.get_market_data(
            "coinbase",
            "USDC/USD",
            "1h",
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
        )

        # Check that data was retrieved
        self.assertEqual(len(data), len(self.market_data))
        mock_db.return_value.get_ohlcv_data_range.assert_called_once()

        # Test cache
        engine.get_market_data(
            "coinbase",
            "USDC/USD",
            "1h",
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
        )
        # Should still be called only once (cached)
        mock_db.return_value.get_ohlcv_data_range.assert_called_once()

    def test_execute_order(self, mock_db):
        """Test order execution."""
        engine = BacktestEngine("dummy_connection")

        # Create an order
        order = Order(
            exchange="coinbase",
            symbol="USDC/USD",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=100.0,
            timestamp=datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
        )

        # Execute order
        success, message = engine.execute_order(order, self.market_data)

        # Check that order was executed
        self.assertTrue(success)
        self.assertEqual(message, "Order executed successfully")
        self.assertEqual(order.status, "filled")
        self.assertIsNotNone(order.executed_price)
        self.assertEqual(order.executed_amount, 100.0)

        # Check that position was created
        position = engine.portfolio.get_position("coinbase", "USDC/USD")
        self.assertIsNotNone(position)
        self.assertEqual(position.amount, 100.0)

        # Check that cash was reduced
        self.assertLess(engine.portfolio.cash, 10000.0)

        # Test sell order
        sell_order = Order(
            exchange="coinbase",
            symbol="USDC/USD",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            amount=50.0,
            timestamp=datetime(2023, 1, 1, 2, 0, tzinfo=UTC),
        )

        # Execute sell order
        success, message = engine.execute_order(sell_order, self.market_data)

        # Check that order was executed
        self.assertTrue(success)
        self.assertEqual(message, "Order executed successfully")
        self.assertEqual(sell_order.status, "filled")

        # Check that position was updated
        self.assertEqual(position.amount, 50.0)

        # Check that cash was increased
        self.assertGreater(
            engine.portfolio.cash, 10000.0 - 100.0 * order.executed_price
        )

    def test_run_backtest(self, mock_db):
        """Test running a backtest."""
        # Mock the database connector
        mock_db.return_value.get_ohlcv_data_range.return_value = self.market_data

        engine = BacktestEngine("dummy_connection")

        # Define a simple strategy function
        def simple_strategy(timestamp, portfolio, market_data, params):
            signals = []

            # Buy on even hours, sell on odd hours
            hour = timestamp.hour

            if hour % 2 == 0:
                signals.append({
                    "exchange": "coinbase",
                    "symbol": "USDC/USD",
                    "side": OrderSide.BUY,
                    "amount": 10.0,
                })
            else:
                # Only sell if we have a position
                position = portfolio.get_position("coinbase", "USDC/USD")
                if position and position.amount > 0:
                    signals.append({
                        "exchange": "coinbase",
                        "symbol": "USDC/USD",
                        "side": OrderSide.SELL,
                        "amount": min(10.0, position.amount),
                    })

            return signals

        # Run backtest
        results = engine.run_backtest(
            strategy_func=simple_strategy,
            exchanges=["coinbase"],
            symbols=["USDC/USD"],
            timeframe="1h",
            start_date=datetime(2023, 1, 1, tzinfo=UTC),
            end_date=datetime(2023, 1, 2, tzinfo=UTC),
            strategy_params={},
        )

        # Check that results were generated
        self.assertIn("portfolio", results)
        self.assertIn("metrics", results)

        # Check that trades were executed
        self.assertGreater(len(results["portfolio"]["orders"]), 0)

        # Check that metrics were calculated
        self.assertIn("total_return", results["metrics"])
        self.assertIn("sharpe_ratio", results["metrics"])
        self.assertIn("max_drawdown", results["metrics"])


if __name__ == "__main__":
    unittest.main()
