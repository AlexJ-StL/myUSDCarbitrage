"""Tests for strategy management system."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.api import models
from src.api.strategies.manager import StrategyManager
from src.api.strategies.comparison import StrategyComparison
from src.api.strategies.types import ArbitrageStrategy, create_strategy


class TestStrategyManager:
    """Test strategy manager functionality."""

    def test_create_strategy(self, db_session):
        """Test creating a new strategy."""
        manager = StrategyManager(db_session)

        strategy = manager.create_strategy(
            name="Test Strategy",
            description="A test strategy",
            code="def strategy(timestamp, portfolio, market_data, params): return []",
            parameters={"param1": 1.0},
            author="test_user",
            tags=["test", "arbitrage"],
        )

        assert strategy.name == "Test Strategy"
        assert strategy.description == "A test strategy"
        assert strategy.version == 1
        assert strategy.parameters == {"param1": 1.0}
        assert strategy.is_active is True

        # Check that version was created
        versions = manager.get_strategy_versions(strategy.id)
        assert len(versions) == 1
        assert versions[0].version == 1
        assert versions[0].created_by == "test_user"

    def test_create_duplicate_strategy_fails(self, db_session):
        """Test that creating a strategy with duplicate name fails."""
        manager = StrategyManager(db_session)

        # Create first strategy
        manager.create_strategy(
            name="Duplicate Strategy",
            description="First strategy",
            code="def strategy(): pass",
            parameters={},
            author="test_user",
        )

        # Try to create second strategy with same name
        with pytest.raises(
            ValueError, match="Strategy with name 'Duplicate Strategy' already exists"
        ):
            manager.create_strategy(
                name="Duplicate Strategy",
                description="Second strategy",
                code="def strategy(): pass",
                parameters={},
                author="test_user",
            )

    def test_update_strategy(self, db_session):
        """Test updating a strategy."""
        manager = StrategyManager(db_session)

        # Create strategy
        strategy = manager.create_strategy(
            name="Original Strategy",
            description="Original description",
            code="def strategy(): pass",
            parameters={"param1": 1.0},
            author="test_user",
        )

        # Update strategy
        updated_strategy = manager.update_strategy(
            strategy_id=strategy.id,
            name="Updated Strategy",
            description="Updated description",
            parameters={"param1": 2.0, "param2": "new"},
            author="test_user",
            commit_message="Updated parameters",
        )

        assert updated_strategy.name == "Updated Strategy"
        assert updated_strategy.description == "Updated description"
        assert updated_strategy.parameters == {"param1": 2.0, "param2": "new"}
        assert updated_strategy.version == 2

        # Check that new version was created
        versions = manager.get_strategy_versions(strategy.id)
        assert len(versions) == 2

    def test_get_strategy_versions(self, db_session):
        """Test getting strategy versions."""
        manager = StrategyManager(db_session)

        # Create strategy
        strategy = manager.create_strategy(
            name="Versioned Strategy",
            description="A strategy with versions",
            code="def strategy(): pass",
            parameters={"param1": 1.0},
            author="test_user",
        )

        # Update strategy multiple times
        for i in range(3):
            manager.update_strategy(
                strategy_id=strategy.id,
                parameters={"param1": float(i + 2)},
                author="test_user",
                commit_message=f"Update {i + 2}",
            )

        # Get versions
        versions = manager.get_strategy_versions(strategy.id)
        assert len(versions) == 4  # Initial + 3 updates

        # Check versions are ordered by version number (descending)
        version_numbers = [v.version for v in versions]
        assert version_numbers == [4, 3, 2, 1]

    def test_revert_to_version(self, db_session):
        """Test reverting strategy to previous version."""
        manager = StrategyManager(db_session)

        # Create strategy
        strategy = manager.create_strategy(
            name="Revert Strategy",
            description="Original description",
            code="def strategy(): pass",
            parameters={"param1": 1.0},
            author="test_user",
        )

        # Update strategy
        manager.update_strategy(
            strategy_id=strategy.id,
            description="Updated description",
            parameters={"param1": 2.0},
            author="test_user",
            commit_message="Update 1",
        )

        # Revert to version 1
        reverted_strategy = manager.revert_to_version(
            strategy_id=strategy.id,
            version=1,
            author="test_user",
        )

        assert reverted_strategy.description == "Original description"
        assert reverted_strategy.parameters == {"param1": 1.0}
        assert reverted_strategy.version == 3  # New version created

    def test_compare_versions(self, db_session):
        """Test comparing strategy versions."""
        manager = StrategyManager(db_session)

        # Create strategy
        strategy = manager.create_strategy(
            name="Compare Strategy",
            description="Original description",
            code="def strategy(): return 'v1'",
            parameters={"param1": 1.0},
            author="test_user",
        )

        # Update strategy
        manager.update_strategy(
            strategy_id=strategy.id,
            code="def strategy(): return 'v2'",
            parameters={"param1": 2.0, "param2": "new"},
            author="test_user",
            commit_message="Version 2",
        )

        # Compare versions
        comparison = manager.compare_versions(strategy.id, 1, 2)

        assert comparison["version1"]["version"] == 1
        assert comparison["version2"]["version"] == 2
        assert "code_diff" in comparison
        assert "parameter_diff" in comparison

        # Check parameter differences
        param_diff = comparison["parameter_diff"]
        assert param_diff["param1"]["status"] == "changed"
        assert param_diff["param1"]["old_value"] == 1.0
        assert param_diff["param1"]["new_value"] == 2.0
        assert param_diff["param2"]["status"] == "added"
        assert param_diff["param2"]["value"] == "new"

    def test_export_import_strategy(self, db_session):
        """Test exporting and importing strategies."""
        manager = StrategyManager(db_session)

        # Create strategy with tags
        strategy = manager.create_strategy(
            name="Export Strategy",
            description="A strategy to export",
            code="def strategy(): pass",
            parameters={"param1": 1.0},
            author="test_user",
            tags=["export", "test"],
        )

        # Export strategy
        exported_data = manager.export_strategy(strategy.id, include_history=True)

        assert exported_data["name"] == "Export Strategy"
        assert exported_data["description"] == "A strategy to export"
        assert exported_data["parameters"] == {"param1": 1.0}
        assert exported_data["tags"] == ["export", "test"]
        assert "history" in exported_data

        # Import strategy with new name
        exported_data["name"] = "Imported Strategy"
        imported_strategy = manager.import_strategy(exported_data, "import_user")

        assert imported_strategy.name == "Imported Strategy"
        assert imported_strategy.description == "A strategy to export"
        assert imported_strategy.parameters == {"param1": 1.0}

    def test_list_strategies_with_filters(self, db_session):
        """Test listing strategies with various filters."""
        manager = StrategyManager(db_session)

        # Create multiple strategies
        strategies = []
        for i in range(5):
            strategy = manager.create_strategy(
                name=f"Strategy {i}",
                description=f"Description {i}",
                code="def strategy(): pass",
                parameters={"param1": float(i)},
                author="test_user",
                tags=["test"] if i % 2 == 0 else ["other"],
                is_active=i < 3,  # First 3 are active
            )
            strategies.append(strategy)

        # Test listing all strategies
        all_strategies = manager.list_strategies()
        assert len(all_strategies) == 5

        # Test listing active only
        active_strategies = manager.list_strategies(active_only=True)
        assert len(active_strategies) == 3

        # Test filtering by tag
        test_strategies = manager.list_strategies(tag="test")
        assert len(test_strategies) == 3  # Strategies 0, 2, 4

        # Test search
        search_results = manager.list_strategies(search="Strategy 1")
        assert len(search_results) == 1
        assert search_results[0].name == "Strategy 1"


class TestStrategyTypes:
    """Test strategy type implementations."""

    def test_arbitrage_strategy_creation(self):
        """Test creating an arbitrage strategy."""
        parameters = {
            "min_spread": 0.002,
            "max_position": 500.0,
            "exchanges": ["coinbase", "kraken"],
        }

        strategy = create_strategy("arbitrage", parameters)
        assert isinstance(strategy, ArbitrageStrategy)
        assert strategy.name == "ArbitrageStrategy"
        assert strategy.parameters["min_spread"] == 0.002
        assert strategy.parameters["max_position"] == 500.0

    def test_arbitrage_strategy_signals(self):
        """Test arbitrage strategy signal generation."""
        parameters = {
            "min_spread": 0.001,
            "max_position": 1000.0,
            "exchanges": ["coinbase", "kraken"],
        }

        strategy = ArbitrageStrategy(parameters)

        # Mock portfolio and market data
        portfolio = Mock()
        portfolio.get_position.return_value = None
        portfolio.positions = []

        market_data = {
            "coinbase:USDC/USD": {"close": 1.0000},
            "kraken:USDC/USD": {"close": 1.0015},  # 0.15% spread
        }

        timestamp = datetime.now()
        signals = strategy.generate_signals(timestamp, portfolio, market_data)

        # Should generate buy signal for coinbase and sell signal for kraken
        assert len(signals) == 2

        buy_signal = next(s for s in signals if s["side"].name == "BUY")
        sell_signal = next(s for s in signals if s["side"].name == "SELL")

        assert buy_signal["exchange"] == "coinbase"
        assert sell_signal["exchange"] == "kraken"

    def test_invalid_strategy_type(self):
        """Test creating strategy with invalid type."""
        with pytest.raises(ValueError, match="Unsupported strategy type: invalid"):
            create_strategy("invalid", {})


class TestStrategyComparison:
    """Test strategy comparison functionality."""

    @patch("src.api.strategies.comparison.BacktestEngine")
    def test_compare_strategies(self, mock_backtest_engine, db_session):
        """Test comparing multiple strategies."""
        # Create mock strategies
        strategy1 = models.Strategy(
            id=1,
            name="Strategy 1",
            description="First strategy",
            code="def strategy(): pass",
            parameters={},
            version=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        strategy2 = models.Strategy(
            id=2,
            name="Strategy 2",
            description="Second strategy",
            code="def strategy(): pass",
            parameters={},
            version=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        db_session.add_all([strategy1, strategy2])
        db_session.commit()

        # Mock backtest results
        mock_backtest_engine.return_value.run_backtest.return_value = {
            "metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.05,
                "win_rate": 0.6,
                "profit_factor": 1.5,
                "total_trades": 100,
            },
            "portfolio_history": [
                {"timestamp": datetime.now(), "equity": 10000},
                {"timestamp": datetime.now(), "equity": 11500},
            ],
        }

        comparison = StrategyComparison(db_session, mock_backtest_engine.return_value)

        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        result = comparison.compare_strategies(
            strategy_ids=[1, 2],
            start_date=start_date,
            end_date=end_date,
        )

        assert "strategies" in result
        assert "summary" in result
        assert "rankings" in result
        assert len(result["strategies"]) == 2

    @patch("src.api.strategies.comparison.BacktestEngine")
    def test_ab_test(self, mock_backtest_engine, db_session):
        """Test A/B testing functionality."""
        # Create mock strategies
        strategy_a = models.Strategy(
            id=1,
            name="Strategy A",
            description="Strategy A",
            code="def strategy(): pass",
            parameters={},
            version=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        strategy_b = models.Strategy(
            id=2,
            name="Strategy B",
            description="Strategy B",
            code="def strategy(): pass",
            parameters={},
            version=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        db_session.add_all([strategy_a, strategy_b])
        db_session.commit()

        # Mock backtest results with different performance
        def mock_backtest_side_effect(*args, **kwargs):
            strategy_func = kwargs.get("strategy_func")
            if hasattr(strategy_func, "__name__") and "A" in str(strategy_func):
                return {
                    "metrics": {
                        "total_return": 0.20,
                        "sharpe_ratio": 1.5,
                        "max_drawdown": -0.03,
                        "win_rate": 0.65,
                        "profit_factor": 1.8,
                    },
                    "portfolio_history": [{"equity": 10000}, {"equity": 12000}],
                }
            else:
                return {
                    "metrics": {
                        "total_return": 0.10,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.08,
                        "win_rate": 0.55,
                        "profit_factor": 1.2,
                    },
                    "portfolio_history": [{"equity": 10000}, {"equity": 11000}],
                }

        mock_backtest_engine.return_value.run_backtest.side_effect = (
            mock_backtest_side_effect
        )

        comparison = StrategyComparison(db_session, mock_backtest_engine.return_value)

        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        result = comparison.run_ab_test(
            strategy_a_id=1,
            strategy_b_id=2,
            start_date=start_date,
            end_date=end_date,
        )

        assert "strategy_a" in result
        assert "strategy_b" in result
        assert "statistical_tests" in result
        assert "conclusion" in result

        # Strategy A should be the winner
        assert result["conclusion"]["winner"] == "strategy_a"


@pytest.fixture
def db_session():
    """Create a test database session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.api.database import Base

    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session

    session.close()
