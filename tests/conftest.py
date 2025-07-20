"""
Test fixtures for the USDC arbitrage backtesting application.

This module contains pytest fixtures that can be used across multiple test files.
"""


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "performance: mark test as a performance test")


import os
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy.orm import Session

# Add src directory to Python path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    return MagicMock(spec=Session)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=24, freq="1H"),
        "open": [1.0] * 24,
        "high": [1.1] * 24,
        "low": [0.9] * 24,
        "close": [1.0 + 0.001 * i for i in range(24)],
        "volume": [100.0 + 10.0 * i for i in range(24)],
    })


@pytest.fixture
def sample_multi_exchange_data():
    """Create sample multi-exchange OHLCV data for testing."""
    timestamps = pd.date_range(start="2023-01-01", periods=24, freq="1H")
    data = []

    # Coinbase data
    for i, ts in enumerate(timestamps):
        data.append({
            "exchange": "coinbase",
            "timestamp": ts,
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0 + 0.001 * i,
            "volume": 100.0 + 10.0 * i,
        })

    # Kraken data (slightly different prices)
    for i, ts in enumerate(timestamps):
        data.append({
            "exchange": "kraken",
            "timestamp": ts,
            "open": 1.001,
            "high": 1.101,
            "low": 0.901,
            "close": 1.001 + 0.001 * i,
            "volume": 90.0 + 10.0 * i,
        })

    # Binance data (slightly different prices)
    for i, ts in enumerate(timestamps):
        data.append({
            "exchange": "binance",
            "timestamp": ts,
            "open": 0.999,
            "high": 1.099,
            "low": 0.899,
            "close": 0.999 + 0.001 * i,
            "volume": 110.0 + 10.0 * i,
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_backtest_result():
    """Create sample backtest result data for testing."""
    return {
        "strategy_name": "Test Arbitrage Strategy",
        "strategy_id": 1,
        "start_date": "2023-01-01T00:00:00Z",
        "end_date": "2023-01-31T23:59:59Z",
        "initial_capital": 10000.0,
        "final_value": 11500.0,
        "metrics": {
            "total_return": 0.15,
            "annualized_return": 0.45,
            "sharpe_ratio": 1.8,
            "sortino_ratio": 2.2,
            "max_drawdown": 0.05,
            "max_drawdown_duration": 3,
            "cagr": 0.45,
            "volatility": 0.08,
            "win_rate": 0.65,
            "profit_factor": 2.1,
            "avg_trade": 0.02,
        },
        "equity_curve": [
            {"timestamp": "2023-01-01T00:00:00Z", "equity": 10000.0},
            {"timestamp": "2023-01-02T00:00:00Z", "equity": 10050.0},
            # ... more data points
            {"timestamp": "2023-01-31T23:59:59Z", "equity": 11500.0},
        ],
        "trades": [
            {
                "timestamp": "2023-01-01T02:00:00Z",
                "exchange": "coinbase",
                "symbol": "USDC/USD",
                "side": "buy",
                "amount": 1000.0,
                "price": 0.998,
                "fee": 1.0,
            },
            {
                "timestamp": "2023-01-01T03:00:00Z",
                "exchange": "kraken",
                "symbol": "USDC/USD",
                "side": "sell",
                "amount": 1000.0,
                "price": 1.002,
                "fee": 1.0,
            },
            # ... more trades
        ],
        "drawdowns": [
            {
                "start": "2023-01-05T00:00:00Z",
                "end": "2023-01-08T00:00:00Z",
                "depth": 0.05,
                "duration": 3,
            },
            # ... more drawdowns
        ],
        "monthly_returns": [
            {"month": "2023-01", "return": 0.15},
        ],
    }


@pytest.fixture
def sample_arbitrage_opportunities():
    """Create sample arbitrage opportunities for testing."""
    return [
        {
            "timestamp": datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
            "buy_exchange": "coinbase",
            "sell_exchange": "kraken",
            "buy_price": 0.998,
            "sell_price": 1.002,
            "diff": 0.004,
            "pct_diff": 0.4,
            "profit_potential": 0.004,
        },
        {
            "timestamp": datetime(2023, 1, 1, 2, 0, tzinfo=UTC),
            "buy_exchange": "binance",
            "sell_exchange": "kraken",
            "buy_price": 0.997,
            "sell_price": 1.003,
            "diff": 0.006,
            "pct_diff": 0.6,
            "profit_potential": 0.006,
        },
        {
            "timestamp": datetime(2023, 1, 1, 3, 0, tzinfo=UTC),
            "buy_exchange": "coinbase",
            "sell_exchange": "binance",
            "buy_price": 0.996,
            "sell_price": 1.004,
            "diff": 0.008,
            "pct_diff": 0.8,
            "profit_potential": 0.008,
        },
    ]
