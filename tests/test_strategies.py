"""Tests for strategy functionality."""

import pandas as pd
import pytest

from strategies.simple_arb import strategy_simple_arb


def test_strategy_simple_arb_valid() -> None:
    """Test simple arbitrage strategy with valid data."""
    # Create test DataFrame
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=10),
            "close": [1.00, 1.01, 1.02, 0.99, 1.01, 1.05, 0.98, 0.97, 1.00, 1.02],
        }
    )
    result = strategy_simple_arb(
        df, buy_threshold=0.99, sell_threshold=1.03, initial_capital=1000.0
    )
    assert isinstance(result, dict)
    assert "trades" in result
    assert "portfolio" in result
    assert len(result["trades"]) > 0  # Expected trades


def test_strategy_simple_arb_no_opportunities() -> None:
    """Test simple arbitrage strategy with no trading opportunities."""
    # Create test DataFrame without opportunities
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=5),
            "close": [1.01, 1.02, 1.03, 1.02, 1.01],
        }
    )
    result = strategy_simple_arb(
        df, buy_threshold=1.00, sell_threshold=1.04, initial_capital=1000.0
    )
    assert len(result["trades"]) == 0  # No trades should occur


def test_simple_arb_strategy_invalid_data() -> None:
    """Test simple arbitrage strategy with invalid data type."""
    # Edge case: invalid data type
    with pytest.raises(AttributeError):
        # Pass a string instead of DataFrame
        strategy_simple_arb(
            "invalid", buy_threshold=0.99, sell_threshold=1.03, initial_capital=1000.0
        )


# Add more test cases when ready
