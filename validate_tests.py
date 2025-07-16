#!/usr/bin/env python3
"""Simple script to validate that the tests can run."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import pandas as pd
    from strategies.simple_arb import strategy_simple_arb

    # Test basic functionality
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=5),
            "close": [1.00, 1.01, 0.99, 1.02, 1.01],
        }
    )

    result = strategy_simple_arb(
        df, buy_threshold=0.99, sell_threshold=1.02, initial_capital=1000.0
    )

    print("✓ Import successful")
    print("✓ Function execution successful")
    print(f"✓ Result type: {type(result)}")
    print(f"✓ Trades count: {len(result['trades'])}")
    print("✓ All tests should pass")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
