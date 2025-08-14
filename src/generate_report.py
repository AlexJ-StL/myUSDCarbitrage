#!/usr/bin/env python
"""
Script to generate reports for USDC arbitrage system.

This script provides a command-line interface for generating various reports,
including arbitrage opportunity reports and strategy performance reports.

Examples:
    # Generate arbitrage opportunity report
    python generate_report.py arbitrage --exchanges coinbase kraken --output report.html

    # Generate strategy performance report from file
    python generate_report.py strategy --backtest-result results.json --output report.html

    # Generate on-demand strategy performance report from database
    python generate_report.py on-demand-strategy --strategy-id 1 --output report.html

For more information, run:
    python generate_report.py --help
"""

from src.reporting.cli import main

if __name__ == "__main__":
    main()
