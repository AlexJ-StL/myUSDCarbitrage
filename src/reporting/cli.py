"""Command-line interface for generating reports."""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.api.database import DBConnector
from src.reporting.report_generator import (
    generate_arbitrage_report,
    generate_strategy_performance_report,
)
from src.reporting.on_demand_report_generator import (
    OnDemandReportGenerator,
    get_on_demand_report_generator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reports for USDC arbitrage system"
    )

    subparsers = parser.add_subparsers(
        dest="report_type", help="Type of report to generate"
    )

    # Arbitrage report parser
    arb_parser = subparsers.add_parser(
        "arbitrage", help="Generate arbitrage opportunity report"
    )
    arb_parser.add_argument(
        "--exchanges",
        nargs="+",
        required=True,
        help="List of exchanges to compare (e.g., coinbase kraken binance)",
    )
    arb_parser.add_argument(
        "--symbol",
        default="USDC/USD",
        help="Trading symbol to analyze (default: USDC/USD)",
    )
    arb_parser.add_argument(
        "--start-time",
        type=lambda s: datetime.fromisoformat(s),
        help="Start time for analysis period (ISO format, e.g., 2023-01-01T00:00:00)",
    )
    arb_parser.add_argument(
        "--end-time",
        type=lambda s: datetime.fromisoformat(s),
        help="End time for analysis period (ISO format, e.g., 2023-01-31T23:59:59)",
    )
    arb_parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Minimum percentage difference to be considered an opportunity (default: 0.001 for 0.1%%)",
    )
    arb_parser.add_argument(
        "--output",
        default="arbitrage_report.html",
        help="Output file path (default: arbitrage_report.html)",
    )
    arb_parser.add_argument(
        "--format",
        choices=["html", "json", "csv"],
        default="html",
        help="Output format (default: html)",
    )

    # Strategy performance report parser
    strat_parser = subparsers.add_parser(
        "strategy", help="Generate strategy performance report"
    )
    strat_parser.add_argument(
        "--backtest-result",
        help="Path to JSON file containing backtest results",
    )
    strat_parser.add_argument(
        "--benchmark-data", help="Path to JSON file containing benchmark data"
    )
    strat_parser.add_argument(
        "--sections",
        nargs="+",
        help="Sections to include in the report (default: all sections)",
    )
    strat_parser.add_argument(
        "--output",
        default="strategy_report.html",
        help="Output file path (default: strategy_report.html)",
    )
    strat_parser.add_argument(
        "--format",
        choices=["html", "json", "csv"],
        default="html",
        help="Output format (default: html)",
    )

    # On-demand strategy report parser (using database)
    on_demand_strat_parser = subparsers.add_parser(
        "on-demand-strategy",
        help="Generate on-demand strategy performance report from database",
    )
    on_demand_strat_parser.add_argument(
        "--strategy-id",
        type=int,
        help="ID of the strategy to analyze",
    )
    on_demand_strat_parser.add_argument(
        "--backtest-id",
        type=int,
        help="ID of the specific backtest to analyze",
    )
    on_demand_strat_parser.add_argument(
        "--start-date",
        type=lambda s: datetime.fromisoformat(s),
        help="Start date for analysis period (ISO format, e.g., 2023-01-01T00:00:00)",
    )
    on_demand_strat_parser.add_argument(
        "--end-date",
        type=lambda s: datetime.fromisoformat(s),
        help="End date for analysis period (ISO format, e.g., 2023-01-31T23:59:59)",
    )
    on_demand_strat_parser.add_argument(
        "--include-benchmark",
        action="store_true",
        help="Include benchmark comparison",
    )
    on_demand_strat_parser.add_argument(
        "--benchmark-symbol",
        default="BTC/USD",
        help="Symbol to use as benchmark (default: BTC/USD)",
    )
    on_demand_strat_parser.add_argument(
        "--sections",
        nargs="+",
        help="Sections to include in the report (default: all sections)",
    )
    on_demand_strat_parser.add_argument(
        "--output",
        default="strategy_report.html",
        help="Output file path (default: strategy_report.html)",
    )
    on_demand_strat_parser.add_argument(
        "--format",
        choices=["html", "json", "csv"],
        default="html",
        help="Output format (default: html)",
    )

    return parser.parse_args()


def get_db_session() -> Session:
    """Create a database session."""
    try:
        # Get database connection string from environment or use default
        connection_string = (
            "postgresql://arb_user:strongpassword@localhost:5432/usdc_arbitrage"
        )

        engine = create_engine(connection_string)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        return SessionLocal()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)


def generate_arbitrage_report_cli(args):
    """Generate arbitrage report from command-line arguments."""
    try:
        # Set default times if not provided
        end_time = args.end_time or datetime.now()
        start_time = args.start_time or (end_time - timedelta(days=1))

        if start_time >= end_time:
            logger.error("Start time must be before end time")
            sys.exit(1)

        # Get database session
        db = get_db_session()

        # Generate report
        logger.info(
            f"Generating arbitrage report for {args.symbol} across {', '.join(args.exchanges)}"
        )
        logger.info(f"Period: {start_time} to {end_time}")
        logger.info(f"Threshold: {args.threshold * 100}%")

        html_report = generate_arbitrage_report(
            db=db,
            start_time=start_time,
            end_time=end_time,
            exchanges=args.exchanges,
            symbol=args.symbol,
            threshold=args.threshold,
        )

        # Write report to file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html_report)

        logger.info(f"Report saved to {args.output}")

    except Exception as e:
        logger.error(f"Failed to generate arbitrage report: {e}")
        sys.exit(1)


def generate_strategy_report_cli(args):
    """Generate strategy performance report from command-line arguments."""
    try:
        # Load backtest result
        with open(args.backtest_result, "r") as f:
            backtest_result = json.load(f)

        # Load benchmark data if provided
        benchmark_data = None
        if args.benchmark_data:
            with open(args.benchmark_data, "r") as f:
                benchmark_data = json.load(f)

        # Generate report
        logger.info(
            f"Generating strategy performance report for {backtest_result.get('strategy_name', 'Unknown Strategy')}"
        )

        html_report = generate_strategy_performance_report(
            backtest_result=backtest_result,
            benchmark_data=benchmark_data,
            include_sections=args.sections,
        )

        # Write report to file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html_report)

        logger.info(f"Report saved to {args.output}")

    except Exception as e:
        logger.error(f"Failed to generate strategy performance report: {e}")
        sys.exit(1)


def generate_on_demand_strategy_report_cli(args):
    """Generate on-demand strategy performance report from database."""
    try:
        # Validate inputs
        if not args.strategy_id and not args.backtest_id:
            logger.error("Either --strategy-id or --backtest-id must be provided")
            sys.exit(1)

        # Get database session and connector
        db = get_db_session()
        db_connector = DBConnector(
            "postgresql://arb_user:strongpassword@localhost:5432/usdc_arbitrage"
        )

        # Create report generator
        report_generator = get_on_demand_report_generator(db, db_connector)

        # Generate report
        logger.info("Generating on-demand strategy performance report")

        report = report_generator.generate_strategy_performance_report(
            strategy_id=args.strategy_id,
            backtest_id=args.backtest_id,
            start_date=args.start_date,
            end_date=args.end_date,
            include_benchmark=args.include_benchmark,
            benchmark_symbol=args.benchmark_symbol,
            include_sections=args.sections,
            output_format=args.format,
        )

        # Write report to file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report["content"])

        logger.info(f"Report saved to {args.output}")

    except Exception as e:
        logger.error(f"Failed to generate on-demand strategy report: {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    if args.report_type == "arbitrage":
        generate_arbitrage_report_cli(args)
    elif args.report_type == "strategy":
        generate_strategy_report_cli(args)
    elif args.report_type == "on-demand-strategy":
        generate_on_demand_strategy_report_cli(args)
    else:
        logger.error("No report type specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
