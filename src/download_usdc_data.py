"""USDC data downloader with enhanced retry logic and data validation."""

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta

from api.data_downloader import EnhancedDataDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_downloader.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download USDC price data")

    parser.add_argument(
        "--exchanges",
        type=str,
        default="coinbase,kraken,binance",
        help="Comma-separated list of exchanges (default: coinbase,kraken,binance)",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="USDC/USD",
        help="Trading pair symbol (default: USDC/USD)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe (e.g., 1m, 5m, 1h, 1d) (default: 1h)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to download (default: 7)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (overrides --days)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now(UTC).strftime("%Y-%m-%d"),
        help=f"End date in YYYY-MM-DD format (default: today)",
    )

    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill missing data using other exchanges",
    )

    parser.add_argument(
        "--resolve-conflicts",
        choices=["newer", "older", "average"],
        default="newer",
        help="Conflict resolution strategy (default: newer)",
    )

    parser.add_argument(
        "--db-connection",
        type=str,
        default="postgresql://postgres:postgres@localhost:5432/usdc_arbitrage",
        help="Database connection string",
    )

    return parser.parse_args()


def main():
    """Main function to download and store USDC price data."""
    args = parse_arguments()

    # Parse exchanges
    exchanges = [exchange.strip() for exchange in args.exchanges.split(",")]

    # Parse dates
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=UTC)

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        start_date = end_date - timedelta(days=args.days)

    logger.info(f"Downloading data for {args.symbol} from {start_date} to {end_date}")
    logger.info(f"Exchanges: {exchanges}")
    logger.info(f"Timeframe: {args.timeframe}")

    # Initialize downloader
    downloader = EnhancedDataDownloader(args.db_connection)

    # Download and store data
    results = downloader.download_and_store_data(
        exchanges, args.symbol, args.timeframe, start_date, end_date
    )

    # Resolve conflicts if requested
    if args.resolve_conflicts:
        for exchange in exchanges:
            resolved = downloader.resolve_conflicts(
                exchange, args.symbol, args.timeframe, args.resolve_conflicts
            )
            logger.info(f"Resolved {resolved} conflicts for {exchange}")

    # Backfill missing data if requested
    if args.backfill and len(exchanges) > 1:
        primary_exchange = exchanges[0]
        fallback_exchanges = exchanges[1:]

        logger.info(
            f"Backfilling missing data for {primary_exchange} using {fallback_exchanges}"
        )

        backfill_results = downloader.backfill_missing_data(
            primary_exchange,
            fallback_exchanges,
            args.symbol,
            args.timeframe,
            start_date,
            end_date,
        )

        logger.info(f"Backfill results: {backfill_results}")

    # Log results
    logger.info(f"Download results: {results}")

    # Print summary
    print("\nDownload Summary:")
    print("=" * 50)
    for exchange, result in results.items():
        status = result.get("status", "unknown")
        records = result.get("records", 0)

        if status == "success":
            print(f"{exchange}: Successfully downloaded {records} records")
        elif status == "no_data":
            print(f"{exchange}: No new data available")
        elif status == "validation_failed":
            issues = result.get("issues", [])
            print(f"{exchange}: Validation failed - {', '.join(issues)}")
        elif status == "error":
            error = result.get("error", "Unknown error")
            print(f"{exchange}: Error - {error}")
        else:
            print(f"{exchange}: {status}")

    print("=" * 50)


if __name__ == "__main__":
    main()
