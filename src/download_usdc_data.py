import ccxt
import logging
from datetime import datetime, timezone, timedelta
from api.database import Database, DBConnector
from api.data_validation import DataValidator
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database
DB = Database()


def fetch_ohlcv(exchange, symbol, timeframe, since, until, sleep_time=1):
    all_ohlcv = []
    current_since = since
    # Get timeframe duration in milliseconds
    timeframe_duration = exchange.timeframes[timeframe] * 1000

    while current_since < until:
        logger.info(
            f"Fetching from {datetime.fromtimestamp(current_since/1000, timezone.utc)}"
        )
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + timeframe_duration
            time.sleep(sleep_time)
            if ohlcv[-1][0] >= until:
                break
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            time.sleep(sleep_time)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break

    return all_ohlcv


def main():
    # Initialize exchanges
    coinbase = getattr(ccxt, 'coinbase')()
    kraken = getattr(ccxt, 'kraken')()
    binance = getattr(ccxt, 'binance')()

    # Define parameters
    symbol = "USDC/USD"
    timeframe = "1h"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=1)

    # Fetch data
    coinbase_data = fetch_ohlcv(
        coinbase,
        symbol,
        timeframe,
        int(start_date.timestamp() * 1000),
        int(end_date.timestamp() * 1000),
    )
    kraken_data = fetch_ohlcv(
        kraken,
        symbol,
        timeframe,
        int(start_date.timestamp() * 1000),
        int(end_date.timestamp() * 1000),
    )
    binance_data = fetch_ohlcv(
        binance,
        symbol,
        timeframe,
        int(start_date.timestamp() * 1000),
        int(end_date.timestamp() * 1000),
    )

    # Validate data
    validator = DataValidator(connection_string="connection_string")
    coinbase_validation = validator.validate_data("coinbase", symbol, timeframe)
    kraken_validation = validator.validate_data("kraken", symbol, timeframe)
    binance_validation = validator.validate_data("binance", symbol, timeframe)

    # Check for critical errors
    if (
        coinbase_validation["price_errors"]
        or kraken_validation["price_errors"]
        or binance_validation["price_errors"]
    ):
        logger.error("Critical data issues detected")
        return

    # Insert data into database
    logger.info(f"Inserting {len(coinbase_data)} records into database...")
    DB.insert_data("coinbase", symbol, timeframe, coinbase_data)
    logger.info(f"Inserting {len(kraken_data)} records into database...")
    DB.insert_data("kraken", symbol, timeframe, kraken_data)
    logger.info(f"Inserting {len(binance_data)} records into database...")
    DB.insert_data("binance", symbol, timeframe, binance_data)


if __name__ == "__main__":
    main()
