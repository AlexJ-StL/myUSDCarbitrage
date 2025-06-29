import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from api.database import DBConnector
from api.data_validation import DataValidator
import logging

# Set up logging
logger = logging.getLogger('data_downloader')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('data_downloader.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


def fetch_ohlcv(exchange, symbol, timeframe, since, until, sleep_time=1):
    all_ohlcv = []
    current_since = since
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    while current_since < until:
        logger.info(f"Fetching from {datetime.utcfromtimestamp(current_since/1000)}")
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + timeframe_ms
            time.sleep(sleep_time)
            if ohlcv[-1][0] >= until:
                break
        except Exception as e:
            logger.error(f"Error: {e}")
            break
    return all_ohlcv

def main():
    db = DBConnector()
    validator = DataValidator()
    exchanges = {
        'coinbase': ccxt.coinbase(),
        'kraken': ccxt.kraken(),
        'binance': ccxt.binance()
    }
    symbol_config = {
        'coinbase': 'USDC/USD',
        'kraken': 'USDC/USD',
        'binance': 'USDC/USDT'
    }
    timeframes = ['1h', '4h', '1d']
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365*3)  # 3 years

    for name, exchange in exchanges.items():
        try:
            exchange.load_markets()
        except ccxt.ExchangeNotAvailable:
            logger.warning(f"Could not connect to {name}. It may be unavailable in your region.")
            continue
        actual_symbol = symbol_config.get(name)

        if actual_symbol not in exchange.symbols:
            logger.warning(f"{actual_symbol} not available on {name}")
            continue

        for tf in timeframes:
            logger.info(f"\nFetching {actual_symbol} from {name} ({tf})...")
            since = int(start_date.timestamp() * 1000)
            until = int(end_date.timestamp() * 1000)

            ohlcv = fetch_ohlcv(exchange, actual_symbol, tf, since, until)
            if ohlcv:
                logger.info(f"Inserting {len(ohlcv)} records into database...")
                #db.insert_ohlcv(name, actual_symbol, tf, ohlcv)

                # Validate the updated dataset
                validation_results = validator.validate_dataset(
                    name,
                    actual_symbol,
                    tf
                )

                # Handle critical errors
                if validation_results.get('price_errors'):
                    logger.error(f"Critical data issues detected for {name}/{tf}")

if __name__ == "__main__":
    main()
