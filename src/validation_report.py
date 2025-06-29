from .data_validation import DataValidator
import pandas as pd
import json

def generate_validation_report():
    validator = DataValidator()
    log_entries = []

    # All exchange/timeframe combinations
    exchanges = ['coinbase', 'kraken', 'binance']
    timeframes = ['1h', '4h', '1d']
    symbol_map = {
        'coinbase': 'USDC/USD',
        'kraken': 'USDC/USD',
        'binance': 'USDC/USDT'
    }

    # Include additional timeframes if needed
    for exchange in exchanges:
        for timeframe in timeframes:
            result = validator.validate_dataset(
                exchange,
                symbol_map.get(exchange),
                timeframe
            )
            log_entries.append(result)

    # Convert to dataframe
    report_df = pd.DataFrame(log_entries)

    # Calculate issue counts
    for col in ['price_errors', 'time_gaps', 'outliers', 'volume_anomalies', 'changepoints']:
        if col in report_df.columns:
            report_df[col + '_count'] = report_df[col].apply(len)

    # Save to HTML report
    report_df.to_html('data_validation_report.html', index=False)
    report_df.to_json('data_validation_report.json', indent=2, default=str)
    print("Validation report generated")

if __name__ == "__main__":
    generate_validation_report()
