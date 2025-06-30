from api.data_validation import DataValidator
import pandas as pd


def generate_validation_report():
    connection_string = (
        "postgresql://arb_user:strongpassword@localhost:5432/usdc_arbitrage"
    )
    validator = DataValidator(connection_string)
    validator.validate_data("exchange_name", "symbol_name", "timeframe_value")
    log_entries = []

    exchanges = ["coinbase", "kraken", "binance"]
    timeframes = ["1h", "4h", "1d"]
    symbol_map = {"coinbase": "USDC/USD", "kraken": "USDC/USD", "binance": "USDC/USDT"}

    for exchange in exchanges:
        for timeframe in timeframes:
            result = validator.validate_dataset(
                exchange, symbol_map.get(exchange), timeframe
            )
            log_entries.append(result)

    report_df = pd.DataFrame(log_entries)

    for col in [
        "price_errors",
        "time_gaps",
        "outliers",
        "volume_anomalies",
        "changepoints",
    ]:
        if col in report_df.columns:
            report_df[col + "_count"] = report_df[col].apply(len)

    report_df.to_html("data_validation_report.html", index=False)
    print("Validation report generated")


if __name__ == "__main__":
    generate_validation_report()
