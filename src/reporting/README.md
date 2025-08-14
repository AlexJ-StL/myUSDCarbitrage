# Reporting Module

This module provides functionality for generating reports on arbitrage opportunities and strategy performance.

## On-Demand Report Generation

The on-demand report generation system allows users to generate reports for arbitrage opportunities and strategy performance through various interfaces:

1. **API Endpoints**: Generate reports via REST API
2. **Command-Line Interface**: Generate reports via CLI
3. **Programmatic Interface**: Generate reports via Python code

### API Endpoints

#### Arbitrage Opportunity Reports

```
POST /api/v1/reports/arbitrage
GET /api/v1/reports/arbitrage
```

Parameters:
- `exchanges`: List of exchanges to compare (required, min 2)
- `symbol`: Trading symbol to analyze (default: "USDC/USD")
- `start_time`: Start time for analysis period (default: 24 hours ago)
- `end_time`: End time for analysis period (default: now)
- `threshold`: Minimum percentage difference to be considered an opportunity (default: 0.001)
- `output_format`: Output format (html, json, csv) (default: html)

#### Strategy Performance Reports

```
POST /api/v1/reports/strategy
GET /api/v1/reports/strategy
```

Parameters:
- `strategy_id`: ID of the strategy to analyze (required if backtest_id not provided)
- `backtest_id`: ID of the specific backtest to analyze (required if strategy_id not provided)
- `start_date`: Start date for analysis period (optional)
- `end_date`: End date for analysis period (optional)
- `include_benchmark`: Whether to include benchmark comparison (default: false)
- `benchmark_symbol`: Symbol to use as benchmark (default: "BTC/USD")
- `include_sections`: Specific sections to include in the report (optional)
- `output_format`: Output format (html, json, csv) (default: html)

### Command-Line Interface

#### Arbitrage Opportunity Reports

```bash
python -m src.generate_report arbitrage --exchanges coinbase kraken --symbol USDC/USD --output arbitrage_report.html
```

Parameters:
- `--exchanges`: List of exchanges to compare (required)
- `--symbol`: Trading symbol to analyze (default: "USDC/USD")
- `--start-time`: Start time for analysis period (ISO format)
- `--end-time`: End time for analysis period (ISO format)
- `--threshold`: Minimum percentage difference (default: 0.001)
- `--output`: Output file path (default: "arbitrage_report.html")
- `--format`: Output format (html, json, csv) (default: html)

#### Strategy Performance Reports (from file)

```bash
python -m src.generate_report strategy --backtest-result backtest_result.json --output strategy_report.html
```

Parameters:
- `--backtest-result`: Path to JSON file containing backtest results
- `--benchmark-data`: Path to JSON file containing benchmark data (optional)
- `--sections`: Sections to include in the report (optional)
- `--output`: Output file path (default: "strategy_report.html")
- `--format`: Output format (html, json, csv) (default: html)

#### Strategy Performance Reports (from database)

```bash
python -m src.generate_report on-demand-strategy --strategy-id 1 --output strategy_report.html
```

Parameters:
- `--strategy-id`: ID of the strategy to analyze
- `--backtest-id`: ID of the specific backtest to analyze
- `--start-date`: Start date for analysis period (ISO format)
- `--end-date`: End date for analysis period (ISO format)
- `--include-benchmark`: Include benchmark comparison
- `--benchmark-symbol`: Symbol to use as benchmark (default: "BTC/USD")
- `--sections`: Sections to include in the report (optional)
- `--output`: Output file path (default: "strategy_report.html")
- `--format`: Output format (html, json, csv) (default: html)

### Programmatic Interface

```python
from sqlalchemy.orm import Session
from src.api.database import DBConnector
from src.reporting.on_demand_report_generator import get_on_demand_report_generator

# Create database session and connector
db_session = Session()
db_connector = DBConnector("connection_string")

# Create report generator
report_generator = get_on_demand_report_generator(db_session, db_connector)

# Generate arbitrage opportunity report
arbitrage_report = report_generator.generate_arbitrage_opportunity_report(
    exchanges=["coinbase", "kraken"],
    symbol="USDC/USD",
    threshold=0.001,
    output_format="html",
)

# Generate strategy performance report
strategy_report = report_generator.generate_strategy_performance_report(
    strategy_id=1,
    include_benchmark=True,
    benchmark_symbol="BTC/USD",
    output_format="html",
)

# Access report content
html_content = arbitrage_report["content"]
```

## Report Structure

### Arbitrage Opportunity Reports

The arbitrage opportunity reports include the following sections:

1. **Executive Summary**: Key metrics and overview of arbitrage opportunities
2. **Data Analysis**: Price comparison charts and opportunity distribution
3. **Top Opportunities**: Detailed list of the best arbitrage opportunities
4. **Exchange Pair Analysis**: Analysis of exchange pairs with the most opportunities

### Strategy Performance Reports

The strategy performance reports include the following sections:

1. **Executive Summary**: Key performance metrics and overview
2. **Performance Metrics**: Detailed performance metrics
3. **Equity Curve**: Portfolio value over time
4. **Drawdown Analysis**: Analysis of drawdowns
5. **Monthly Returns**: Heatmap of monthly returns
6. **Trade Analysis**: Analysis of individual trades
7. **Risk Metrics**: Risk-adjusted performance metrics
8. **Portfolio Composition**: Breakdown of portfolio composition
9. **Position History**: History of positions over time
10. **Exchange Exposure**: Exposure to different exchanges

## Output Formats

Reports can be generated in the following formats:

1. **HTML**: Interactive reports with charts and tables
2. **JSON**: Machine-readable data format
3. **CSV**: Tabular data format for spreadsheet applications

## Customization

Reports can be customized by:

1. **Selecting specific sections** to include
2. **Choosing the output format**
3. **Filtering by date range**
4. **Including benchmark comparison**