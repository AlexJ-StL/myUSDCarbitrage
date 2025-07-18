# USDC Arbitrage API Client SDKs

This directory contains client SDKs for the USDC Arbitrage API in multiple programming languages.

## Python Client

The Python client provides a comprehensive interface to the USDC Arbitrage API.

### Installation

```bash
pip install usdc-arbitrage-client
```

### Usage

```python
from usdc_arbitrage_client import USDCArbitrageClient

# Initialize client
client = USDCArbitrageClient(
    base_url="http://localhost:8000",
    api_version="1.0"
)

# Login
token_response = client.login("username", "password")

# List strategies
strategies = client.list_strategies(active_only=True, limit=10)

# Run a backtest
backtest_request = {
    "strategy_id": 1,
    "start_date": "2025-01-01T00:00:00Z",
    "end_date": "2025-01-31T23:59:59Z",
    "exchanges": ["coinbase", "kraken", "binance"],
    "symbols": ["USDC/USD"],
    "timeframe": "1h",
    "initial_balance": 10000.0
}
backtest_response = client.run_backtest(backtest_request)

# Get backtest result
result = client.get_backtest(backtest_response["backtest_id"])

# Export data
data = client.export_data(
    data_type="market_data",
    format="csv",
    start_date="2025-01-01T00:00:00Z",
    end_date="2025-01-31T23:59:59Z",
    exchanges=["coinbase"]
)

# Save exported data
with open("market_data.csv", "wb") as f:
    f.write(data)
```

## JavaScript Client

The JavaScript client provides a comprehensive interface to the USDC Arbitrage API for browser and Node.js environments.

### Installation

```bash
npm install usdc-arbitrage-client
```

### Usage

```javascript
const USDCArbitrageClient = require('usdc-arbitrage-client');

// Initialize client
const client = new USDCArbitrageClient(
    'http://localhost:8000',
    '1.0'
);

// Login
client.login('username', 'password')
    .then(tokenResponse => {
        console.log('Logged in successfully');
        
        // List strategies
        return client.listStrategies({ activeOnly: true, limit: 10 });
    })
    .then(strategies => {
        console.log('Strategies:', strategies);
        
        // Run a backtest
        const backtestRequest = {
            strategy_id: 1,
            start_date: '2025-01-01T00:00:00Z',
            end_date: '2025-01-31T23:59:59Z',
            exchanges: ['coinbase', 'kraken', 'binance'],
            symbols: ['USDC/USD'],
            timeframe: '1h',
            initial_balance: 10000.0
        };
        
        return client.runBacktest(backtestRequest);
    })
    .then(backtestResponse => {
        console.log('Backtest started:', backtestResponse);
        
        // Get backtest result
        return client.getBacktest(backtestResponse.backtest_id);
    })
    .then(result => {
        console.log('Backtest result:', result);
        
        // Export data
        return client.exportData({
            dataType: 'market_data',
            format: 'csv',
            startDate: new Date('2025-01-01T00:00:00Z'),
            endDate: new Date('2025-01-31T23:59:59Z'),
            exchanges: ['coinbase']
        });
    })
    .then(data => {
        console.log('Exported data received');
        
        // Save exported data (Node.js environment)
        const fs = require('fs');
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                fs.writeFile('market_data.csv', reader.result, err => {
                    if (err) reject(err);
                    else resolve();
                });
            };
            reader.onerror = reject;
            reader.readAsArrayBuffer(data);
        });
    })
    .then(() => {
        console.log('Data saved to market_data.csv');
    })
    .catch(error => {
        console.error('Error:', error);
    });

// WebSocket example
const ws = client.connectToBacktestWebSocket(
    'client123',
    message => console.log('Received message:', message),
    error => console.error('WebSocket error:', error),
    event => console.log('WebSocket closed:', event)
);

// Subscribe to backtest updates
client.subscribeToBacktest(ws, 1);

// Later, unsubscribe
client.unsubscribeFromBacktest(ws, 1);

// Close connection when done
ws.close();
```

## API Versioning

The API supports versioning through the `Accept-Version` header. Both client SDKs allow you to specify the API version when initializing the client.

Available versions:
- `1.0` (default)
- `2.0` (latest)

Example:
```python
# Python
client = USDCArbitrageClient(api_version="2.0")
```

```javascript
// JavaScript
const client = new USDCArbitrageClient('http://localhost:8000', '2.0');
```

## Authentication

Both clients support JWT authentication. After successful login, the token is stored in the client instance and automatically used for subsequent requests.

## Error Handling

The clients provide comprehensive error handling with detailed error messages from the API.

## WebSocket Support

Both clients support WebSocket connections for real-time backtest monitoring.