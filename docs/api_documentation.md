# USDC Arbitrage Backtesting API Documentation

## API Overview

The USDC Arbitrage Backtesting System provides a comprehensive REST API that allows users to interact with all aspects of the system. This documentation covers the available endpoints, authentication methods, request/response formats, and best practices.

## Base URL

All API endpoints are accessible at:

```
https://api.myusdcarbitrage.com/api
```

## Authentication

### JWT Authentication

All API requests require authentication using JSON Web Tokens (JWT). To obtain a token:

```http
POST /auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

Include the access token in the Authorization header for all subsequent requests:

```http
GET /strategies
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Token Refresh

When the access token expires, use the refresh token to obtain a new one:

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### API Keys

For automated systems, you can use API keys instead of JWT tokens:

```http
GET /strategies
X-API-Key: your_api_key
```

## Rate Limiting

API requests are subject to rate limiting to prevent abuse. The current limits are:

- 100 requests per minute for standard users
- 300 requests per minute for premium users

Rate limit information is included in the response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1626816000
```

## API Endpoints

### Authentication Endpoints

#### Login

```http
POST /auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

#### Refresh Token

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### Logout

```http
POST /auth/logout
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### User Management Endpoints

#### Get Current User

```http
GET /users/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Update User

```http
PATCH /users/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "email": "new_email@example.com"
}
```

#### Change Password

```http
POST /users/change-password
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "current_password": "current_password",
  "new_password": "new_password"
}
```

### Data Management Endpoints

#### Check Data Availability

```http
GET /data/availability?exchange=coinbase&symbol=USDC/USD
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Download Data

```http
POST /data/download
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "exchange": "coinbase",
  "symbol": "USDC/USD",
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z"
}
```

#### Check Data Validation Status

```http
GET /data/validation-status?exchange=coinbase&symbol=USDC/USD
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Get Data Gaps

```http
GET /data/gaps?exchange=coinbase&symbol=USDC/USD&start_date=2023-01-01T00:00:00Z&end_date=2023-01-31T23:59:59Z
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Strategy Management Endpoints

#### List Strategies

```http
GET /strategies
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Get Strategy

```http
GET /strategies/1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Create Strategy

```http
POST /strategies
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "name": "Simple Arbitrage Strategy",
  "description": "Basic arbitrage strategy between exchanges",
  "strategy_type": "arbitrage",
  "parameters": {
    "threshold": 0.001,
    "exchanges": ["coinbase", "kraken", "binance"],
    "symbols": ["USDC/USD"],
    "position_size": 1000.0
  }
}
```

#### Update Strategy

```http
PATCH /strategies/1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "parameters": {
    "threshold": 0.002
  }
}
```

#### Delete Strategy

```http
DELETE /strategies/1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Get Strategy Versions

```http
GET /strategies/1/versions
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Rollback Strategy

```http
POST /strategies/1/rollback
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "version": 2
}
```

### Backtesting Endpoints

#### Run Backtest

```http
POST /backtest/run
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "strategy_id": 1,
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "initial_capital": 10000.0,
  "exchanges": ["coinbase", "kraken"],
  "timeframes": ["1h"]
}
```

#### Check Backtest Status

```http
GET /backtest/status/12345678-1234-5678-1234-567812345678
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Cancel Backtest

```http
POST /backtest/cancel/12345678-1234-5678-1234-567812345678
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### List Backtests

```http
GET /backtest/list?strategy_id=1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Results Analysis Endpoints

#### Get Backtest Results

```http
GET /results/1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Get Performance Metrics

```http
GET /results/1/metrics
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Get Trade Analysis

```http
GET /results/1/trades
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Compare Results

```http
POST /results/compare
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "backtest_ids": [1, 2]
}
```

### Visualization Endpoints

#### Get Equity Curve

```http
GET /visualization/equity-curve/1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Get Drawdown Chart

```http
GET /visualization/drawdown/1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Get Trade Distribution

```http
GET /visualization/trade-distribution/1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Report Generation Endpoints

#### Generate Report

```http
POST /results/generate-report
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "backtest_id": 1,
  "format": "html"
}
```

#### Export Data

```http
GET /data-export/backtest/1?format=csv
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Advanced Features Endpoints

#### Walk-Forward Optimization

```http
POST /backtest/walk-forward
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "strategy_id": 1,
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-06-30T23:59:59Z",
  "initial_capital": 10000.0,
  "window_size": 30,
  "step_size": 7,
  "parameter_ranges": {
    "threshold": [0.0005, 0.001, 0.002, 0.003]
  }
}
```

#### Monte Carlo Simulation

```http
POST /backtest/monte-carlo
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "backtest_id": 1,
  "simulations": 1000,
  "confidence_level": 0.95
}
```

#### Risk Analysis

```http
GET /results/1/risk
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### System Monitoring Endpoints

#### Health Check

```http
GET /monitoring/health
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### System Metrics

```http
GET /monitoring/metrics
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Data Pipeline Status

```http
GET /monitoring/data-pipeline
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "threshold",
      "issue": "Value must be greater than 0"
    }
  }
}
```

### Common Error Codes

| Code                    | Description                  |
| ----------------------- | ---------------------------- |
| `AUTHENTICATION_ERROR`  | Authentication failed        |
| `AUTHORIZATION_ERROR`   | Insufficient permissions     |
| `VALIDATION_ERROR`      | Invalid input parameters     |
| `RESOURCE_NOT_FOUND`    | Requested resource not found |
| `RATE_LIMIT_EXCEEDED`   | API rate limit exceeded      |
| `INTERNAL_SERVER_ERROR` | Server encountered an error  |

## Pagination

For endpoints that return multiple items, pagination is supported:

```http
GET /strategies?page=2&limit=10
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Response includes pagination metadata:

```json
{
  "items": [...],
  "pagination": {
    "total": 45,
    "page": 2,
    "limit": 10,
    "pages": 5
  }
}
```

## Filtering and Sorting

Many endpoints support filtering and sorting:

```http
GET /strategies?sort=created_at:desc&strategy_type=arbitrage
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Versioning

The API uses versioning to ensure backward compatibility:

```http
GET /v1/strategies
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Best Practices

1. **Rate Limiting**: Respect rate limits to avoid being throttled
2. **Pagination**: Use pagination for endpoints that return multiple items
3. **Error Handling**: Implement proper error handling in your client code
4. **Token Management**: Securely store and refresh tokens
5. **Filtering**: Use filtering to reduce payload size and improve performance

## SDK Libraries

Official client libraries are available for:

- Python: [GitHub Repository](https://github.com/myusdcarbitrage/python-client)
- JavaScript: [GitHub Repository](https://github.com/myusdcarbitrage/js-client)

## Interactive Documentation

Interactive API documentation is available at:

```
https://api.myusdcarbitrage.com/api/docs
```

This Swagger UI allows you to explore and test all API endpoints directly from your browser.