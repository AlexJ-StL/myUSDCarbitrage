# USDC Arbitrage Backtesting System

A comprehensive backtesting system for USDC arbitrage strategies across multiple exchanges and timeframes.

## Features

- **Advanced Data Pipeline**: Validates, processes, and fills gaps in OHLCV data
- **Backtesting Engine**: Tests arbitrage strategies with realistic transaction costs and slippage
- **Strategy Management**: Version control, comparison, and A/B testing for strategies
- **Security**: JWT authentication, RBAC, and API security enhancements
- **Monitoring**: System health, business metrics, and error tracking
- **API**: Comprehensive endpoints with documentation
- **Analytics**: Performance visualization and risk management
- **Deployment**: Containerization, CI/CD, and database optimization

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL with TimescaleDB extension
- Redis (for rate limiting and caching)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/usdc-arbitrage.git
   cd usdc-arbitrage
   ```

2. Create and activate a virtual environment:

   ```bash
   uv venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Unix/MacOS
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   uv pip install -e .
   ```

4. Set up the database:

   ```bash
   # Run the database setup script
   psql -U postgres -f .sql/database_setup.sql
   ```

5. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

```bash
uvicorn src.api.main:app --reload
```

The API will be available at http://localhost:8000

### Running Tests

```bash
pytest tests/
```

## Deployment

### Docker

Build and run the Docker image:

```bash
docker build -t usdc-arbitrage .
docker run -p 8000:8000 usdc-arbitrage
```

### Kubernetes

Deploy to Kubernetes:

```bash
kubectl apply -k k8s/overlays/production
```

## Project Structure

- `src/`: Source code
  - `api/`: API endpoints and middleware
  - `data/`: Data processing and validation
  - `models/`: Database models
  - `strategies/`: Trading strategies
  - `monitoring/`: System monitoring and alerting
  - `reporting/`: Report generation
- `tests/`: Test suite
- `k8s/`: Kubernetes configuration
- `terraform/`: Infrastructure as code
- `scripts/`: Utility scripts
- `.sql/`: Database setup and migration scripts

## License

This project is licensed under the MIT License - see the LICENSE file for details.
