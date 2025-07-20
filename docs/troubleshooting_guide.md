# Troubleshooting Guide

This guide provides solutions for common issues encountered when using the USDC Arbitrage Backtesting System.

## Table of Contents

1. [API Issues](#api-issues)
2. [Authentication Problems](#authentication-problems)
3. [Data Pipeline Issues](#data-pipeline-issues)
4. [Backtesting Engine Problems](#backtesting-engine-problems)
5. [Database Issues](#database-issues)
6. [Performance Problems](#performance-problems)
7. [Deployment Issues](#deployment-issues)
8. [Monitoring and Alerting](#monitoring-and-alerting)

## API Issues

### API Endpoint Returns 404

**Symptoms:**
- API endpoint returns a 404 Not Found error

**Possible Causes:**
- Incorrect API URL
- Endpoint not implemented
- API service not running

**Solutions:**
1. Verify the API URL is correct
2. Check API documentation to confirm the endpoint exists
3. Verify the API service is running:
   ```bash
   docker-compose ps
   # or
   kubectl get pods -n usdc-arbitrage
   ```
4. Check API logs for errors:
   ```bash
   docker-compose logs api
   # or
   kubectl logs deployment/api -n usdc-arbitrage
   ```

### API Returns 500 Internal Server Error

**Symptoms:**
- API endpoint returns a 500 Internal Server Error

**Possible Causes:**
- Exception in API code
- Database connection issue
- Resource constraints

**Solutions:**
1. Check API logs for detailed error information:
   ```bash
   docker-compose logs api
   # or
   kubectl logs deployment/api -n usdc-arbitrage
   ```
2. Verify database connection:
   ```bash
   docker-compose exec api python -c "from src.config.database import get_db; next(get_db())"
   ```
3. Check system resources (CPU, memory):
   ```bash
   docker stats
   # or
   kubectl top pods -n usdc-arbitrage
   ```
4. Restart the API service:
   ```bash
   docker-compose restart api
   # or
   kubectl rollout restart deployment/api -n usdc-arbitrage
   ```

### API Rate Limiting Issues

**Symptoms:**
- API returns 429 Too Many Requests
- Requests are being throttled

**Possible Causes:**
- Exceeding rate limits
- Misconfigured rate limiting settings

**Solutions:**
1. Reduce request frequency
2. Implement request batching
3. Check rate limit configuration:
   ```bash
   docker-compose exec api cat src/config/settings.py | grep RATE_LIMIT
   ```
4. Request rate limit increase (for premium users)

## Authentication Problems

### JWT Token Authentication Failure

**Symptoms:**
- API returns 401 Unauthorized
- "Invalid token" or "Token expired" error

**Possible Causes:**
- Expired JWT token
- Invalid token format
- Incorrect JWT secret

**Solutions:**
1. Request a new token:
   ```bash
   curl -X POST https://api.myusdcarbitrage.com/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "your_username", "password": "your_password"}'
   ```
2. Check token expiration:
   ```bash
   # Decode JWT token (replace YOUR_TOKEN)
   echo "YOUR_TOKEN" | cut -d "." -f 2 | base64 -d 2>/dev/null | jq .
   ```
3. Verify JWT secret configuration:
   ```bash
   docker-compose exec api cat .env | grep JWT_SECRET
   ```
4. Use refresh token to get a new access token:
   ```bash
   curl -X POST https://api.myusdcarbitrage.com/api/auth/refresh \
     -H "Content-Type: application/json" \
     -d '{"refresh_token": "your_refresh_token"}'
   ```

### User Account Issues

**Symptoms:**
- Unable to log in
- "Account disabled" error
- Permission denied errors

**Possible Causes:**
- Account locked or disabled
- Incorrect credentials
- Insufficient permissions

**Solutions:**
1. Verify username and password
2. Check account status:
   ```bash
   docker-compose exec api python -c "from src.models.user import User; from src.config.database import get_db; db = next(get_db()); print(db.query(User).filter(User.username == 'your_username').first().is_active)"
   ```
3. Reset password (admin only):
   ```bash
   docker-compose exec api python scripts/reset_password.py --username your_username --new_password new_secure_password
   ```
4. Check user roles and permissions:
   ```bash
   docker-compose exec api python -c "from src.models.user import User, Role; from src.config.database import get_db; db = next(get_db()); user = db.query(User).filter(User.username == 'your_username').first(); print([role.name for role in user.roles])"
   ```

## Data Pipeline Issues

### Data Download Failures

**Symptoms:**
- Data download jobs fail
- Missing data for specific exchanges or timeframes
- Error logs showing API connection issues

**Possible Causes:**
- Exchange API rate limits
- Network connectivity issues
- Invalid API credentials
- Exchange API changes

**Solutions:**
1. Check data downloader logs:
   ```bash
   cat data_downloader.log
   ```
2. Verify exchange API status:
   ```bash
   curl -I https://api.exchange.com/status
   ```
3. Check API credentials:
   ```bash
   docker-compose exec api cat .env | grep EXCHANGE_API
   ```
4. Implement retry with backoff:
   ```bash
   docker-compose exec api python src/download_usdc_data.py --exchange coinbase --retry
   ```
5. Manually trigger data download:
   ```bash
   docker-compose exec api python src/download_usdc_data.py --exchange coinbase --symbol USDC/USD --timeframe 1h --start 2023-01-01 --end 2023-01-31
   ```

### Data Validation Errors

**Symptoms:**
- Data validation failures
- Anomalies detected in data
- Gaps in time series data

**Possible Causes:**
- Corrupt data from source
- Exchange API issues
- Data processing errors

**Solutions:**
1. Check validation logs:
   ```bash
   cat data_validation.log
   ```
2. Run manual validation:
   ```bash
   docker-compose exec api python -c "from src.data.validator import validate_data; validate_data('coinbase', 'USDC/USD', '1h')"
   ```
3. Identify and fill data gaps:
   ```bash
   docker-compose exec api python -c "from src.data.gap_detector import detect_and_fill_gaps; detect_and_fill_gaps('coinbase', 'USDC/USD', '1h')"
   ```
4. Reset validation flags for reprocessing:
   ```bash
   docker-compose exec api python -c "from src.config.database import get_db; db = next(get_db()); db.execute('UPDATE market_data SET is_validated = FALSE WHERE exchange = \\'coinbase\\' AND symbol = \\'USDC/USD\\'')"
   ```

## Backtesting Engine Problems

### Backtest Job Failures

**Symptoms:**
- Backtest jobs fail to complete
- Error messages in backtest logs
- Timeout errors

**Possible Causes:**
- Insufficient resources
- Data quality issues
- Strategy implementation errors
- Timeout limits exceeded

**Solutions:**
1. Check backtest logs:
   ```bash
   cat backtesting.log
   ```
2. Verify data availability:
   ```bash
   docker-compose exec api python -c "from src.data.availability import check_data_availability; print(check_data_availability('coinbase', 'USDC/USD', '1h', '2023-01-01', '2023-01-31'))"
   ```
3. Run with smaller date range:
   ```bash
   curl -X POST https://api.myusdcarbitrage.com/api/backtest/run \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -d '{
       "strategy_id": 1,
       "start_date": "2023-01-01T00:00:00Z",
       "end_date": "2023-01-07T23:59:59Z",
       "initial_capital": 10000.0,
       "exchanges": ["coinbase"],
       "timeframes": ["1h"]
     }'
   ```
4. Increase worker resources:
   ```bash
   # Docker Compose
   docker-compose up -d --scale worker=3
   
   # Kubernetes
   kubectl scale deployment worker --replicas=3 -n usdc-arbitrage
   ```
5. Debug strategy implementation:
   ```bash
   docker-compose exec api python -c "from src.strategies.manager import get_strategy; strategy = get_strategy(1); print(strategy.parameters)"
   ```

### Performance Metrics Calculation Issues

**Symptoms:**
- Missing or incorrect performance metrics
- NaN values in results
- Inconsistent metrics across runs

**Possible Causes:**
- Division by zero in calculations
- Insufficient trade data
- Implementation errors in metrics calculation

**Solutions:**
1. Check for division by zero:
   ```bash
   docker-compose exec api python -c "from src.backtesting.metrics import check_metrics_calculation; check_metrics_calculation(backtest_id=1)"
   ```
2. Verify trade count:
   ```bash
   docker-compose exec api python -c "from src.config.database import get_db; db = next(get_db()); print(db.execute('SELECT COUNT(*) FROM backtest_trades WHERE backtest_id = 1').scalar())"
   ```
3. Recalculate metrics:
   ```bash
   docker-compose exec api python -c "from src.backtesting.metrics import recalculate_metrics; recalculate_metrics(backtest_id=1)"
   ```

## Database Issues

### Database Connection Failures

**Symptoms:**
- "Could not connect to database" errors
- API service fails to start
- Timeout errors on database operations

**Possible Causes:**
- Database service not running
- Incorrect connection parameters
- Network issues
- Resource constraints

**Solutions:**
1. Check database service status:
   ```bash
   docker-compose ps postgres
   # or
   kubectl get pods -l app=postgres -n usdc-arbitrage
   ```
2. Verify connection parameters:
   ```bash
   docker-compose exec api cat .env | grep DB_
   ```
3. Test database connection:
   ```bash
   docker-compose exec postgres psql -U postgres -d usdc_arbitrage -c "SELECT 1"
   ```
4. Check database logs:
   ```bash
   docker-compose logs postgres
   # or
   kubectl logs -l app=postgres -n usdc-arbitrage
   ```
5. Restart database service:
   ```bash
   docker-compose restart postgres
   # or
   kubectl rollout restart statefulset/postgres -n usdc-arbitrage
   ```

### Database Performance Issues

**Symptoms:**
- Slow query performance
- Timeouts on complex operations
- High CPU/memory usage

**Possible Causes:**
- Missing indexes
- Inefficient queries
- Insufficient resources
- Large tables without partitioning

**Solutions:**
1. Identify slow queries:
   ```bash
   docker-compose exec postgres psql -U postgres -d usdc_arbitrage -c "SELECT query, calls, total_time, mean_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"
   ```
2. Add missing indexes:
   ```bash
   docker-compose exec postgres psql -U postgres -d usdc_arbitrage -c "CREATE INDEX IF NOT EXISTS idx_market_data_exchange_symbol_time ON market_data (exchange, symbol, timestamp);"
   ```
3. Optimize TimescaleDB chunks:
   ```bash
   docker-compose exec postgres psql -U postgres -d usdc_arbitrage -c "SELECT set_chunk_time_interval('market_data', INTERVAL '1 day');"
   ```
4. Enable TimescaleDB compression:
   ```bash
   docker-compose exec postgres psql -U postgres -d usdc_arbitrage -c "ALTER TABLE market_data SET (timescaledb.compress = true);"
   docker-compose exec postgres psql -U postgres -d usdc_arbitrage -c "SELECT add_compression_policy('market_data', INTERVAL '7 days');"
   ```
5. Increase database resources:
   ```bash
   # Update docker-compose.yml or Kubernetes resource limits
   ```

## Performance Problems

### Slow API Response Times

**Symptoms:**
- API requests take a long time to complete
- Timeouts on complex operations
- High latency reported in monitoring

**Possible Causes:**
- Inefficient database queries
- Missing caching
- Resource constraints
- Network latency

**Solutions:**
1. Implement Redis caching:
   ```python
   # Example Redis caching implementation
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def get_cached_data(exchange, symbol, timeframe):
       # Implementation
   ```
2. Optimize database queries:
   ```python
   # Use specific columns instead of SELECT *
   db.query(MarketData.timestamp, MarketData.close).filter(...)
   ```
3. Add database indexes for common queries
4. Implement pagination for large result sets:
   ```python
   def get_data(page=1, limit=100):
       offset = (page - 1) * limit
       return db.query(MarketData).offset(offset).limit(limit).all()
   ```
5. Profile API endpoints:
   ```bash
   docker-compose exec api python -m cProfile -o api_profile.prof src/api/main.py
   ```

### Memory Leaks

**Symptoms:**
- Increasing memory usage over time
- OOM (Out of Memory) errors
- Service restarts due to memory limits

**Possible Causes:**
- Memory leaks in code
- Large objects not being garbage collected
- Inefficient data processing

**Solutions:**
1. Monitor memory usage:
   ```bash
   docker stats
   # or
   kubectl top pods -n usdc-arbitrage
   ```
2. Use memory profiling:
   ```bash
   docker-compose exec api python -m memory_profiler src/performance_optimization.py
   ```
3. Implement proper cleanup in long-running processes:
   ```python
   # Example cleanup
   import gc
   
   def process_large_dataset():
       # Processing
       gc.collect()  # Force garbage collection
   ```
4. Set appropriate memory limits:
   ```bash
   # Docker Compose
   docker-compose up -d --scale api=3 --memory=2g
   
   # Kubernetes
   # Update resource limits in deployment YAML
   ```

## Deployment Issues

### Docker Deployment Problems

**Symptoms:**
- Services fail to start
- Container exits immediately
- "Image not found" errors

**Possible Causes:**
- Missing environment variables
- Incorrect Docker configuration
- Resource constraints
- Permission issues

**Solutions:**
1. Check container logs:
   ```bash
   docker-compose logs
   ```
2. Verify environment variables:
   ```bash
   docker-compose config
   ```
3. Pull latest images:
   ```bash
   docker-compose pull
   ```
4. Check disk space:
   ```bash
   df -h
   ```
5. Rebuild containers:
   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   ```

### Kubernetes Deployment Issues

**Symptoms:**
- Pods in CrashLoopBackOff state
- Pending pods
- Service not accessible

**Possible Causes:**
- Resource constraints
- Configuration errors
- Secret or ConfigMap issues
- Network policies

**Solutions:**
1. Check pod status:
   ```bash
   kubectl get pods -n usdc-arbitrage
   ```
2. View pod logs:
   ```bash
   kubectl logs pod/pod-name -n usdc-arbitrage
   ```
3. Describe pod for events:
   ```bash
   kubectl describe pod/pod-name -n usdc-arbitrage
   ```
4. Verify secrets and config maps:
   ```bash
   kubectl get secrets -n usdc-arbitrage
   kubectl get configmaps -n usdc-arbitrage
   ```
5. Check resource allocation:
   ```bash
   kubectl describe nodes | grep -A 10 "Allocated resources"
   ```
6. Restart deployments:
   ```bash
   kubectl rollout restart deployment/api -n usdc-arbitrage
   ```

## Monitoring and Alerting

### Missing Alerts

**Symptoms:**
- No alerts received for system issues
- Monitoring dashboards show problems but no alerts triggered

**Possible Causes:**
- Alert configuration issues
- Notification channel problems
- Threshold settings incorrect

**Solutions:**
1. Check alert configuration:
   ```bash
   docker-compose exec prometheus cat /etc/prometheus/alerts.yml
   # or
   kubectl exec -it prometheus-0 -n monitoring -- cat /etc/prometheus/alerts.yml
   ```
2. Test notification channels:
   ```bash
   # For email alerts
   docker-compose exec alertmanager amtool alert add alertname=TestAlert severity=info --annotation=summary="Test alert"
   ```
3. Verify alert manager status:
   ```bash
   curl -s http://localhost:9093/api/v2/status | jq
   ```
4. Check alert thresholds:
   ```bash
   # Example: CPU usage alert threshold
   docker-compose exec prometheus cat /etc/prometheus/rules/cpu_usage.yml
   ```

### Monitoring System Issues

**Symptoms:**
- Monitoring dashboards not updating
- Missing metrics
- Grafana shows "No data" errors

**Possible Causes:**
- Prometheus not scraping targets
- Metrics endpoint issues
- Storage problems

**Solutions:**
1. Check Prometheus targets:
   ```bash
   curl -s http://localhost:9090/api/v1/targets | jq
   ```
2. Verify metrics endpoints:
   ```bash
   curl -s http://localhost:8000/metrics
   ```
3. Check Prometheus storage:
   ```bash
   docker-compose exec prometheus df -h /prometheus
   ```
4. Restart monitoring stack:
   ```bash
   docker-compose restart prometheus grafana
   # or
   kubectl rollout restart statefulset/prometheus -n monitoring
   kubectl rollout restart deployment/grafana -n monitoring
   ```

## Additional Resources

If you continue to experience issues after trying these troubleshooting steps, please consult:

1. [System Architecture Documentation](system_architecture.md)
2. [API Documentation](api_documentation.md)
3. [Deployment and Maintenance Guide](deployment_maintenance.md)

For urgent issues, contact the support team:
- Email: support@myusdcarbitrage.com
- Slack: #usdc-arbitrage-support