# Centralized Logging System

This directory contains the centralized logging system for the USDC Arbitrage Backtesting API. The system provides comprehensive logging, error tracking, performance profiling, and log analysis capabilities.

## Components

### Centralized Logging

The `centralized_logging.py` module provides a unified interface to all logging components:

- Structured logging with JSON format
- Error tracking with automatic categorization
- Performance profiling and bottleneck identification
- Log aggregation and search capabilities

### Log Configuration

The `logging_config.py` module configures the logging system:

- Structured JSON formatter
- Redis log handler for centralized collection
- Performance logger for execution time and memory usage
- Error tracker for automatic error categorization

### Log Aggregation

The `log_aggregation.py` module provides log aggregation and search capabilities:

- Search logs with various filters
- Export logs in multiple formats (JSON, CSV, TXT)
- Generate log statistics
- Clean up old logs

### Log Analysis

The `log_analysis.py` module analyzes logs to identify patterns and insights:

- Identify error patterns and recurring issues
- Detect log volume anomalies
- Identify chains of related errors
- Generate insights from log data

### Performance Profiling

The `performance_profiler.py` module provides performance profiling and bottleneck identification:

- Profile function execution time and memory usage
- Identify performance bottlenecks
- Track database query performance
- Generate performance trends

### Log Rotation

The `log_rotation.py` module handles log rotation and archiving:

- Rotate logs based on size
- Compress rotated logs
- Archive old logs
- Clean up old backups

### Database Query Logging

The `db_query_logging.py` module provides database query logging:

- Log query execution time
- Track slow queries
- Count queries per request
- Log query parameters

## Usage

### Basic Logging

```python
import logging

# Get logger
logger = logging.getLogger(__name__)

# Log messages
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# Log with extra data
logger.info("User logged in", extra={"user_id": 123, "ip_address": "192.168.1.1"})
```

### Error Tracking

```python
from src.monitoring.centralized_logging import track_error

try:
    # Some code that might raise an exception
    result = 1 / 0
except Exception as e:
    # Track error
    error_id = track_error(
        error=e,
        context="Division operation",
        metadata={"value": 0},
    )
    print(f"Error tracked with ID: {error_id}")
```

### Performance Profiling

```python
from src.monitoring.centralized_logging import profile
from src.monitoring.performance_profiler import profile_block, profile_async_block

# Profile function with decorator
@profile(operation_name="my_function")
def my_function():
    # Function code
    pass

# Profile code block
with profile_block("my_operation"):
    # Code to profile
    pass

# Profile async code block
async def my_async_function():
    async with profile_async_block("my_async_operation"):
        # Async code to profile
        pass
```

### Database Query Logging

```python
from src.monitoring.db_query_logging import log_query

# Log database query with decorator
@log_query
def execute_query(session, query):
    return session.execute(query)
```

### Log Search and Analysis

```python
import asyncio
from src.monitoring.centralized_logging import get_centralized_logging

async def analyze_logs():
    centralized_logging = get_centralized_logging()
    
    # Search logs
    logs = await centralized_logging.search_logs(
        query="error",
        log_level="error",
        limit=100,
    )
    
    # Analyze error patterns
    patterns = await centralized_logging.analyze_error_patterns(hours=24)
    
    # Generate insights
    insights = await centralized_logging.generate_log_insights(days=7)
    
    # Get dashboard data
    dashboard = await centralized_logging.get_logging_dashboard(hours=24)
    
    return dashboard

# Run analysis
dashboard = asyncio.run(analyze_logs())
```

## API Endpoints

The centralized logging system provides the following API endpoints:

- `GET /logging/centralized/search` - Search logs with various filters
- `POST /logging/centralized/search` - Search logs with request body
- `GET /logging/centralized/statistics` - Get log statistics
- `GET /logging/centralized/export` - Export logs in multiple formats
- `POST /logging/centralized/errors/track` - Track an error manually
- `GET /logging/centralized/performance/bottlenecks` - Analyze performance bottlenecks
- `GET /logging/centralized/analysis/error-patterns` - Analyze error patterns
- `GET /logging/centralized/insights` - Generate log insights
- `GET /logging/centralized/dashboard` - Get comprehensive logging dashboard
- `POST /logging/centralized/cleanup` - Clean up old logs
- `POST /logging/centralized/rotate` - Manually trigger log rotation
- `POST /logging/centralized/archive` - Archive old logs

## Configuration

The centralized logging system can be configured with the following environment variables:

- `LOG_LEVEL` - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `REDIS_URL` - Redis URL for centralized log collection
- `LOG_FILE` - Log file path
- `LOG_ROTATION_SIZE_MB` - Maximum log file size in MB
- `LOG_ROTATION_BACKUP_COUNT` - Number of backup files to keep
- `LOG_ROTATION_COMPRESS` - Whether to compress rotated logs
- `LOG_ARCHIVE_DIR` - Directory for archived logs