"""Centralized logging configuration with structured format."""

import json
import logging
import logging.config
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def __init__(self, include_extra: bool = True):
        """Initialize structured formatter."""
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "getMessage",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                }:
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)

            if extra_fields:
                log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class RedisLogHandler(logging.Handler):
    """Custom log handler that sends logs to Redis for centralized collection."""

    def __init__(
        self, redis_url: str = "redis://localhost:6379/0", key_prefix: str = "logs"
    ):
        """Initialize Redis log handler."""
        super().__init__()
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis_client: Optional[redis.Redis] = None
        self.max_logs = 10000  # Maximum logs to keep in Redis

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    def emit(self, record: logging.LogRecord):
        """Emit log record to Redis."""
        try:
            # Format the log record
            log_message = self.format(record)

            # Create Redis key based on log level and timestamp
            timestamp = int(record.created)
            key = f"{self.key_prefix}:{record.levelname.lower()}:{timestamp}"

            # Store log in Redis with TTL (7 days)
            self.redis_client.setex(key, 7 * 24 * 3600, log_message)

            # Add to sorted set for time-based queries
            self.redis_client.zadd(
                f"{self.key_prefix}_index:{record.levelname.lower()}",
                {key: record.created},
            )

            # Maintain maximum log count
            self.redis_client.zremrangebyrank(
                f"{self.key_prefix}_index:{record.levelname.lower()}",
                0,
                -(self.max_logs + 1),
            )

        except Exception as e:
            # Don't let logging errors break the application
            print(f"Failed to emit log to Redis: {e}", file=sys.stderr)


class PerformanceLogger:
    """Logger for performance profiling and bottleneck identification."""

    def __init__(self, logger_name: str = "performance"):
        """Initialize performance logger."""
        self.logger = logging.getLogger(logger_name)

    def log_execution_time(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log execution time for an operation."""
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                "performance_metric": True,
                "metadata": metadata or {},
            },
        )

    def log_memory_usage(
        self,
        operation: str,
        memory_mb: float,
        peak_memory_mb: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log memory usage for an operation."""
        extra_data = {
            "operation": operation,
            "memory_mb": memory_mb,
            "memory_metric": True,
            "metadata": metadata or {},
        }

        if peak_memory_mb is not None:
            extra_data["peak_memory_mb"] = peak_memory_mb

        self.logger.info(
            f"Memory usage for {operation}: {memory_mb:.2f} MB", extra=extra_data
        )

    def log_database_query(
        self,
        query: str,
        duration_ms: float,
        rows_affected: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log database query performance."""
        extra_data = {
            "query": query[:200] + "..."
            if len(query) > 200
            else query,  # Truncate long queries
            "duration_ms": duration_ms,
            "database_metric": True,
            "metadata": metadata or {},
        }

        if rows_affected is not None:
            extra_data["rows_affected"] = rows_affected

        self.logger.info(
            f"Database query completed in {duration_ms:.2f}ms", extra=extra_data
        )


class ErrorTracker:
    """System for tracking and categorizing errors automatically."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize error tracker."""
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
        self.logger = logging.getLogger("error_tracker")

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    def track_error(
        self,
        error: Exception,
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ):
        """Track an error with automatic categorization."""
        try:
            error_data = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
                "metadata": metadata or {},
                "user_id": user_id,
                "category": self._categorize_error(error),
                "severity": self._determine_severity(error),
            }

            # Generate error ID
            error_id = f"error_{int(datetime.now(timezone.utc).timestamp())}_{hash(str(error))}"
            error_data["error_id"] = error_id

            # Store error in Redis
            self.redis_client.setex(
                f"errors:{error_id}",
                7 * 24 * 3600,  # 7 days TTL
                json.dumps(error_data, default=str),
            )

            # Add to error index
            self.redis_client.zadd(
                "errors_index",
                {f"errors:{error_id}": datetime.now(timezone.utc).timestamp()},
            )

            # Add to category index
            self.redis_client.zadd(
                f"errors_category:{error_data['category']}",
                {f"errors:{error_id}": datetime.now(timezone.utc).timestamp()},
            )

            # Increment error counter
            self.redis_client.incr(f"error_count:{error_data['category']}")
            self.redis_client.expire(f"error_count:{error_data['category']}", 24 * 3600)

            # Log the error
            self.logger.error(
                f"Error tracked: {error_data['error_type']} in {context}",
                extra={
                    "error_id": error_id,
                    "error_category": error_data["category"],
                    "error_severity": error_data["severity"],
                    "context": context,
                    "user_id": user_id,
                    "metadata": metadata,
                },
                exc_info=True,
            )

            return error_id

        except Exception as e:
            # Don't let error tracking break the application
            self.logger.error(f"Failed to track error: {e}")
            return None

    def _categorize_error(self, error: Exception) -> str:
        """Automatically categorize error based on type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()

        # Database errors
        if any(
            keyword in error_type.lower()
            for keyword in ["sql", "database", "connection"]
        ):
            return "database"

        # Network/API errors
        if any(
            keyword in error_type.lower()
            for keyword in ["http", "connection", "timeout", "network"]
        ):
            return "network"

        # Authentication/Authorization errors
        if any(
            keyword in error_message
            for keyword in ["unauthorized", "forbidden", "authentication", "permission"]
        ):
            return "auth"

        # Validation errors
        if any(
            keyword in error_type.lower() for keyword in ["validation", "value", "type"]
        ):
            return "validation"

        # File/IO errors
        if any(
            keyword in error_type.lower() for keyword in ["file", "io", "permission"]
        ):
            return "io"

        # Memory errors
        if any(keyword in error_type.lower() for keyword in ["memory", "overflow"]):
            return "memory"

        # Default category
        return "application"

    def _determine_severity(self, error: Exception) -> str:
        """Determine error severity based on type."""
        error_type = type(error).__name__

        # Critical errors that might crash the application
        critical_errors = ["MemoryError", "SystemExit", "KeyboardInterrupt"]
        if error_type in critical_errors:
            return "critical"

        # High severity errors
        high_severity_errors = ["DatabaseError", "ConnectionError", "TimeoutError"]
        if error_type in high_severity_errors:
            return "high"

        # Medium severity errors
        medium_severity_errors = ["ValueError", "TypeError", "AttributeError"]
        if error_type in medium_severity_errors:
            return "medium"

        # Default to low severity
        return "low"

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period."""
        try:
            import time

            start_time = time.time() - (hours * 3600)
            end_time = time.time()

            # Get errors from the time period
            error_keys = self.redis_client.zrangebyscore(
                "errors_index", start_time, end_time
            )

            # Collect statistics
            stats = {
                "total_errors": len(error_keys),
                "by_category": {},
                "by_severity": {},
                "by_hour": {},
                "top_errors": [],
            }

            error_counts = {}

            for key in error_keys:
                error_data_str = self.redis_client.get(key)
                if error_data_str:
                    try:
                        error_data = json.loads(error_data_str)

                        # Count by category
                        category = error_data.get("category", "unknown")
                        stats["by_category"][category] = (
                            stats["by_category"].get(category, 0) + 1
                        )

                        # Count by severity
                        severity = error_data.get("severity", "unknown")
                        stats["by_severity"][severity] = (
                            stats["by_severity"].get(severity, 0) + 1
                        )

                        # Count by error type
                        error_type = error_data.get("error_type", "unknown")
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1

                    except json.JSONDecodeError:
                        continue

            # Get top error types
            stats["top_errors"] = sorted(
                error_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get error statistics: {e}")
            return {"error": str(e)}


def setup_logging(
    log_level: str = "INFO",
    enable_redis_logging: bool = True,
    redis_url: str = "redis://localhost:6379/0",
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Setup centralized logging configuration."""

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Base logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": StructuredFormatter,
                "include_extra": True,
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level.upper(),
                "formatter": "structured",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": log_level.upper(),
            "handlers": ["console"],
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "level": "WARNING",  # Reduce SQL query noise
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }

    # Add Redis handler if enabled
    if enable_redis_logging:
        config["handlers"]["redis"] = {
            "()": RedisLogHandler,
            "redis_url": redis_url,
            "level": "WARNING",  # Only send warnings and errors to Redis
            "formatter": "structured",
        }
        config["root"]["handlers"].append("redis")

    # Add file handler if specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "level": log_level.upper(),
            "formatter": "structured",
        }
        config["root"]["handlers"].append("file")

    # Apply configuration
    logging.config.dictConfig(config)

    return config


# Global instances
performance_logger = PerformanceLogger()
error_tracker = ErrorTracker()


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    return performance_logger


def get_error_tracker() -> ErrorTracker:
    """Get error tracker instance."""
    return error_tracker
