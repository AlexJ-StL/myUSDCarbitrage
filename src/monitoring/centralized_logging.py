"""Centralized logging system with structured format and error tracking."""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import redis

from .logging_config import (
    setup_logging,
    get_error_tracker,
    ErrorTracker,
    PerformanceLogger,
    get_performance_logger,
)
from .log_aggregation import LogAggregator, get_log_aggregator
from .log_analysis import LogAnalyzer, get_log_analyzer
from .performance_profiler import (
    PerformanceProfiler,
    get_performance_profiler,
    profile_function,
    profile_block,
    profile_async_block,
)

# Configure module logger
logger = logging.getLogger(__name__)


class CentralizedLogging:
    """Centralized logging system that integrates all logging components."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ):
        """Initialize centralized logging system."""
        self.redis_url = redis_url
        self.log_level = log_level
        self.log_file = log_file
        self._initialized = False

        # Component instances
        self.error_tracker: Optional[ErrorTracker] = None
        self.log_aggregator: Optional[LogAggregator] = None
        self.log_analyzer: Optional[LogAnalyzer] = None
        self.performance_profiler: Optional[PerformanceProfiler] = None
        self.performance_logger: Optional[PerformanceLogger] = None

    def initialize(self) -> Dict[str, Any]:
        """Initialize all logging components."""
        if self._initialized:
            return {"status": "already_initialized"}

        # Setup basic logging configuration
        config = setup_logging(
            log_level=self.log_level,
            enable_redis_logging=True,
            redis_url=self.redis_url,
            log_file=self.log_file,
        )

        # Initialize components
        self.error_tracker = get_error_tracker()
        self.log_aggregator = LogAggregator(redis_url=self.redis_url)
        self.log_analyzer = LogAnalyzer(
            redis_url=self.redis_url, log_aggregator=self.log_aggregator
        )
        self.performance_profiler = get_performance_profiler()
        self.performance_logger = get_performance_logger()

        self._initialized = True

        logger.info(
            "Centralized logging system initialized",
            extra={
                "component": "centralized_logging",
                "log_level": self.log_level,
                "redis_url": self.redis_url.split("@")[-1],  # Hide credentials
                "log_file": self.log_file,
            },
        )

        return {
            "status": "initialized",
            "log_level": self.log_level,
            "log_file": self.log_file,
            "components": [
                "error_tracker",
                "log_aggregator",
                "log_analyzer",
                "performance_profiler",
                "performance_logger",
            ],
        }

    def track_error(
        self,
        error: Exception,
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Track an error with automatic categorization."""
        if not self._initialized:
            self.initialize()

        return self.error_tracker.track_error(
            error=error, context=context, metadata=metadata, user_id=user_id
        )

    def log_execution_time(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log execution time for an operation."""
        if not self._initialized:
            self.initialize()

        self.performance_logger.log_execution_time(
            operation=operation, duration_ms=duration_ms, metadata=metadata
        )

    def log_memory_usage(
        self,
        operation: str,
        memory_mb: float,
        peak_memory_mb: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log memory usage for an operation."""
        if not self._initialized:
            self.initialize()

        self.performance_logger.log_memory_usage(
            operation=operation,
            memory_mb=memory_mb,
            peak_memory_mb=peak_memory_mb,
            metadata=metadata,
        )

    def log_database_query(
        self,
        query: str,
        duration_ms: float,
        rows_affected: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log database query performance."""
        if not self._initialized:
            self.initialize()

        self.performance_logger.log_database_query(
            query=query,
            duration_ms=duration_ms,
            rows_affected=rows_affected,
            metadata=metadata,
        )

    def profile_function(
        self,
        operation_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Decorator for profiling function execution."""
        if not self._initialized:
            self.initialize()

        return profile_function(operation_name=operation_name, metadata=metadata)

    async def search_logs(
        self,
        query: Optional[str] = None,
        log_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Search logs with various filters."""
        if not self._initialized:
            self.initialize()

        return await self.log_aggregator.search_logs(
            query=query,
            log_level=log_level,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )

    async def get_log_statistics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get log statistics for the specified time period."""
        if not self._initialized:
            self.initialize()

        return await self.log_aggregator.get_log_statistics(
            start_time=start_time, end_time=end_time
        )

    async def analyze_bottlenecks(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance data to identify bottlenecks."""
        if not self._initialized:
            self.initialize()

        return await self.performance_profiler.analyze_bottlenecks(hours=hours)

    async def analyze_error_patterns(
        self, hours: int = 24, min_occurrences: int = 3
    ) -> Dict[str, Any]:
        """Analyze error logs to identify patterns and recurring issues."""
        if not self._initialized:
            self.initialize()

        return await self.log_analyzer.analyze_error_patterns(
            hours=hours, min_occurrences=min_occurrences
        )

    async def generate_log_insights(self, days: int = 7) -> Dict[str, Any]:
        """Generate insights from log data."""
        if not self._initialized:
            self.initialize()

        return await self.log_analyzer.generate_log_insights(days=days)

    async def export_logs(
        self,
        format_type: str = "json",
        query: Optional[str] = None,
        log_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> Union[str, bytes]:
        """Export logs in specified format."""
        if not self._initialized:
            self.initialize()

        return await self.log_aggregator.export_logs(
            format_type=format_type,
            query=query,
            log_level=log_level,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    async def cleanup_old_logs(self, retention_days: int = 7) -> int:
        """Clean up logs older than retention period."""
        if not self._initialized:
            self.initialize()

        return await self.log_aggregator.cleanup_old_logs(retention_days=retention_days)

    async def get_logging_dashboard(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive logging dashboard data."""
        if not self._initialized:
            self.initialize()

        try:
            # Collect all dashboard data concurrently
            log_stats, error_stats, insights, bottlenecks = await asyncio.gather(
                self.log_aggregator.get_log_statistics(
                    start_time=datetime.now(timezone.utc) - timedelta(hours=hours),
                    end_time=datetime.now(timezone.utc),
                ),
                self.error_tracker.get_error_statistics(hours=hours),
                self.log_analyzer.generate_log_insights(days=hours // 24 or 1),
                self.performance_profiler.analyze_bottlenecks(hours=hours),
                return_exceptions=True,
            )

            # Handle any exceptions
            dashboard_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "period_hours": hours,
                "log_statistics": log_stats
                if not isinstance(log_stats, Exception)
                else {"error": str(log_stats)},
                "error_statistics": error_stats
                if not isinstance(error_stats, Exception)
                else {"error": str(error_stats)},
                "insights": insights.get("insights", [])
                if not isinstance(insights, Exception)
                else [],
                "performance_bottlenecks": bottlenecks.get("bottlenecks", {})
                if not isinstance(bottlenecks, Exception)
                else {},
            }

            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to get logging dashboard: {e}")
            return {"error": str(e)}


# Global centralized logging instance
centralized_logging = CentralizedLogging()


def get_centralized_logging() -> CentralizedLogging:
    """Get centralized logging instance."""
    return centralized_logging


def initialize_logging(
    log_level: str = "INFO",
    redis_url: str = "redis://localhost:6379/0",
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize centralized logging system."""
    global centralized_logging
    centralized_logging = CentralizedLogging(
        redis_url=redis_url, log_level=log_level, log_file=log_file
    )
    return centralized_logging.initialize()


# Convenience functions
def track_error(
    error: Exception,
    context: str,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> Optional[str]:
    """Track an error with automatic categorization."""
    return centralized_logging.track_error(
        error=error, context=context, metadata=metadata, user_id=user_id
    )


def log_execution_time(
    operation: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None
):
    """Log execution time for an operation."""
    centralized_logging.log_execution_time(
        operation=operation, duration_ms=duration_ms, metadata=metadata
    )


def log_memory_usage(
    operation: str,
    memory_mb: float,
    peak_memory_mb: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Log memory usage for an operation."""
    centralized_logging.log_memory_usage(
        operation=operation,
        memory_mb=memory_mb,
        peak_memory_mb=peak_memory_mb,
        metadata=metadata,
    )


def log_database_query(
    query: str,
    duration_ms: float,
    rows_affected: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Log database query performance."""
    centralized_logging.log_database_query(
        query=query,
        duration_ms=duration_ms,
        rows_affected=rows_affected,
        metadata=metadata,
    )


def profile(
    operation_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
):
    """Decorator for profiling function execution."""
    return centralized_logging.profile_function(
        operation_name=operation_name, metadata=metadata
    )


# Exception handler for FastAPI
def create_exception_handler(app):
    """Create exception handler for FastAPI."""

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler that tracks errors."""
        from fastapi import Request
        from fastapi.responses import JSONResponse

        # Track error
        error_id = track_error(
            error=exc,
            context=f"{request.method} {request.url.path}",
            metadata={
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_host": request.client.host if request.client else None,
            },
        )

        # Log error
        logger.error(
            f"Unhandled exception in {request.method} {request.url.path}",
            exc_info=exc,
            extra={
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
            },
        )

        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later.",
                "error_id": error_id,
                "type": "internal_error",
            },
        )

    return global_exception_handler


# Database query logging
class DatabaseQueryLogger:
    """Logger for database queries."""

    @staticmethod
    def before_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ):
        """SQLAlchemy event hook for before cursor execute."""
        conn.info.setdefault("query_start_time", time.time())

    @staticmethod
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        """SQLAlchemy event hook for after cursor execute."""
        total_time = time.time() - conn.info.get("query_start_time", time.time())

        # Log query execution time
        log_database_query(
            query=statement,
            duration_ms=total_time * 1000,
            rows_affected=cursor.rowcount if hasattr(cursor, "rowcount") else None,
            metadata={
                "executemany": executemany,
                "parameters": str(parameters)[:200] if parameters else None,
            },
        )


def setup_sqlalchemy_query_logging(engine):
    """Setup SQLAlchemy query logging."""
    from sqlalchemy import event

    event.listen(
        engine, "before_cursor_execute", DatabaseQueryLogger.before_cursor_execute
    )
    event.listen(
        engine, "after_cursor_execute", DatabaseQueryLogger.after_cursor_execute
    )
