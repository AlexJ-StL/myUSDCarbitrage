"""Centralized logging system with structured format and aggregation capabilities."""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .logging_config import (
    setup_logging,
    get_error_tracker,
    get_performance_logger,
    ErrorTracker,
    PerformanceLogger,
)
from .log_aggregation import LogAggregator, get_log_aggregator
from .performance_profiler import get_performance_profiler, PerformanceProfiler


class CentralizedLoggingSystem:
    """Centralized logging system that integrates all logging components."""

    def __init__(
        self,
        log_level: str = "INFO",
        enable_redis_logging: bool = True,
        redis_url: str = "redis://localhost:6379/0",
        log_file: Optional[str] = None,
    ):
        """Initialize centralized logging system."""
        self.log_level = log_level
        self.enable_redis_logging = enable_redis_logging
        self.redis_url = redis_url
        self.log_file = log_file
        self.config = None
        self.error_tracker = None
        self.log_aggregator = None
        self.performance_profiler = None
        self.performance_logger = None
        self.initialized = False

    def initialize(self):
        """Initialize all logging components."""
        if self.initialized:
            return

        # Setup basic logging configuration
        self.config = setup_logging(
            log_level=self.log_level,
            enable_redis_logging=self.enable_redis_logging,
            redis_url=self.redis_url,
            log_file=self.log_file,
        )

        # Initialize components
        self.error_tracker = get_error_tracker()
        self.performance_logger = get_performance_logger()
        self.log_aggregator = LogAggregator(redis_url=self.redis_url)
        self.performance_profiler = get_performance_profiler()

        self.initialized = True
        logging.info(
            "Centralized logging system initialized",
            extra={"component": "logging_system", "action": "initialize"},
        )

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        if not self.initialized:
            self.initialize()
        return logging.getLogger(name)

    def track_error(
        self,
        error: Exception,
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Track an error with automatic categorization."""
        if not self.initialized:
            self.initialize()
        return self.error_tracker.track_error(
            error=error, context=context, metadata=metadata, user_id=user_id
        )

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
        if not self.initialized:
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
        if not self.initialized:
            self.initialize()
        return await self.log_aggregator.get_log_statistics(
            start_time=start_time, end_time=end_time
        )

    async def analyze_bottlenecks(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance data to identify bottlenecks."""
        if not self.initialized:
            self.initialize()
        return await self.performance_profiler.analyze_bottlenecks(hours=hours)

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
        if not self.initialized:
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
        if not self.initialized:
            self.initialize()
        return await self.log_aggregator.cleanup_old_logs(retention_days=retention_days)


# Global centralized logging system instance
centralized_logging = CentralizedLoggingSystem()


def initialize_logging(
    log_level: str = "INFO",
    enable_redis_logging: bool = True,
    redis_url: str = "redis://localhost:6379/0",
    log_file: Optional[str] = None,
):
    """Initialize the centralized logging system."""
    global centralized_logging
    centralized_logging = CentralizedLoggingSystem(
        log_level=log_level,
        enable_redis_logging=enable_redis_logging,
        redis_url=redis_url,
        log_file=log_file,
    )
    centralized_logging.initialize()
    return centralized_logging


def get_centralized_logging() -> CentralizedLoggingSystem:
    """Get the centralized logging system instance."""
    if not centralized_logging.initialized:
        centralized_logging.initialize()
    return centralized_logging


def get_application_logger(name: str) -> logging.Logger:
    """Get an application logger with the specified name."""
    return centralized_logging.get_logger(name)
