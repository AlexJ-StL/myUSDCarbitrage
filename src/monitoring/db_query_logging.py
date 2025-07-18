"""Database query logging middleware and utilities."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .centralized_logging import log_database_query

# Configure module logger
logger = logging.getLogger(__name__)


class QueryLoggingMiddleware:
    """Middleware for logging database queries."""

    def __init__(self, engine: Engine):
        """Initialize query logging middleware."""
        self.engine = engine
        self._setup_event_listeners()

    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners."""
        event.listen(self.engine, "before_cursor_execute", self._before_cursor_execute)
        event.listen(self.engine, "after_cursor_execute", self._after_cursor_execute)

    def _before_cursor_execute(
        self, conn, cursor, statement, parameters, context, executemany
    ):
        """SQLAlchemy event hook for before cursor execute."""
        conn.info.setdefault("query_start_time", time.time())
        conn.info.setdefault("query_count", 0)
        conn.info["query_count"] += 1

    def _after_cursor_execute(
        self, conn, cursor, statement, parameters, context, executemany
    ):
        """SQLAlchemy event hook for after cursor execute."""
        total_time = time.time() - conn.info.get("query_start_time", time.time())
        duration_ms = total_time * 1000

        # Log query execution time
        log_database_query(
            query=statement,
            duration_ms=duration_ms,
            rows_affected=cursor.rowcount if hasattr(cursor, "rowcount") else None,
            metadata={
                "executemany": executemany,
                "parameters": str(parameters)[:200] if parameters else None,
                "query_count": conn.info.get("query_count", 1),
            },
        )

        # Log slow queries
        if duration_ms > 1000:  # Queries taking more than 1 second
            logger.warning(
                f"Slow query detected: {duration_ms:.2f}ms",
                extra={
                    "query": statement[:500] + "..."
                    if len(statement) > 500
                    else statement,
                    "duration_ms": duration_ms,
                    "rows_affected": cursor.rowcount
                    if hasattr(cursor, "rowcount")
                    else None,
                    "executemany": executemany,
                },
            )


def log_query(func: Callable) -> Callable:
    """Decorator for logging database queries in custom functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            # Extract query information if possible
            query_info = "Custom database operation"
            rows_affected = None

            # Try to extract query information from result
            if hasattr(result, "statement"):
                query_info = str(result.statement)
            if hasattr(result, "rowcount"):
                rows_affected = result.rowcount

            # Log query
            log_database_query(
                query=query_info,
                duration_ms=duration_ms,
                rows_affected=rows_affected,
                metadata={
                    "function": func.__name__,
                    "module": func.__module__,
                },
            )

            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_database_query(
                query=f"Failed operation: {func.__name__}",
                duration_ms=duration_ms,
                metadata={
                    "function": func.__name__,
                    "module": func.__module__,
                    "error": str(e),
                },
            )
            raise

    return wrapper


def setup_query_logging(engine: Engine) -> QueryLoggingMiddleware:
    """Setup query logging for a SQLAlchemy engine."""
    return QueryLoggingMiddleware(engine)


class SessionWithQueryLogging(Session):
    """SQLAlchemy Session subclass with query logging."""

    def __init__(self, *args, **kwargs):
        """Initialize session with query logging."""
        super().__init__(*args, **kwargs)
        self.query_count = 0
        self.query_time = 0.0

    def execute(self, *args, **kwargs):
        """Execute a query with logging."""
        start_time = time.time()
        try:
            result = super().execute(*args, **kwargs)
            duration = time.time() - start_time
            self.query_count += 1
            self.query_time += duration

            # Extract query information
            statement = args[0] if args else "Unknown query"
            if hasattr(statement, "compile"):
                statement = str(statement.compile(dialect=self.bind.dialect))

            # Log query
            log_database_query(
                query=statement[:500] + "..."
                if len(str(statement)) > 500
                else str(statement),
                duration_ms=duration * 1000,
                rows_affected=result.rowcount if hasattr(result, "rowcount") else None,
                metadata={
                    "query_count": self.query_count,
                    "total_query_time": self.query_time,
                },
            )

            return result
        except Exception as e:
            duration = time.time() - start_time
            log_database_query(
                query="Failed query execution",
                duration_ms=duration * 1000,
                metadata={
                    "error": str(e),
                    "query_count": self.query_count,
                },
            )
            raise
