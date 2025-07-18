"""Error categorization and analysis system."""

import asyncio
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import redis

logger = logging.getLogger(__name__)


class ErrorCategorizer:
    """System for categorizing and analyzing errors."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize error categorizer."""
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
        self.error_categories = {
            "database": [
                "database",
                "sql",
                "connection",
                "postgres",
                "redis",
                "timescale",
                "deadlock",
                "timeout",
                "constraint",
                "foreign key",
                "unique violation",
            ],
            "network": [
                "network",
                "http",
                "connection",
                "timeout",
                "socket",
                "api",
                "request",
                "response",
                "url",
                "dns",
                "ssl",
                "tls",
            ],
            "authentication": [
                "auth",
                "unauthorized",
                "forbidden",
                "permission",
                "access denied",
                "token",
                "jwt",
                "login",
                "password",
                "credential",
            ],
            "validation": [
                "validation",
                "invalid",
                "schema",
                "format",
                "required",
                "missing",
                "constraint",
                "type error",
                "value error",
            ],
            "data_quality": [
                "data quality",
                "corrupt",
                "invalid data",
                "missing data",
                "gap",
                "anomaly",
                "outlier",
                "inconsistent",
            ],
            "system_resource": [
                "memory",
                "cpu",
                "disk",
                "resource",
                "capacity",
                "overflow",
                "out of memory",
                "leak",
                "full",
                "quota",
            ],
            "configuration": [
                "config",
                "setting",
                "environment",
                "variable",
                "parameter",
                "option",
                "property",
                "missing config",
            ],
            "external_service": [
                "external",
                "service",
                "api",
                "third party",
                "dependency",
                "integration",
                "exchange",
                "provider",
            ],
        }

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    def categorize_error(self, error_type: str, error_message: str) -> str:
        """Categorize error based on type and message."""
        error_text = f"{error_type} {error_message}".lower()

        # Check each category for matching keywords
        for category, keywords in self.error_categories.items():
            for keyword in keywords:
                if keyword.lower() in error_text:
                    return category

        # Default category
        return "application"

    def determine_severity(self, error_type: str, error_message: str) -> str:
        """Determine error severity based on type and message."""
        error_text = f"{error_type} {error_message}".lower()

        # Critical errors
        critical_patterns = [
            r"out\s+of\s+memory",
            r"disk\s+full",
            r"cannot\s+allocate",
            r"deadlock",
            r"system\s+crash",
            r"fatal",
            r"emergency",
            r"data\s+corruption",
            r"security\s+breach",
        ]
        for pattern in critical_patterns:
            if re.search(pattern, error_text):
                return "critical"

        # High severity errors
        high_severity_types = [
            "DatabaseError",
            "ConnectionError",
            "TimeoutError",
            "AuthenticationError",
            "SecurityError",
            "MemoryError",
            "SystemError",
            "IOError",
            "RuntimeError",
        ]
        high_severity_patterns = [
            r"connection\s+refused",
            r"cannot\s+connect",
            r"timeout",
            r"unauthorized",
            r"forbidden",
            r"permission\s+denied",
            r"not\s+found",
            r"unavailable",
        ]

        if error_type in high_severity_types:
            return "high"

        for pattern in high_severity_patterns:
            if re.search(pattern, error_text):
                return "high"

        # Medium severity errors
        medium_severity_types = [
            "ValueError",
            "TypeError",
            "AttributeError",
            "KeyError",
            "IndexError",
            "ValidationError",
            "AssertionError",
        ]
        if error_type in medium_severity_types:
            return "medium"

        # Default to low severity
        return "low"

    async def analyze_error_patterns(
        self, hours: int = 24, min_occurrences: int = 3
    ) -> Dict[str, Any]:
        """Analyze error patterns to identify recurring issues."""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            start_timestamp = start_time.timestamp()
            end_timestamp = datetime.now(timezone.utc).timestamp()

            # Get error keys from the time period
            error_keys = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zrangebyscore(
                    "errors_index", start_timestamp, end_timestamp
                ),
            )

            if not error_keys:
                return {
                    "patterns_found": 0,
                    "message": "No errors found in the specified time period",
                }

            # Collect error data
            errors = []
            for key in error_keys:
                error_data_str = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                if error_data_str:
                    try:
                        error_data = json.loads(error_data_str)
                        errors.append(error_data)
                    except json.JSONDecodeError:
                        continue

            # Analyze patterns
            error_types = Counter([e.get("error_type", "unknown") for e in errors])
            error_categories = Counter([e.get("category", "unknown") for e in errors])
            error_contexts = Counter([e.get("context", "unknown") for e in errors])

            # Find common error messages (simplified for similar messages)
            simplified_messages = []
            for error in errors:
                message = error.get("error_message", "")
                # Replace specific values with placeholders
                simplified = re.sub(r"\d+", "<NUM>", message)
                simplified = re.sub(r"\'[^\']+\'", "<STR>", simplified)
                simplified = re.sub(r"\"[^\"]+\"", "<STR>", simplified)
                simplified_messages.append(simplified)

            common_messages = Counter(simplified_messages)

            # Identify recurring patterns
            recurring_patterns = []
            for message, count in common_messages.most_common(10):
                if count >= min_occurrences:
                    # Find examples of this pattern
                    examples = []
                    for error in errors:
                        orig_message = error.get("error_message", "")
                        simp_message = re.sub(r"\d+", "<NUM>", orig_message)
                        simp_message = re.sub(r"\'[^\']+\'", "<STR>", simp_message)
                        simp_message = re.sub(r"\"[^\"]+\"", "<STR>", simp_message)

                        if simp_message == message:
                            examples.append({
                                "error_id": error.get("error_id", ""),
                                "timestamp": error.get("timestamp", ""),
                                "context": error.get("context", ""),
                                "original_message": orig_message,
                            })
                            if len(examples) >= 3:  # Limit to 3 examples
                                break

                    recurring_patterns.append({
                        "pattern": message,
                        "count": count,
                        "examples": examples,
                    })

            # Identify error hotspots (contexts with many errors)
            hotspots = []
            for context, count in error_contexts.most_common(5):
                if count >= min_occurrences:
                    context_errors = [e for e in errors if e.get("context") == context]
                    categories = Counter([
                        e.get("category", "unknown") for e in context_errors
                    ])
                    severities = Counter([
                        e.get("severity", "unknown") for e in context_errors
                    ])

                    hotspots.append({
                        "context": context,
                        "error_count": count,
                        "categories": dict(categories),
                        "severities": dict(severities),
                    })

            return {
                "analysis_period_hours": hours,
                "total_errors": len(errors),
                "patterns_found": len(recurring_patterns),
                "recurring_patterns": recurring_patterns,
                "error_hotspots": hotspots,
                "top_error_types": dict(error_types.most_common(10)),
                "error_categories": dict(error_categories),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to analyze error patterns: {e}")
            return {"error": str(e)}

    async def find_correlated_errors(
        self, error_id: str, time_window_minutes: int = 5
    ) -> Dict[str, Any]:
        """Find errors that occurred close in time to the specified error."""
        try:
            # Get the specified error
            error_key = f"errors:{error_id}"
            error_data_str = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, error_key
            )

            if not error_data_str:
                return {"error": f"Error with ID {error_id} not found"}

            error_data = json.loads(error_data_str)
            error_timestamp = datetime.fromisoformat(
                error_data["timestamp"].replace("Z", "+00:00")
            )

            # Define time window
            start_time = error_timestamp - timedelta(minutes=time_window_minutes)
            end_time = error_timestamp + timedelta(minutes=time_window_minutes)

            start_timestamp = start_time.timestamp()
            end_timestamp = end_time.timestamp()

            # Get errors in the time window
            error_keys = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zrangebyscore(
                    "errors_index", start_timestamp, end_timestamp
                ),
            )

            # Collect correlated errors
            correlated_errors = []
            for key in error_keys:
                if key == error_key:  # Skip the original error
                    continue

                corr_error_data_str = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )

                if corr_error_data_str:
                    try:
                        corr_error_data = json.loads(corr_error_data_str)
                        correlated_errors.append({
                            "error_id": corr_error_data.get("error_id", ""),
                            "error_type": corr_error_data.get("error_type", ""),
                            "error_message": corr_error_data.get("error_message", ""),
                            "context": corr_error_data.get("context", ""),
                            "category": corr_error_data.get("category", ""),
                            "severity": corr_error_data.get("severity", ""),
                            "timestamp": corr_error_data.get("timestamp", ""),
                            "time_difference_seconds": abs(
                                (
                                    datetime.fromisoformat(
                                        corr_error_data["timestamp"].replace(
                                            "Z", "+00:00"
                                        )
                                    )
                                    - error_timestamp
                                ).total_seconds()
                            ),
                        })
                    except (json.JSONDecodeError, KeyError):
                        continue

            # Sort by time difference
            correlated_errors.sort(key=lambda x: x["time_difference_seconds"])

            return {
                "original_error": {
                    "error_id": error_data.get("error_id", ""),
                    "error_type": error_data.get("error_type", ""),
                    "error_message": error_data.get("error_message", ""),
                    "context": error_data.get("context", ""),
                    "timestamp": error_data.get("timestamp", ""),
                },
                "time_window_minutes": time_window_minutes,
                "correlated_errors_count": len(correlated_errors),
                "correlated_errors": correlated_errors,
            }

        except Exception as e:
            logger.error(f"Failed to find correlated errors: {e}")
            return {"error": str(e)}


# Global error categorizer instance
error_categorizer = ErrorCategorizer()


def get_error_categorizer() -> ErrorCategorizer:
    """Get error categorizer instance."""
    return error_categorizer
