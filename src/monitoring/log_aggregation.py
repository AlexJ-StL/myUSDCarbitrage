"""Log aggregation and search capabilities."""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union

import redis

logger = logging.getLogger(__name__)


class LogAggregator:
    """Aggregates and provides search capabilities for logs."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize log aggregator."""
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

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
        try:
            # Set default time range if not provided
            if end_time is None:
                end_time = datetime.now(timezone.utc)
            if start_time is None:
                start_time = end_time - timedelta(hours=24)

            # Convert to timestamps
            start_timestamp = start_time.timestamp()
            end_timestamp = end_time.timestamp()

            # Determine which log levels to search
            levels_to_search = []
            if log_level:
                levels_to_search = [log_level.lower()]
            else:
                levels_to_search = ["debug", "info", "warning", "error", "critical"]

            # Collect log keys from all relevant levels
            all_log_keys = []
            for level in levels_to_search:
                keys = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.redis_client.zrangebyscore(
                        f"logs_index:{level}",
                        start_timestamp,
                        end_timestamp,
                        start=0,
                        num=-1,  # Get all matching keys
                        desc=True,  # Most recent first
                    ),
                )
                all_log_keys.extend(keys)

            # Sort by timestamp (most recent first)
            all_log_keys.sort(key=lambda x: float(x.split(":")[-1]), reverse=True)

            # Apply pagination
            paginated_keys = all_log_keys[offset : offset + limit]

            # Retrieve log data
            logs = []
            for key in paginated_keys:
                log_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                if log_data:
                    try:
                        log_entry = json.loads(log_data)

                        # Apply text search filter if provided
                        if query and not self._matches_query(log_entry, query):
                            continue

                        logs.append(log_entry)

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode log data for key: {key}")

            return {
                "logs": logs,
                "total_found": len(logs),
                "total_available": len(all_log_keys),
                "query": query,
                "log_level": log_level,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "limit": limit,
                "offset": offset,
            }

        except Exception as e:
            logger.error(f"Failed to search logs: {e}")
            return {
                "logs": [],
                "error": str(e),
                "total_found": 0,
                "total_available": 0,
            }

    def _matches_query(self, log_entry: Dict[str, Any], query: str) -> bool:
        """Check if log entry matches the search query."""
        query_lower = query.lower()

        # Search in message
        if query_lower in log_entry.get("message", "").lower():
            return True

        # Search in logger name
        if query_lower in log_entry.get("logger", "").lower():
            return True

        # Search in module name
        if query_lower in log_entry.get("module", "").lower():
            return True

        # Search in exception information
        exception_info = log_entry.get("exception", {})
        if isinstance(exception_info, dict):
            if query_lower in str(exception_info.get("message", "")).lower():
                return True
            if query_lower in str(exception_info.get("type", "")).lower():
                return True

        # Search in extra fields
        extra_info = log_entry.get("extra", {})
        if isinstance(extra_info, dict):
            for key, value in extra_info.items():
                if query_lower in str(value).lower():
                    return True

        return False

    async def get_log_statistics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get log statistics for the specified time period."""
        try:
            # Set default time range
            if end_time is None:
                end_time = datetime.now(timezone.utc)
            if start_time is None:
                start_time = end_time - timedelta(hours=24)

            start_timestamp = start_time.timestamp()
            end_timestamp = end_time.timestamp()

            # Count logs by level
            level_counts = {}
            log_levels = ["debug", "info", "warning", "error", "critical"]

            for level in log_levels:
                count = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda l=level: self.redis_client.zcount(
                        f"logs_index:{l}", start_timestamp, end_timestamp
                    ),
                )
                level_counts[level] = count

            # Get hourly distribution
            hourly_counts = {}
            current_time = start_time
            while current_time < end_time:
                hour_start = current_time.timestamp()
                hour_end = (current_time + timedelta(hours=1)).timestamp()

                hour_total = 0
                for level in log_levels:
                    hour_count = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda l=level: self.redis_client.zcount(
                            f"logs_index:{l}", hour_start, hour_end
                        ),
                    )
                    hour_total += hour_count

                hourly_counts[current_time.strftime("%Y-%m-%d %H:00")] = hour_total
                current_time += timedelta(hours=1)

            # Get top loggers
            top_loggers = await self._get_top_loggers(start_timestamp, end_timestamp)

            # Get recent errors
            recent_errors = await self._get_recent_errors(
                start_timestamp, end_timestamp
            )

            return {
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "total_logs": sum(level_counts.values()),
                "by_level": level_counts,
                "hourly_distribution": hourly_counts,
                "top_loggers": top_loggers,
                "recent_errors": recent_errors,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get log statistics: {e}")
            return {"error": str(e)}

    async def _get_top_loggers(
        self, start_timestamp: float, end_timestamp: float, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top loggers by message count."""
        try:
            logger_counts = {}

            # Get logs from all levels
            for level in ["debug", "info", "warning", "error", "critical"]:
                keys = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.redis_client.zrangebyscore(
                        f"logs_index:{level}", start_timestamp, end_timestamp
                    ),
                )

                for key in keys:
                    log_data = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.get, key
                    )
                    if log_data:
                        try:
                            log_entry = json.loads(log_data)
                            logger_name = log_entry.get("logger", "unknown")
                            logger_counts[logger_name] = (
                                logger_counts.get(logger_name, 0) + 1
                            )
                        except json.JSONDecodeError:
                            continue

            # Sort and return top loggers
            top_loggers = sorted(
                logger_counts.items(), key=lambda x: x[1], reverse=True
            )[:limit]

            return [
                {"logger": logger_name, "count": count}
                for logger_name, count in top_loggers
            ]

        except Exception as e:
            logger.error(f"Failed to get top loggers: {e}")
            return []

    async def _get_recent_errors(
        self, start_timestamp: float, end_timestamp: float, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent error logs."""
        try:
            error_keys = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zrangebyscore(
                    "logs_index:error",
                    start_timestamp,
                    end_timestamp,
                    start=0,
                    num=limit,
                    desc=True,
                ),
            )

            recent_errors = []
            for key in error_keys:
                log_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                if log_data:
                    try:
                        log_entry = json.loads(log_data)
                        recent_errors.append({
                            "timestamp": log_entry.get("timestamp"),
                            "logger": log_entry.get("logger"),
                            "message": log_entry.get("message"),
                            "module": log_entry.get("module"),
                            "function": log_entry.get("function"),
                            "exception": log_entry.get("exception", {}).get("type"),
                        })
                    except json.JSONDecodeError:
                        continue

            return recent_errors

        except Exception as e:
            logger.error(f"Failed to get recent errors: {e}")
            return []

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
        try:
            # Get logs using search functionality
            search_result = await self.search_logs(
                query=query,
                log_level=log_level,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

            logs = search_result.get("logs", [])

            if format_type.lower() == "json":
                return json.dumps(logs, indent=2, default=str)

            elif format_type.lower() == "csv":
                import csv
                import io

                output = io.StringIO()
                if logs:
                    # Get all unique keys from logs
                    all_keys = set()
                    for log in logs:
                        all_keys.update(log.keys())

                    writer = csv.DictWriter(output, fieldnames=sorted(all_keys))
                    writer.writeheader()

                    for log in logs:
                        # Flatten nested objects for CSV
                        flattened_log = self._flatten_dict(log)
                        writer.writerow(flattened_log)

                return output.getvalue()

            elif format_type.lower() == "txt":
                lines = []
                for log in logs:
                    timestamp = log.get("timestamp", "")
                    level = log.get("level", "")
                    logger_name = log.get("logger", "")
                    message = log.get("message", "")

                    line = f"[{timestamp}] {level} {logger_name}: {message}"

                    # Add exception info if present
                    if "exception" in log:
                        exception_info = log["exception"]
                        if isinstance(exception_info, dict):
                            exc_type = exception_info.get("type", "")
                            exc_message = exception_info.get("message", "")
                            line += f" | Exception: {exc_type}: {exc_message}"

                    lines.append(line)

                return "\n".join(lines)

            else:
                raise ValueError(f"Unsupported export format: {format_type}")

        except Exception as e:
            logger.error(f"Failed to export logs: {e}")
            return f"Error exporting logs: {str(e)}"

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "_"
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, str(v) if v is not None else ""))
        return dict(items)

    async def cleanup_old_logs(self, retention_days: int = 7):
        """Clean up logs older than retention period."""
        try:
            cutoff_timestamp = time.time() - (retention_days * 24 * 3600)

            log_levels = ["debug", "info", "warning", "error", "critical"]
            total_removed = 0

            for level in log_levels:
                # Get old log keys
                old_keys = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.redis_client.zrangebyscore(
                        f"logs_index:{level}", 0, cutoff_timestamp
                    ),
                )

                # Delete old logs
                if old_keys:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda keys=old_keys: self.redis_client.delete(*keys)
                    )

                    # Remove from index
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.redis_client.zremrangebyscore(
                            f"logs_index:{level}", 0, cutoff_timestamp
                        ),
                    )

                    total_removed += len(old_keys)

            if total_removed > 0:
                logger.info(f"Cleaned up {total_removed} old log entries")

            return total_removed

        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")
            return 0


# Global log aggregator instance
log_aggregator = LogAggregator()


async def get_log_aggregator() -> LogAggregator:
    """Dependency to get log aggregator instance."""
    return log_aggregator
