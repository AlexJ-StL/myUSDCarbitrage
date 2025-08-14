"""Log analysis system for identifying patterns and insights."""

import asyncio
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import redis

from .log_aggregation import LogAggregator

logger = logging.getLogger(__name__)


class LogAnalyzer:
    """Analyzes logs to identify patterns and insights."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        log_aggregator: Optional[LogAggregator] = None,
    ):
        """Initialize log analyzer."""
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
        self.log_aggregator = log_aggregator or LogAggregator(redis_url=redis_url)

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    async def analyze_error_patterns(
        self, hours: int = 24, min_occurrences: int = 3
    ) -> Dict[str, Any]:
        """Analyze error logs to identify patterns and recurring issues."""
        try:
            # Get error logs
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            search_results = await self.log_aggregator.search_logs(
                log_level="error", start_time=start_time, end_time=end_time, limit=1000
            )

            error_logs = search_results.get("logs", [])

            if not error_logs:
                return {
                    "message": "No error logs found for analysis",
                    "period_hours": hours,
                }

            # Extract error types and messages
            error_types = []
            error_messages = []
            error_modules = []
            error_functions = []

            for log in error_logs:
                exception_info = log.get("exception", {})
                if isinstance(exception_info, dict):
                    error_type = exception_info.get("type")
                    if error_type:
                        error_types.append(error_type)

                error_messages.append(log.get("message", ""))
                error_modules.append(log.get("module", ""))
                error_functions.append(log.get("function", ""))

            # Analyze error types
            error_type_counts = Counter(error_types)
            common_error_types = [
                {"type": error_type, "count": count}
                for error_type, count in error_type_counts.most_common(10)
                if count >= min_occurrences
            ]

            # Analyze error modules
            module_counts = Counter(error_modules)
            error_prone_modules = [
                {"module": module, "count": count}
                for module, count in module_counts.most_common(10)
                if count >= min_occurrences
            ]

            # Analyze error functions
            function_counts = Counter(error_functions)
            error_prone_functions = [
                {"function": function, "count": count}
                for function, count in function_counts.most_common(10)
                if count >= min_occurrences
            ]

            # Find similar error messages using pattern matching
            error_patterns = await self._find_error_message_patterns(
                error_messages, min_occurrences
            )

            # Analyze error correlations
            correlations = await self._analyze_error_correlations(error_logs)

            return {
                "period_hours": hours,
                "total_errors": len(error_logs),
                "common_error_types": common_error_types,
                "error_prone_modules": error_prone_modules,
                "error_prone_functions": error_prone_functions,
                "error_patterns": error_patterns,
                "correlations": correlations,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to analyze error patterns: {e}")
            return {"error": str(e)}

    async def _find_error_message_patterns(
        self, messages: List[str], min_occurrences: int = 3
    ) -> List[Dict[str, Any]]:
        """Find patterns in error messages."""
        try:
            # Normalize messages
            normalized_messages = []
            for msg in messages:
                if not msg:
                    continue

                # Replace specific values with placeholders
                normalized = re.sub(r"\b\d+\b", "<NUM>", msg)  # Numbers
                normalized = re.sub(
                    r"\b[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
                    "<UUID>",
                    normalized,
                )  # UUIDs
                normalized = re.sub(
                    r"\b[0-9a-f]{24}\b", "<ID>", normalized
                )  # MongoDB ObjectIDs
                normalized = re.sub(
                    r"\b\w+@\w+\.\w+\b", "<EMAIL>", normalized
                )  # Email addresses
                normalized = re.sub(
                    r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", normalized
                )  # IP addresses

                normalized_messages.append(normalized)

            # Count normalized messages
            pattern_counts = Counter(normalized_messages)

            # Extract common patterns
            common_patterns = []
            for pattern, count in pattern_counts.most_common(20):
                if count >= min_occurrences:
                    # Find example original messages
                    examples = []
                    for msg in messages:
                        normalized = re.sub(r"\b\d+\b", "<NUM>", msg)
                        normalized = re.sub(
                            r"\b[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
                            "<UUID>",
                            normalized,
                        )
                        normalized = re.sub(r"\b[0-9a-f]{24}\b", "<ID>", normalized)
                        normalized = re.sub(r"\b\w+@\w+\.\w+\b", "<EMAIL>", normalized)
                        normalized = re.sub(
                            r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", normalized
                        )

                        if normalized == pattern and msg not in examples:
                            examples.append(msg)
                            if len(examples) >= 3:
                                break

                    common_patterns.append({
                        "pattern": pattern,
                        "count": count,
                        "examples": examples,
                    })

            return common_patterns

        except Exception as e:
            logger.error(f"Failed to find error message patterns: {e}")
            return []

    async def _analyze_error_correlations(
        self, error_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze correlations between errors."""
        try:
            # Group errors by time windows
            time_windows = {}
            window_size_minutes = 5

            for log in error_logs:
                try:
                    timestamp = log.get("timestamp", "")
                    if timestamp:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        # Round to nearest window
                        window_key = dt.replace(
                            minute=dt.minute - dt.minute % window_size_minutes,
                            second=0,
                            microsecond=0,
                        ).isoformat()

                        if window_key not in time_windows:
                            time_windows[window_key] = []

                        time_windows[window_key].append(log)
                except ValueError:
                    continue

            # Find co-occurring error types
            co_occurrences = {}

            for window, logs in time_windows.items():
                if len(logs) < 2:
                    continue

                # Extract error types in this window
                error_types = set()
                for log in logs:
                    exception_info = log.get("exception", {})
                    if isinstance(exception_info, dict):
                        error_type = exception_info.get("type")
                        if error_type:
                            error_types.add(error_type)

                # Record co-occurrences
                error_types_list = sorted(error_types)
                for i in range(len(error_types_list)):
                    for j in range(i + 1, len(error_types_list)):
                        pair = (error_types_list[i], error_types_list[j])
                        co_occurrences[pair] = co_occurrences.get(pair, 0) + 1

            # Find significant correlations
            correlations = []
            for (type1, type2), count in sorted(
                co_occurrences.items(), key=lambda x: x[1], reverse=True
            ):
                if count >= 2:  # At least 2 co-occurrences
                    correlations.append({
                        "error_types": [type1, type2],
                        "co_occurrence_count": count,
                    })

            return {
                "co_occurring_errors": correlations[:10],
                "window_size_minutes": window_size_minutes,
            }

        except Exception as e:
            logger.error(f"Failed to analyze error correlations: {e}")
            return {"error": str(e)}

    async def analyze_log_volume_anomalies(
        self, days: int = 7, anomaly_threshold: float = 2.0
    ) -> Dict[str, Any]:
        """Analyze log volume to detect anomalies."""
        try:
            # Get log statistics for each day
            end_time = datetime.now(timezone.utc)
            daily_stats = []

            for day in range(days):
                day_end = end_time - timedelta(days=day)
                day_start = day_end - timedelta(days=1)

                stats = await self.log_aggregator.get_log_statistics(
                    start_time=day_start, end_time=day_end
                )

                daily_stats.append({
                    "date": day_start.date().isoformat(),
                    "total_logs": stats.get("total_logs", 0),
                    "by_level": stats.get("by_level", {}),
                })

            # Calculate baseline (average excluding today)
            if len(daily_stats) <= 1:
                return {
                    "message": "Insufficient data for anomaly detection",
                    "period_days": days,
                }

            baseline_stats = daily_stats[1:]  # Exclude today

            baseline_total = sum(day["total_logs"] for day in baseline_stats) / len(
                baseline_stats
            )
            baseline_by_level = {}

            for level in ["debug", "info", "warning", "error", "critical"]:
                baseline_by_level[level] = sum(
                    day["by_level"].get(level, 0) for day in baseline_stats
                ) / len(baseline_stats)

            # Check for anomalies
            today_stats = daily_stats[0]
            anomalies = []

            # Total volume anomaly
            if today_stats["total_logs"] > baseline_total * anomaly_threshold:
                anomalies.append({
                    "type": "high_total_volume",
                    "current": today_stats["total_logs"],
                    "baseline": baseline_total,
                    "ratio": today_stats["total_logs"] / baseline_total
                    if baseline_total > 0
                    else float("inf"),
                })

            # Level-specific anomalies
            for level in ["warning", "error", "critical"]:
                today_level = today_stats["by_level"].get(level, 0)
                baseline_level = baseline_by_level.get(level, 0)

                if today_level > baseline_level * anomaly_threshold and today_level > 5:
                    anomalies.append({
                        "type": f"high_{level}_volume",
                        "level": level,
                        "current": today_level,
                        "baseline": baseline_level,
                        "ratio": today_level / baseline_level
                        if baseline_level > 0
                        else float("inf"),
                    })

            return {
                "period_days": days,
                "daily_stats": daily_stats,
                "baseline": {
                    "total_logs": baseline_total,
                    "by_level": baseline_by_level,
                },
                "today": today_stats,
                "anomalies": anomalies,
                "anomaly_threshold": anomaly_threshold,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to analyze log volume anomalies: {e}")
            return {"error": str(e)}

    async def identify_error_chains(self, hours: int = 24) -> Dict[str, Any]:
        """Identify chains of related errors that may indicate cascading failures."""
        try:
            # Get error and warning logs
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            error_results = await self.log_aggregator.search_logs(
                log_level="error", start_time=start_time, end_time=end_time, limit=500
            )

            warning_results = await self.log_aggregator.search_logs(
                log_level="warning", start_time=start_time, end_time=end_time, limit=500
            )

            error_logs = error_results.get("logs", [])
            warning_logs = warning_results.get("logs", [])

            if not error_logs:
                return {
                    "message": "No error logs found for analysis",
                    "period_hours": hours,
                }

            # Sort all logs by timestamp
            all_logs = error_logs + warning_logs
            all_logs.sort(key=lambda x: x.get("timestamp", ""))

            # Identify potential error chains
            chains = []
            current_chain = []
            chain_timeout_minutes = 5

            for i, log in enumerate(all_logs):
                if not current_chain:
                    current_chain.append(log)
                    continue

                # Check if this log is within the timeout window of the last log in the chain
                try:
                    last_timestamp = datetime.fromisoformat(
                        current_chain[-1].get("timestamp", "").replace("Z", "+00:00")
                    )
                    current_timestamp = datetime.fromisoformat(
                        log.get("timestamp", "").replace("Z", "+00:00")
                    )

                    time_diff = (
                        current_timestamp - last_timestamp
                    ).total_seconds() / 60

                    if time_diff <= chain_timeout_minutes:
                        # Check if logs are related
                        if self._are_logs_related(current_chain[-1], log):
                            current_chain.append(log)
                        else:
                            # Not related, check if we have a valid chain
                            if len(current_chain) >= 3:
                                chains.append(current_chain)
                            # Start a new chain
                            current_chain = [log]
                    else:
                        # Timeout exceeded, check if we have a valid chain
                        if len(current_chain) >= 3:
                            chains.append(current_chain)
                        # Start a new chain
                        current_chain = [log]

                except (ValueError, TypeError):
                    # Invalid timestamp, skip
                    continue

            # Check the last chain
            if len(current_chain) >= 3:
                chains.append(current_chain)

            # Format chains for output
            formatted_chains = []
            for chain in chains:
                formatted_chain = []
                for log in chain:
                    formatted_chain.append({
                        "timestamp": log.get("timestamp", ""),
                        "level": log.get("level", ""),
                        "message": log.get("message", ""),
                        "module": log.get("module", ""),
                        "function": log.get("function", ""),
                        "exception_type": log.get("exception", {}).get("type")
                        if isinstance(log.get("exception"), dict)
                        else None,
                    })

                formatted_chains.append({
                    "length": len(formatted_chain),
                    "start_time": formatted_chain[0]["timestamp"],
                    "end_time": formatted_chain[-1]["timestamp"],
                    "logs": formatted_chain,
                })

            return {
                "period_hours": hours,
                "total_errors": len(error_logs),
                "total_warnings": len(warning_logs),
                "chains_found": len(formatted_chains),
                "chains": formatted_chains,
                "chain_timeout_minutes": chain_timeout_minutes,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to identify error chains: {e}")
            return {"error": str(e)}

    def _are_logs_related(self, log1: Dict[str, Any], log2: Dict[str, Any]) -> bool:
        """Check if two logs are potentially related."""
        # Check if they're from the same module
        if log1.get("module") == log2.get("module"):
            return True

        # Check if one mentions the other's module
        if log1.get("module") and log1.get("module") in log2.get("message", ""):
            return True
        if log2.get("module") and log2.get("module") in log1.get("message", ""):
            return True

        # Check for common IDs or references in the messages
        msg1 = log1.get("message", "")
        msg2 = log2.get("message", "")

        # Look for UUIDs, IDs, etc.
        id_pattern = r"\b[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\b|\b[0-9a-f]{24}\b"
        ids1 = set(re.findall(id_pattern, msg1))
        ids2 = set(re.findall(id_pattern, msg2))

        if ids1 and ids2 and ids1.intersection(ids2):
            return True

        return False

    async def generate_log_insights(self, days: int = 7) -> Dict[str, Any]:
        """Generate insights from log data."""
        try:
            # Collect various analyses
            error_patterns = await self.analyze_error_patterns(hours=24 * days)
            volume_anomalies = await self.analyze_log_volume_anomalies(days=days)
            error_chains = await self.identify_error_chains(hours=24 * days)

            # Get overall log statistics
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)

            stats = await self.log_aggregator.get_log_statistics(
                start_time=start_time, end_time=end_time
            )

            # Generate insights
            insights = []

            # Error pattern insights
            common_errors = error_patterns.get("common_error_types", [])
            if common_errors:
                insights.append({
                    "type": "common_errors",
                    "title": f"Most common error: {common_errors[0]['type']}",
                    "description": f"Occurred {common_errors[0]['count']} times in the last {days} days",
                    "priority": "high" if common_errors[0]["count"] > 10 else "medium",
                })

            error_modules = error_patterns.get("error_prone_modules", [])
            if error_modules:
                insights.append({
                    "type": "error_prone_module",
                    "title": f"Most error-prone module: {error_modules[0]['module']}",
                    "description": f"Generated {error_modules[0]['count']} errors in the last {days} days",
                    "priority": "high" if error_modules[0]["count"] > 10 else "medium",
                })

            # Volume anomaly insights
            anomalies = volume_anomalies.get("anomalies", [])
            for anomaly in anomalies:
                if anomaly["type"] == "high_error_volume":
                    insights.append({
                        "type": "volume_anomaly",
                        "title": "Unusual error volume detected",
                        "description": f"Error volume is {anomaly['ratio']:.1f}x higher than baseline",
                        "priority": "high",
                    })
                elif anomaly["type"] == "high_critical_volume":
                    insights.append({
                        "type": "volume_anomaly",
                        "title": "Unusual critical error volume detected",
                        "description": f"Critical error volume is {anomaly['ratio']:.1f}x higher than baseline",
                        "priority": "critical",
                    })

            # Error chain insights
            chains = error_chains.get("chains", [])
            if chains:
                longest_chain = max(chains, key=lambda x: x["length"])
                insights.append({
                    "type": "error_chain",
                    "title": f"Detected error chain of {longest_chain['length']} related logs",
                    "description": "Possible cascading failure detected",
                    "priority": "high" if longest_chain["length"] > 5 else "medium",
                })

            # Overall health insights
            total_logs = stats.get("total_logs", 0)
            error_count = stats.get("by_level", {}).get("error", 0)
            critical_count = stats.get("by_level", {}).get("critical", 0)

            error_ratio = error_count / total_logs if total_logs > 0 else 0

            if error_ratio < 0.01 and critical_count == 0:
                insights.append({
                    "type": "system_health",
                    "title": "System health is good",
                    "description": f"Error ratio is low at {error_ratio:.1%}",
                    "priority": "low",
                })
            elif error_ratio > 0.05 or critical_count > 0:
                insights.append({
                    "type": "system_health",
                    "title": "System health needs attention",
                    "description": f"Error ratio is {error_ratio:.1%} with {critical_count} critical errors",
                    "priority": "high",
                })

            return {
                "period_days": days,
                "insights": insights,
                "log_statistics": {
                    "total_logs": total_logs,
                    "by_level": stats.get("by_level", {}),
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to generate log insights: {e}")
            return {"error": str(e)}


# Global log analyzer instance
log_analyzer = LogAnalyzer()


async def get_log_analyzer() -> LogAnalyzer:
    """Dependency to get log analyzer instance."""
    return log_analyzer
