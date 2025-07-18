"""Performance profiling and bottleneck identification system."""

import asyncio
import functools
import logging
import time
import tracemalloc
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
import redis

from .logging_config import PerformanceLogger

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Comprehensive performance profiling system."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize performance profiler."""
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
        self.performance_logger = PerformanceLogger("performance_profiler")
        self.active_profiles = {}

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    @contextmanager
    def profile_execution(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for profiling execution time and memory usage."""
        # Start memory tracking
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]

        # Get initial system stats
        process = psutil.Process()
        start_cpu_time = process.cpu_times()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Calculate execution time
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000

            # Calculate memory usage
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            memory_used_mb = (current_memory - start_memory) / (1024 * 1024)
            peak_memory_mb = peak_memory / (1024 * 1024)
            tracemalloc.stop()

            # Calculate CPU usage
            end_cpu_time = process.cpu_times()
            cpu_time_used = (end_cpu_time.user - start_cpu_time.user) + (
                end_cpu_time.system - start_cpu_time.system
            )

            # Create performance profile
            profile_data = {
                "operation": operation_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_time_ms": execution_time_ms,
                "memory_used_mb": memory_used_mb,
                "peak_memory_mb": peak_memory_mb,
                "cpu_time_seconds": cpu_time_used,
                "metadata": metadata or {},
            }

            # Store profile data
            asyncio.create_task(self._store_profile_data(profile_data))

            # Log performance metrics
            self.performance_logger.log_execution_time(
                operation_name,
                execution_time_ms,
                {
                    "memory_used_mb": memory_used_mb,
                    "peak_memory_mb": peak_memory_mb,
                    "cpu_time_seconds": cpu_time_used,
                    **(metadata or {}),
                },
            )

    @asynccontextmanager
    async def profile_async_execution(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Async context manager for profiling async operations."""
        # Start memory tracking
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]

        # Get initial system stats
        process = psutil.Process()
        start_cpu_time = process.cpu_times()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Calculate execution time
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000

            # Calculate memory usage
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            memory_used_mb = (current_memory - start_memory) / (1024 * 1024)
            peak_memory_mb = peak_memory / (1024 * 1024)
            tracemalloc.stop()

            # Calculate CPU usage
            end_cpu_time = process.cpu_times()
            cpu_time_used = (end_cpu_time.user - start_cpu_time.user) + (
                end_cpu_time.system - start_cpu_time.system
            )

            # Create performance profile
            profile_data = {
                "operation": operation_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_time_ms": execution_time_ms,
                "memory_used_mb": memory_used_mb,
                "peak_memory_mb": peak_memory_mb,
                "cpu_time_seconds": cpu_time_used,
                "metadata": metadata or {},
            }

            # Store profile data
            await self._store_profile_data(profile_data)

            # Log performance metrics
            self.performance_logger.log_execution_time(
                operation_name,
                execution_time_ms,
                {
                    "memory_used_mb": memory_used_mb,
                    "peak_memory_mb": peak_memory_mb,
                    "cpu_time_seconds": cpu_time_used,
                    **(metadata or {}),
                },
            )

    def profile_function(
        self,
        operation_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Decorator for profiling function execution."""

        def decorator(func: Callable) -> Callable:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.profile_async_execution(op_name, metadata):
                        return await func(*args, **kwargs)

                return async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.profile_execution(op_name, metadata):
                        return func(*args, **kwargs)

                return sync_wrapper

        return decorator

    async def _store_profile_data(self, profile_data: Dict[str, Any]):
        """Store performance profile data in Redis."""
        try:
            import json

            # Generate profile ID
            timestamp = int(time.time())
            profile_id = f"profile_{timestamp}_{hash(profile_data['operation'])}"

            # Store profile data
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.setex(
                    f"profiles:{profile_id}",
                    7 * 24 * 3600,  # 7 days TTL
                    json.dumps(profile_data, default=str),
                ),
            )

            # Add to time-based index
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zadd(
                    "profiles_index", {f"profiles:{profile_id}": timestamp}
                ),
            )

            # Add to operation-specific index
            operation_key = (
                profile_data["operation"].replace(".", "_").replace(" ", "_")
            )
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zadd(
                    f"profiles_operation:{operation_key}",
                    {f"profiles:{profile_id}": timestamp},
                ),
            )

        except Exception as e:
            logger.error(f"Failed to store profile data: {e}")

    async def get_performance_profiles(
        self, operation: Optional[str] = None, hours: int = 24, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve performance profiles with optional filtering."""
        try:
            import json

            start_time = time.time() - (hours * 3600)
            end_time = time.time()

            # Choose appropriate index
            if operation:
                operation_key = operation.replace(".", "_").replace(" ", "_")
                index_key = f"profiles_operation:{operation_key}"
            else:
                index_key = "profiles_index"

            # Get profile keys
            profile_keys = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zrangebyscore(
                    index_key, start_time, end_time, start=0, num=limit, desc=True
                ),
            )

            # Retrieve profile data
            profiles = []
            for key in profile_keys:
                profile_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                if profile_data:
                    try:
                        profiles.append(json.loads(profile_data))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode profile data for key: {key}")

            return profiles

        except Exception as e:
            logger.error(f"Failed to retrieve performance profiles: {e}")
            return []

    async def analyze_bottlenecks(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance data to identify bottlenecks."""
        try:
            profiles = await self.get_performance_profiles(hours=hours, limit=1000)

            if not profiles:
                return {"message": "No performance data available for analysis"}

            # Group profiles by operation
            operations = {}
            for profile in profiles:
                op_name = profile.get("operation", "unknown")
                if op_name not in operations:
                    operations[op_name] = []
                operations[op_name].append(profile)

            # Analyze each operation
            analysis = {
                "analysis_period_hours": hours,
                "total_profiles": len(profiles),
                "operations_analyzed": len(operations),
                "bottlenecks": [],
                "recommendations": [],
                "summary": {},
            }

            operation_stats = []

            for op_name, op_profiles in operations.items():
                # Calculate statistics
                execution_times = [p.get("execution_time_ms", 0) for p in op_profiles]
                memory_usage = [p.get("memory_used_mb", 0) for p in op_profiles]
                cpu_times = [p.get("cpu_time_seconds", 0) for p in op_profiles]

                stats = {
                    "operation": op_name,
                    "call_count": len(op_profiles),
                    "avg_execution_time_ms": sum(execution_times)
                    / len(execution_times),
                    "max_execution_time_ms": max(execution_times),
                    "min_execution_time_ms": min(execution_times),
                    "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                    "max_memory_mb": max(memory_usage),
                    "avg_cpu_time_seconds": sum(cpu_times) / len(cpu_times),
                    "total_cpu_time_seconds": sum(cpu_times),
                }

                operation_stats.append(stats)

            # Sort by different metrics to identify bottlenecks
            slowest_operations = sorted(
                operation_stats, key=lambda x: x["avg_execution_time_ms"], reverse=True
            )[:5]
            memory_intensive = sorted(
                operation_stats, key=lambda x: x["avg_memory_mb"], reverse=True
            )[:5]
            cpu_intensive = sorted(
                operation_stats, key=lambda x: x["total_cpu_time_seconds"], reverse=True
            )[:5]
            most_frequent = sorted(
                operation_stats, key=lambda x: x["call_count"], reverse=True
            )[:5]

            analysis["bottlenecks"] = {
                "slowest_operations": slowest_operations,
                "memory_intensive_operations": memory_intensive,
                "cpu_intensive_operations": cpu_intensive,
                "most_frequent_operations": most_frequent,
            }

            # Generate recommendations
            recommendations = []

            # Check for slow operations
            for op in slowest_operations[:3]:
                if op["avg_execution_time_ms"] > 1000:  # > 1 second
                    recommendations.append(
                        f"Operation '{op['operation']}' has high average execution time "
                        f"({op['avg_execution_time_ms']:.1f}ms). Consider optimization."
                    )

            # Check for memory-intensive operations
            for op in memory_intensive[:3]:
                if op["avg_memory_mb"] > 100:  # > 100MB
                    recommendations.append(
                        f"Operation '{op['operation']}' uses significant memory "
                        f"({op['avg_memory_mb']:.1f}MB). Consider memory optimization."
                    )

            # Check for frequently called slow operations
            for op in most_frequent[:3]:
                if op["call_count"] > 100 and op["avg_execution_time_ms"] > 100:
                    recommendations.append(
                        f"Operation '{op['operation']}' is called frequently ({op['call_count']} times) "
                        f"and has moderate execution time ({op['avg_execution_time_ms']:.1f}ms). "
                        "Consider caching or optimization."
                    )

            analysis["recommendations"] = recommendations

            # Summary statistics
            all_execution_times = [p.get("execution_time_ms", 0) for p in profiles]
            all_memory_usage = [p.get("memory_used_mb", 0) for p in profiles]

            analysis["summary"] = {
                "avg_execution_time_ms": sum(all_execution_times)
                / len(all_execution_times),
                "max_execution_time_ms": max(all_execution_times),
                "avg_memory_usage_mb": sum(all_memory_usage) / len(all_memory_usage),
                "max_memory_usage_mb": max(all_memory_usage),
                "total_operations": len(operations),
            }

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze bottlenecks: {e}")
            return {"error": str(e)}

    async def get_operation_trends(
        self, operation: str, days: int = 7
    ) -> Dict[str, Any]:
        """Get performance trends for a specific operation."""
        try:
            profiles = await self.get_performance_profiles(
                operation=operation, hours=days * 24, limit=1000
            )

            if not profiles:
                return {
                    "message": f"No performance data available for operation: {operation}"
                }

            # Group by day
            daily_stats = {}
            for profile in profiles:
                timestamp = profile.get("timestamp", "")
                if timestamp:
                    try:
                        date = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        ).date()
                        date_str = date.isoformat()

                        if date_str not in daily_stats:
                            daily_stats[date_str] = {
                                "execution_times": [],
                                "memory_usage": [],
                                "cpu_times": [],
                                "call_count": 0,
                            }

                        daily_stats[date_str]["execution_times"].append(
                            profile.get("execution_time_ms", 0)
                        )
                        daily_stats[date_str]["memory_usage"].append(
                            profile.get("memory_used_mb", 0)
                        )
                        daily_stats[date_str]["cpu_times"].append(
                            profile.get("cpu_time_seconds", 0)
                        )
                        daily_stats[date_str]["call_count"] += 1

                    except ValueError:
                        continue

            # Calculate daily averages
            trends = []
            for date_str, stats in sorted(daily_stats.items()):
                execution_times = stats["execution_times"]
                memory_usage = stats["memory_usage"]
                cpu_times = stats["cpu_times"]

                trends.append({
                    "date": date_str,
                    "call_count": stats["call_count"],
                    "avg_execution_time_ms": sum(execution_times)
                    / len(execution_times),
                    "max_execution_time_ms": max(execution_times),
                    "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                    "max_memory_mb": max(memory_usage),
                    "total_cpu_time_seconds": sum(cpu_times),
                })

            return {
                "operation": operation,
                "period_days": days,
                "total_profiles": len(profiles),
                "daily_trends": trends,
            }

        except Exception as e:
            logger.error(f"Failed to get operation trends: {e}")
            return {"error": str(e)}


# Global performance profiler instance
performance_profiler = PerformanceProfiler()


def get_performance_profiler() -> PerformanceProfiler:
    """Get performance profiler instance."""
    return performance_profiler


# Convenience decorators
def profile_performance(
    operation_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
):
    """Decorator for profiling function performance."""
    return performance_profiler.profile_function(operation_name, metadata)


@contextmanager
def profile_block(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for profiling code blocks."""
    with performance_profiler.profile_execution(operation_name, metadata):
        yield


@asynccontextmanager
async def profile_async_block(
    operation_name: str, metadata: Optional[Dict[str, Any]] = None
):
    """Async context manager for profiling async code blocks."""
    async with performance_profiler.profile_async_execution(operation_name, metadata):
        yield
