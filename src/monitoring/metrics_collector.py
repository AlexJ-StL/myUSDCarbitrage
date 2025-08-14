"""System metrics collection and monitoring."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil
import redis
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..api.database import get_db

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and stores system performance metrics."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize metrics collector."""
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
        self.metrics_retention_hours = 24  # Keep metrics for 24 hours

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        timestamp = datetime.now(timezone.utc)

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_times = psutil.cpu_times()
        cpu_freq = psutil.cpu_freq()

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk metrics
        disk_usage = psutil.disk_usage("/")
        disk_io = psutil.disk_io_counters()

        # Network metrics
        network_io = psutil.net_io_counters()

        # Process metrics
        process_count = len(psutil.pids())

        # Load average (Unix-like systems)
        try:
            load_avg = psutil.getloadavg()
        except AttributeError:
            # Windows doesn't have load average
            load_avg = [0.0, 0.0, 0.0]

        return {
            "timestamp": timestamp.isoformat(),
            "cpu": {
                "usage_percent": cpu_percent,
                "user_time": cpu_times.user,
                "system_time": cpu_times.system,
                "idle_time": cpu_times.idle,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
            },
            "memory": {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_bytes": memory.used,
                "usage_percent": memory.percent,
                "cached_bytes": getattr(memory, "cached", 0),
                "buffers_bytes": getattr(memory, "buffers", 0),
            },
            "swap": {
                "total_bytes": swap.total,
                "used_bytes": swap.used,
                "free_bytes": swap.free,
                "usage_percent": swap.percent,
            },
            "disk": {
                "total_bytes": disk_usage.total,
                "used_bytes": disk_usage.used,
                "free_bytes": disk_usage.free,
                "usage_percent": (disk_usage.used / disk_usage.total) * 100,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0,
            },
            "network": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv,
                "errors_in": network_io.errin,
                "errors_out": network_io.errout,
                "drops_in": network_io.dropin,
                "drops_out": network_io.dropout,
            },
            "system": {
                "process_count": process_count,
                "load_average_1m": load_avg[0],
                "load_average_5m": load_avg[1],
                "load_average_15m": load_avg[2],
                "boot_time": psutil.boot_time(),
            },
        }

    async def collect_database_metrics(self, db: Session) -> Dict[str, Any]:
        """Collect database performance metrics."""
        timestamp = datetime.now(timezone.utc)

        try:
            # Connection pool metrics
            pool = db.bind.pool
            pool_metrics = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid(),
            }

            # Query performance test
            start_time = time.time()
            db.execute(text("SELECT 1")).scalar()
            query_time = (time.time() - start_time) * 1000

            # Database size and statistics (PostgreSQL specific)
            try:
                db_stats = db.execute(
                    text("""
                    SELECT 
                        pg_database_size(current_database()) as db_size,
                        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                        (SELECT count(*) FROM pg_stat_activity) as total_connections
                """)
                ).fetchone()

                db_size = db_stats[0] if db_stats else 0
                active_connections = db_stats[1] if db_stats else 0
                total_connections = db_stats[2] if db_stats else 0

            except Exception as e:
                logger.warning(f"Could not collect database statistics: {e}")
                db_size = 0
                active_connections = 0
                total_connections = 0

            return {
                "timestamp": timestamp.isoformat(),
                "pool": pool_metrics,
                "performance": {
                    "query_time_ms": round(query_time, 2),
                },
                "database": {
                    "size_bytes": db_size,
                    "active_connections": active_connections,
                    "total_connections": total_connections,
                },
            }

        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
            return {
                "timestamp": timestamp.isoformat(),
                "error": str(e),
            }

    async def collect_redis_metrics(self) -> Dict[str, Any]:
        """Collect Redis performance metrics."""
        timestamp = datetime.now(timezone.utc)

        try:
            # Get Redis info
            info = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.info
            )

            # Test Redis performance
            start_time = time.time()
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            ping_time = (time.time() - start_time) * 1000

            return {
                "timestamp": timestamp.isoformat(),
                "memory": {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_peak": info.get("used_memory_peak", 0),
                    "used_memory_rss": info.get("used_memory_rss", 0),
                    "maxmemory": info.get("maxmemory", 0),
                },
                "connections": {
                    "connected_clients": info.get("connected_clients", 0),
                    "blocked_clients": info.get("blocked_clients", 0),
                    "total_connections_received": info.get(
                        "total_connections_received", 0
                    ),
                },
                "performance": {
                    "ping_time_ms": round(ping_time, 2),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "instantaneous_ops_per_sec": info.get(
                        "instantaneous_ops_per_sec", 0
                    ),
                },
                "persistence": {
                    "rdb_last_save_time": info.get("rdb_last_save_time", 0),
                    "rdb_changes_since_last_save": info.get(
                        "rdb_changes_since_last_save", 0
                    ),
                },
                "keyspace": {
                    key: value for key, value in info.items() if key.startswith("db")
                },
            }

        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")
            return {
                "timestamp": timestamp.isoformat(),
                "error": str(e),
            }

    async def store_metrics(self, metrics: Dict[str, Any], metric_type: str):
        """Store metrics in Redis with TTL."""
        try:
            key = f"metrics:{metric_type}:{int(time.time())}"

            # Store metrics as JSON string
            import json

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.setex(
                    key,
                    self.metrics_retention_hours * 3600,  # TTL in seconds
                    json.dumps(metrics, default=str),
                ),
            )

            # Also store in a sorted set for time-based queries
            score = time.time()
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zadd(
                    f"metrics_index:{metric_type}", {key: score}
                ),
            )

            # Clean up old entries from sorted set
            cutoff_time = time.time() - (self.metrics_retention_hours * 3600)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zremrangebyscore(
                    f"metrics_index:{metric_type}", 0, cutoff_time
                ),
            )

        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")

    async def get_metrics_history(
        self,
        metric_type: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical metrics from Redis."""
        try:
            if start_time is None:
                start_time = time.time() - 3600  # Last hour by default
            if end_time is None:
                end_time = time.time()

            # Get metric keys from sorted set
            keys = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zrangebyscore(
                    f"metrics_index:{metric_type}",
                    start_time,
                    end_time,
                    start=0,
                    num=limit,
                ),
            )

            if not keys:
                return []

            # Get metric data
            import json

            metrics_data = []

            for key in keys:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                if data:
                    try:
                        metrics_data.append(json.loads(data))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode metrics data for key: {key}")

            return sorted(metrics_data, key=lambda x: x.get("timestamp", ""))

        except Exception as e:
            logger.error(f"Failed to retrieve metrics history: {e}")
            return []

    async def collect_and_store_all_metrics(self, db: Session):
        """Collect and store all system metrics."""
        try:
            # Collect all metrics concurrently
            system_metrics, db_metrics, redis_metrics = await asyncio.gather(
                asyncio.get_event_loop().run_in_executor(
                    None, self.collect_system_metrics
                ),
                self.collect_database_metrics(db),
                self.collect_redis_metrics(),
                return_exceptions=True,
            )

            # Store metrics
            if not isinstance(system_metrics, Exception):
                await self.store_metrics(system_metrics, "system")

            if not isinstance(db_metrics, Exception):
                await self.store_metrics(db_metrics, "database")

            if not isinstance(redis_metrics, Exception):
                await self.store_metrics(redis_metrics, "redis")

            logger.info("Successfully collected and stored all metrics")

        except Exception as e:
            logger.error(f"Failed to collect and store metrics: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


async def get_metrics_collector() -> MetricsCollector:
    """Dependency to get metrics collector instance."""
    return metrics_collector
