"""Comprehensive health check system for all services."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil
import redis
from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..api.database import get_db

logger = logging.getLogger(__name__)


class HealthCheckStatus:
    """Health check status constants."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceHealthChecker:
    """Comprehensive service health monitoring."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize health checker with Redis connection."""
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    async def check_database_health(self, db: Session) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        start_time = time.time()

        try:
            # Test basic connectivity
            result = db.execute(text("SELECT 1")).scalar()
            if result != 1:
                raise Exception("Database query returned unexpected result")

            # Test database performance
            db.execute(text("SELECT COUNT(*) FROM information_schema.tables"))

            response_time = (time.time() - start_time) * 1000  # Convert to ms

            # Check connection pool status
            pool_info = {
                "size": db.bind.pool.size(),
                "checked_in": db.bind.pool.checkedin(),
                "checked_out": db.bind.pool.checkedout(),
                "overflow": db.bind.pool.overflow(),
            }

            status = HealthCheckStatus.HEALTHY
            if response_time > 1000:  # > 1 second
                status = HealthCheckStatus.DEGRADED
            elif response_time > 5000:  # > 5 seconds
                status = HealthCheckStatus.UNHEALTHY

            return {
                "status": status,
                "response_time_ms": round(response_time, 2),
                "pool_info": pool_info,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": HealthCheckStatus.UNHEALTHY,
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        start_time = time.time()

        try:
            # Test basic connectivity
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)

            # Test read/write operations
            test_key = "health_check_test"
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.set, test_key, "test_value", "ex", 10
            )

            value = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, test_key
            )

            if value != "test_value":
                raise Exception("Redis read/write test failed")

            # Clean up test key
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.delete, test_key
            )

            response_time = (time.time() - start_time) * 1000

            # Get Redis info
            info = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.info
            )

            memory_usage = info.get("used_memory_human", "unknown")
            connected_clients = info.get("connected_clients", 0)

            status = HealthCheckStatus.HEALTHY
            if response_time > 500:  # > 500ms
                status = HealthCheckStatus.DEGRADED
            elif response_time > 2000:  # > 2 seconds
                status = HealthCheckStatus.UNHEALTHY

            return {
                "status": status,
                "response_time_ms": round(response_time, 2),
                "memory_usage": memory_usage,
                "connected_clients": connected_clients,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": HealthCheckStatus.UNHEALTHY,
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system CPU, memory, and disk usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)

            # Network I/O
            network = psutil.net_io_counters()

            # Process count
            process_count = len(psutil.pids())

            # Determine overall system status
            status = HealthCheckStatus.HEALTHY
            if cpu_percent > 80 or memory_percent > 85 or disk_percent > 90:
                status = HealthCheckStatus.DEGRADED
            if cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
                status = HealthCheckStatus.UNHEALTHY

            return {
                "status": status,
                "cpu": {
                    "usage_percent": round(cpu_percent, 2),
                    "count": cpu_count,
                },
                "memory": {
                    "usage_percent": round(memory_percent, 2),
                    "available_gb": round(memory_available_gb, 2),
                    "total_gb": round(memory.total / (1024**3), 2),
                },
                "disk": {
                    "usage_percent": round(disk_percent, 2),
                    "free_gb": round(disk_free_gb, 2),
                    "total_gb": round(disk.total / (1024**3), 2),
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                },
                "processes": process_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {
                "status": HealthCheckStatus.UNHEALTHY,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def check_service_dependencies(self) -> Dict[str, Any]:
        """Check external service dependencies."""
        dependencies = {}

        # Check external exchange APIs (sample check)
        exchange_apis = [
            {"name": "binance", "url": "https://api.binance.com/api/v3/ping"},
            {"name": "coinbase", "url": "https://api.exchange.coinbase.com/time"},
        ]

        for api in exchange_apis:
            start_time = time.time()
            try:
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(api["url"])
                    response_time = (time.time() - start_time) * 1000

                    if response.status_code == 200:
                        status = HealthCheckStatus.HEALTHY
                        if response_time > 2000:
                            status = HealthCheckStatus.DEGRADED
                    else:
                        status = HealthCheckStatus.UNHEALTHY

                    dependencies[api["name"]] = {
                        "status": status,
                        "response_time_ms": round(response_time, 2),
                        "status_code": response.status_code,
                    }

            except Exception as e:
                dependencies[api["name"]] = {
                    "status": HealthCheckStatus.UNHEALTHY,
                    "error": str(e),
                    "response_time_ms": (time.time() - start_time) * 1000,
                }

        # Overall dependency status
        all_statuses = [dep["status"] for dep in dependencies.values()]
        if all(status == HealthCheckStatus.HEALTHY for status in all_statuses):
            overall_status = HealthCheckStatus.HEALTHY
        elif any(status == HealthCheckStatus.UNHEALTHY for status in all_statuses):
            overall_status = HealthCheckStatus.DEGRADED
        else:
            overall_status = HealthCheckStatus.DEGRADED

        return {
            "status": overall_status,
            "dependencies": dependencies,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def comprehensive_health_check(self, db: Session) -> Dict[str, Any]:
        """Perform comprehensive health check of all services."""
        start_time = time.time()

        # Run all health checks concurrently
        (
            database_health,
            redis_health,
            system_health,
            dependencies_health,
        ) = await asyncio.gather(
            self.check_database_health(db),
            self.check_redis_health(),
            asyncio.get_event_loop().run_in_executor(None, self.check_system_resources),
            self.check_service_dependencies(),
            return_exceptions=True,
        )

        # Handle any exceptions from concurrent execution
        checks = {
            "database": database_health
            if not isinstance(database_health, Exception)
            else {"status": HealthCheckStatus.UNHEALTHY, "error": str(database_health)},
            "redis": redis_health
            if not isinstance(redis_health, Exception)
            else {"status": HealthCheckStatus.UNHEALTHY, "error": str(redis_health)},
            "system": system_health
            if not isinstance(system_health, Exception)
            else {"status": HealthCheckStatus.UNHEALTHY, "error": str(system_health)},
            "dependencies": dependencies_health
            if not isinstance(dependencies_health, Exception)
            else {
                "status": HealthCheckStatus.UNHEALTHY,
                "error": str(dependencies_health),
            },
        }

        # Determine overall system status
        all_statuses = [check["status"] for check in checks.values()]
        if all(status == HealthCheckStatus.HEALTHY for status in all_statuses):
            overall_status = HealthCheckStatus.HEALTHY
        elif any(status == HealthCheckStatus.UNHEALTHY for status in all_statuses):
            overall_status = HealthCheckStatus.UNHEALTHY
        else:
            overall_status = HealthCheckStatus.DEGRADED

        total_time = (time.time() - start_time) * 1000

        return {
            "status": overall_status,
            "checks": checks,
            "total_check_time_ms": round(total_time, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "0.1.0",
        }


# Global health checker instance
health_checker = ServiceHealthChecker()


async def get_health_checker() -> ServiceHealthChecker:
    """Dependency to get health checker instance."""
    return health_checker
