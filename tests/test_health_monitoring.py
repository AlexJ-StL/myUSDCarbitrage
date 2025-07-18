"""Tests for health monitoring system."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.database import Base, get_db
from src.api.main import app
from src.monitoring.health_checks import ServiceHealthChecker, HealthCheckStatus
from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.service_recovery import ServiceRecoveryManager


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_health.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


class TestServiceHealthChecker:
    """Test service health checking functionality."""

    @pytest.fixture
    def health_checker(self):
        """Create health checker instance for testing."""
        return ServiceHealthChecker(redis_url="redis://localhost:6379/1")

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        db = MagicMock()
        db.execute.return_value.scalar.return_value = 1
        db.bind.pool.size.return_value = 10
        db.bind.pool.checkedin.return_value = 8
        db.bind.pool.checkedout.return_value = 2
        db.bind.pool.overflow.return_value = 0
        return db

    @pytest.mark.asyncio
    async def test_database_health_check_success(self, health_checker, mock_db):
        """Test successful database health check."""
        result = await health_checker.check_database_health(mock_db)

        assert result["status"] == HealthCheckStatus.HEALTHY
        assert "response_time_ms" in result
        assert "pool_info" in result
        assert result["pool_info"]["size"] == 10

    @pytest.mark.asyncio
    async def test_database_health_check_failure(self, health_checker):
        """Test database health check failure."""
        mock_db = MagicMock()
        mock_db.execute.side_effect = Exception("Database connection failed")

        result = await health_checker.check_database_health(mock_db)

        assert result["status"] == HealthCheckStatus.UNHEALTHY
        assert "error" in result
        assert "Database connection failed" in result["error"]

    @pytest.mark.asyncio
    @patch("redis.from_url")
    async def test_redis_health_check_success(
        self, mock_redis_from_url, health_checker
    ):
        """Test successful Redis health check."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.set.return_value = True
        mock_redis.get.return_value = "test_value"
        mock_redis.delete.return_value = 1
        mock_redis.info.return_value = {
            "used_memory_human": "1.5M",
            "connected_clients": 5,
        }
        mock_redis_from_url.return_value = mock_redis

        # Reset the cached client
        health_checker._redis_client = None

        result = await health_checker.check_redis_health()

        assert result["status"] == HealthCheckStatus.HEALTHY
        assert "response_time_ms" in result
        assert result["memory_usage"] == "1.5M"
        assert result["connected_clients"] == 5

    @pytest.mark.asyncio
    @patch("redis.from_url")
    async def test_redis_health_check_failure(
        self, mock_redis_from_url, health_checker
    ):
        """Test Redis health check failure."""
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Redis connection failed")
        mock_redis_from_url.return_value = mock_redis

        # Reset the cached client
        health_checker._redis_client = None

        result = await health_checker.check_redis_health()

        assert result["status"] == HealthCheckStatus.UNHEALTHY
        assert "error" in result
        assert "Redis connection failed" in result["error"]

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.net_io_counters")
    @patch("psutil.pids")
    def test_system_resources_check_healthy(
        self, mock_pids, mock_net, mock_disk, mock_memory, mock_cpu, health_checker
    ):
        """Test system resources check with healthy status."""
        # Mock system metrics
        mock_cpu.return_value = 25.0
        mock_memory.return_value = MagicMock(
            percent=60.0, available=4 * 1024**3, total=8 * 1024**3
        )
        mock_disk.return_value = MagicMock(
            used=50 * 1024**3, total=100 * 1024**3, free=50 * 1024**3
        )
        mock_net.return_value = MagicMock(
            bytes_sent=1000000, bytes_recv=2000000, packets_sent=1000, packets_recv=2000
        )
        mock_pids.return_value = list(range(100))

        result = health_checker.check_system_resources()

        assert result["status"] == HealthCheckStatus.HEALTHY
        assert result["cpu"]["usage_percent"] == 25.0
        assert result["memory"]["usage_percent"] == 60.0
        assert result["processes"] == 100

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_system_resources_check_degraded(
        self, mock_disk, mock_memory, mock_cpu, health_checker
    ):
        """Test system resources check with degraded status."""
        # Mock high resource usage
        mock_cpu.return_value = 85.0  # High CPU
        mock_memory.return_value = MagicMock(
            percent=90.0,
            available=1 * 1024**3,
            total=8 * 1024**3,  # High memory
        )
        mock_disk.return_value = MagicMock(
            used=95 * 1024**3,
            total=100 * 1024**3,
            free=5 * 1024**3,  # High disk
        )

        result = health_checker.check_system_resources()

        assert result["status"] == HealthCheckStatus.DEGRADED

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_service_dependencies_check(self, mock_client, health_checker):
        """Test external service dependencies check."""
        # Mock successful HTTP responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_response
        )

        result = await health_checker.check_service_dependencies()

        assert "dependencies" in result
        assert "binance" in result["dependencies"]
        assert "coinbase" in result["dependencies"]

    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self, health_checker, mock_db):
        """Test comprehensive health check combining all checks."""
        with (
            patch.object(health_checker, "check_redis_health") as mock_redis,
            patch.object(health_checker, "check_system_resources") as mock_system,
            patch.object(health_checker, "check_service_dependencies") as mock_deps,
        ):
            # Mock all health checks to return healthy status
            mock_redis.return_value = {"status": HealthCheckStatus.HEALTHY}
            mock_system.return_value = {"status": HealthCheckStatus.HEALTHY}
            mock_deps.return_value = {"status": HealthCheckStatus.HEALTHY}

            result = await health_checker.comprehensive_health_check(mock_db)

            assert result["status"] == HealthCheckStatus.HEALTHY
            assert "checks" in result
            assert "database" in result["checks"]
            assert "redis" in result["checks"]
            assert "system" in result["checks"]
            assert "dependencies" in result["checks"]


class TestMetricsCollector:
    """Test metrics collection functionality."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance for testing."""
        return MetricsCollector(redis_url="redis://localhost:6379/1")

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_collect_system_metrics(
        self, mock_disk, mock_memory, mock_cpu, metrics_collector
    ):
        """Test system metrics collection."""
        # Mock system data
        mock_cpu.return_value = 30.0
        mock_memory.return_value = MagicMock(
            total=8 * 1024**3, available=4 * 1024**3, used=4 * 1024**3, percent=50.0
        )
        mock_disk.return_value = MagicMock(
            total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
        )

        metrics = metrics_collector.collect_system_metrics()

        assert "timestamp" in metrics
        assert "cpu" in metrics
        assert "memory" in metrics
        assert "disk" in metrics
        assert metrics["cpu"]["usage_percent"] == 30.0
        assert metrics["memory"]["usage_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_collect_database_metrics(self, metrics_collector):
        """Test database metrics collection."""
        mock_db = MagicMock()
        mock_db.bind.pool.size.return_value = 10
        mock_db.bind.pool.checkedin.return_value = 8
        mock_db.bind.pool.checkedout.return_value = 2
        mock_db.execute.return_value.scalar.return_value = 1
        mock_db.execute.return_value.fetchone.return_value = (1000000, 5, 10)

        metrics = await metrics_collector.collect_database_metrics(mock_db)

        assert "timestamp" in metrics
        assert "pool" in metrics
        assert "performance" in metrics
        assert "database" in metrics
        assert metrics["pool"]["size"] == 10

    @pytest.mark.asyncio
    @patch("redis.from_url")
    async def test_store_and_retrieve_metrics(
        self, mock_redis_from_url, metrics_collector
    ):
        """Test storing and retrieving metrics."""
        mock_redis = MagicMock()
        mock_redis.setex.return_value = True
        mock_redis.zadd.return_value = 1
        mock_redis.zremrangebyscore.return_value = 0
        mock_redis.zrangebyscore.return_value = ["metrics:test:123456"]
        mock_redis.get.return_value = json.dumps({
            "test": "data",
            "timestamp": "2023-01-01T00:00:00Z",
        })
        mock_redis_from_url.return_value = mock_redis

        # Reset the cached client
        metrics_collector._redis_client = None

        # Test storing metrics
        test_metrics = {"test": "data", "timestamp": "2023-01-01T00:00:00Z"}
        await metrics_collector.store_metrics(test_metrics, "test")

        # Test retrieving metrics
        history = await metrics_collector.get_metrics_history("test")

        assert len(history) == 1
        assert history[0]["test"] == "data"


class TestServiceRecoveryManager:
    """Test service recovery functionality."""

    @pytest.fixture
    def recovery_manager(self):
        """Create service recovery manager instance for testing."""
        return ServiceRecoveryManager(redis_url="redis://localhost:6379/1")

    @patch("psutil.process_iter")
    def test_get_service_processes(self, mock_process_iter, recovery_manager):
        """Test getting service processes."""
        # Mock processes
        mock_proc1 = MagicMock()
        mock_proc1.info = {
            "pid": 1234,
            "name": "uvicorn",
            "cmdline": ["uvicorn", "main:app"],
        }

        mock_proc2 = MagicMock()
        mock_proc2.info = {
            "pid": 5678,
            "name": "celery",
            "cmdline": ["celery", "worker"],
        }

        mock_process_iter.return_value = [mock_proc1, mock_proc2]

        processes = recovery_manager.get_service_processes()

        assert "uvicorn" in processes
        assert "celery" in processes
        assert len(processes["uvicorn"]) == 1
        assert len(processes["celery"]) == 1

    @patch("psutil.process_iter")
    def test_check_service_health(self, mock_process_iter, recovery_manager):
        """Test checking service health."""
        # Mock healthy process
        mock_proc = MagicMock()
        mock_proc.info = {"pid": 1234, "name": "uvicorn", "cmdline": ["uvicorn"]}
        mock_proc.cpu_percent.return_value = 25.0
        mock_proc.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)  # 100MB
        mock_proc.status.return_value = "running"
        mock_proc.create_time.return_value = time.time() - 3600

        mock_process_iter.return_value = [mock_proc]

        health = recovery_manager.check_service_health("uvicorn")

        assert health["status"] == "healthy"
        assert health["process_count"] == 1
        assert len(health["healthy_processes"]) == 1
        assert len(health["unhealthy_processes"]) == 0


class TestHealthEndpoints:
    """Test health check API endpoints."""

    def test_basic_health_check(self):
        """Test basic health check endpoint."""
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_liveness_probe(self):
        """Test Kubernetes liveness probe endpoint."""
        response = client.get("/health/liveness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data

    @patch("src.monitoring.health_checks.ServiceHealthChecker.check_database_health")
    @patch("src.monitoring.health_checks.ServiceHealthChecker.check_redis_health")
    def test_readiness_probe_healthy(self, mock_redis_health, mock_db_health):
        """Test Kubernetes readiness probe when services are healthy."""
        mock_db_health.return_value = {"status": "healthy"}
        mock_redis_health.return_value = {"status": "healthy"}

        response = client.get("/health/readiness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    @patch("src.monitoring.health_checks.ServiceHealthChecker.check_database_health")
    @patch("src.monitoring.health_checks.ServiceHealthChecker.check_redis_health")
    def test_readiness_probe_unhealthy(self, mock_redis_health, mock_db_health):
        """Test Kubernetes readiness probe when services are unhealthy."""
        mock_db_health.return_value = {"status": "unhealthy"}
        mock_redis_health.return_value = {"status": "healthy"}

        response = client.get("/health/readiness")
        assert response.status_code == 503

    def test_services_status(self):
        """Test services status endpoint."""
        with patch(
            "src.monitoring.service_recovery.ServiceRecoveryManager.check_service_health"
        ) as mock_check:
            mock_check.return_value = {"status": "healthy", "process_count": 1}

            response = client.get("/health/services")
            assert response.status_code == 200
            data = response.json()
            assert "services" in data
            assert "timestamp" in data


if __name__ == "__main__":
    pytest.main([__file__])
