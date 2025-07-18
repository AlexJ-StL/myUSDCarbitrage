"""Tests for business metrics and alerting system."""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.database import Base, get_db
from src.api.main import app
from src.monitoring.alerting import AlertingSystem, Alert, AlertType, AlertSeverity
from src.monitoring.business_metrics import BusinessMetricsCollector
from src.monitoring.reporting import ReportGenerator


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_business_metrics.db"
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


class TestAlertingSystem:
    """Test alerting system functionality."""

    @pytest.fixture
    def alerting_system(self):
        """Create alerting system instance for testing."""
        return AlertingSystem(redis_url="redis://localhost:6379/2")

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert for testing."""
        return Alert(
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="This is a test alert",
            source="test_system",
            metadata={"test_key": "test_value"},
        )

    @pytest.mark.asyncio
    @patch("redis.from_url")
    async def test_create_alert(
        self, mock_redis_from_url, alerting_system, sample_alert
    ):
        """Test creating and storing an alert."""
        mock_redis = MagicMock()
        mock_redis.setex.return_value = True
        mock_redis.zadd.return_value = 1
        mock_redis_from_url.return_value = mock_redis

        # Reset the cached client
        alerting_system._redis_client = None

        result = await alerting_system.create_alert(sample_alert)

        assert result is True
        assert mock_redis.setex.called
        assert mock_redis.zadd.called

    @pytest.mark.asyncio
    @patch("redis.from_url")
    async def test_get_alerts(self, mock_redis_from_url, alerting_system):
        """Test retrieving alerts with filtering."""
        mock_redis = MagicMock()

        # Mock alert data
        alert_data = {
            "alert_id": "test_alert_123",
            "alert_type": "system_health",
            "severity": "high",
            "title": "Test Alert",
            "message": "Test message",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        mock_redis.zrangebyscore.return_value = ["alert:test_alert_123"]
        mock_redis.get.return_value = json.dumps(alert_data)
        mock_redis_from_url.return_value = mock_redis

        # Reset the cached client
        alerting_system._redis_client = None

        alerts = await alerting_system.get_alerts(
            severity=AlertSeverity.HIGH, hours=24, limit=10
        )

        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "test_alert_123"
        assert alerts[0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_check_system_health_alerts(self, alerting_system):
        """Test system health alert checking."""
        # Mock health data with high CPU usage
        health_data = {
            "system": {
                "cpu": {"usage_percent": 90.0},
                "memory": {"usage_percent": 70.0},
                "disk": {"usage_percent": 80.0},
            },
            "database": {"response_time_ms": 100.0},
            "redis": {"response_time_ms": 50.0},
        }

        with patch.object(alerting_system, "create_alert") as mock_create_alert:
            mock_create_alert.return_value = True

            await alerting_system.check_system_health_alerts(health_data)

            # Should create alert for high CPU usage
            mock_create_alert.assert_called()
            call_args = mock_create_alert.call_args[0][0]
            assert call_args.alert_type == AlertType.SYSTEM_HEALTH
            assert "CPU" in call_args.title

    @pytest.mark.asyncio
    async def test_check_data_pipeline_alerts(self, alerting_system):
        """Test data pipeline alert checking."""
        mock_db = MagicMock()

        # Mock data gap query result
        gap_result = MagicMock()
        gap_result.exchange = "binance"
        gap_result.symbol = "USDC"
        gap_result.timeframe = "1h"
        gap_result.hours_since_update = 3.0
        gap_result.last_update = datetime.now(timezone.utc) - timedelta(hours=3)

        mock_db.execute.return_value.fetchall.return_value = [gap_result]

        with patch.object(alerting_system, "create_alert") as mock_create_alert:
            mock_create_alert.return_value = True

            await alerting_system.check_data_pipeline_alerts(mock_db)

            # Should create alert for data gap
            mock_create_alert.assert_called()
            call_args = mock_create_alert.call_args[0][0]
            assert call_args.alert_type == AlertType.DATA_PIPELINE
            assert "Gap" in call_args.title


class TestBusinessMetricsCollector:
    """Test business metrics collection functionality."""

    @pytest.fixture
    def metrics_collector(self):
        """Create business metrics collector instance for testing."""
        return BusinessMetricsCollector(redis_url="redis://localhost:6379/2")

    @pytest.mark.asyncio
    async def test_collect_backtest_metrics(self, metrics_collector):
        """Test backtest metrics collection."""
        mock_db = MagicMock()

        # Mock backtest statistics
        backtest_stats = MagicMock()
        backtest_stats.total_backtests = 100
        backtest_stats.successful_backtests = 85
        backtest_stats.failed_backtests = 15
        backtest_stats.avg_return = 0.05
        backtest_stats.avg_sharpe = 1.2
        backtest_stats.avg_drawdown = 0.08
        backtest_stats.total_trades = 1500

        # Mock strategy performance
        strategy_perf = MagicMock()
        strategy_perf.strategy_name = "Test Strategy"
        strategy_perf.backtest_count = 10
        strategy_perf.avg_return = 0.08
        strategy_perf.avg_sharpe = 1.5
        strategy_perf.best_return = 0.15
        strategy_perf.worst_return = -0.02

        mock_db.execute.return_value.fetchone.return_value = backtest_stats
        mock_db.execute.return_value.fetchall.return_value = [strategy_perf]

        metrics = await metrics_collector.collect_backtest_metrics(mock_db)

        assert "timestamp" in metrics
        assert "periods" in metrics
        assert "24h" in metrics["periods"]

        metrics_24h = metrics["periods"]["24h"]
        assert metrics_24h["total_backtests"] == 100
        assert metrics_24h["success_rate"] == 85.0
        assert len(metrics_24h["top_strategies"]) == 1
        assert metrics_24h["top_strategies"][0]["name"] == "Test Strategy"

    @pytest.mark.asyncio
    async def test_collect_data_pipeline_metrics(self, metrics_collector):
        """Test data pipeline metrics collection."""
        mock_db = MagicMock()

        # Mock data statistics
        data_stat = MagicMock()
        data_stat.exchange = "binance"
        data_stat.symbol = "USDC"
        data_stat.timeframe = "1h"
        data_stat.record_count = 1000
        data_stat.latest_data = datetime.now(timezone.utc)
        data_stat.earliest_data = datetime.now(timezone.utc) - timedelta(hours=24)
        data_stat.avg_quality_score = 0.95
        data_stat.low_quality_count = 10

        # Mock quality summary
        quality_summary = MagicMock()
        quality_summary.overall_avg_quality = 0.92
        quality_summary.high_quality_count = 900
        quality_summary.medium_quality_count = 80
        quality_summary.low_quality_count = 20
        quality_summary.total_records = 1000

        mock_db.execute.return_value.fetchall.return_value = [data_stat]
        mock_db.execute.return_value.fetchone.return_value = quality_summary

        metrics = await metrics_collector.collect_data_pipeline_metrics(mock_db)

        assert "timestamp" in metrics
        assert "data_sources" in metrics
        assert "quality_summary" in metrics

        assert len(metrics["data_sources"]) == 1
        assert metrics["data_sources"][0]["exchange"] == "binance"
        assert metrics["quality_summary"]["total_records"] == 1000
        assert metrics["quality_summary"]["high_quality_percentage"] == 90.0

    @pytest.mark.asyncio
    @patch("redis.from_url")
    async def test_store_and_retrieve_business_metrics(
        self, mock_redis_from_url, metrics_collector
    ):
        """Test storing and retrieving business metrics."""
        mock_redis = MagicMock()
        mock_redis.setex.return_value = True
        mock_redis.zadd.return_value = 1
        mock_redis.zremrangebyscore.return_value = 0
        mock_redis.zrangebyscore.return_value = ["business_metrics:test:123456"]

        test_metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_data": "value",
        }
        mock_redis.get.return_value = json.dumps(test_metrics)
        mock_redis_from_url.return_value = mock_redis

        # Reset the cached client
        metrics_collector._redis_client = None

        # Test storing metrics
        await metrics_collector.store_business_metrics(test_metrics, "test")

        # Test retrieving metrics
        history = await metrics_collector.get_business_metrics_history("test", days=1)

        assert len(history) == 1
        assert history[0]["test_data"] == "value"


class TestReportGenerator:
    """Test report generation functionality."""

    @pytest.fixture
    def report_generator(self):
        """Create report generator instance for testing."""
        return ReportGenerator()

    @pytest.mark.asyncio
    async def test_generate_daily_summary(self, report_generator):
        """Test daily summary report generation."""
        mock_db = MagicMock()

        # Mock business metrics
        with (
            patch.object(
                report_generator.business_metrics, "collect_backtest_metrics"
            ) as mock_backtest,
            patch.object(
                report_generator.business_metrics, "collect_data_pipeline_metrics"
            ) as mock_pipeline,
            patch.object(
                report_generator.business_metrics, "collect_user_activity_metrics"
            ) as mock_activity,
            patch.object(report_generator.alerting_system, "get_alerts") as mock_alerts,
        ):
            # Mock return values
            mock_backtest.return_value = {
                "periods": {
                    "24h": {
                        "total_backtests": 50,
                        "success_rate": 90.0,
                        "avg_return": 0.05,
                        "top_strategies": [{"name": "Strategy1", "avg_return": 0.08}],
                    }
                }
            }

            mock_pipeline.return_value = {
                "quality_summary": {
                    "total_records": 1000,
                    "high_quality_percentage": 95.0,
                },
                "gaps_count": 2,
                "data_sources": [{"exchange": "binance"}],
            }

            mock_activity.return_value = {
                "user_activity": {"active_users_24h": 10, "total_requests_24h": 500}
            }

            mock_alerts.return_value = [
                {"severity": "high", "title": "Test Alert"},
                {"severity": "medium", "title": "Another Alert"},
            ]

            report = await report_generator.generate_daily_summary(mock_db)

            assert report["report_type"] == "daily_summary"
            assert "executive_summary" in report
            assert "backtest_performance" in report
            assert "data_pipeline" in report
            assert "alerts_summary" in report

            exec_summary = report["executive_summary"]
            assert exec_summary["total_backtests"] == 50
            assert exec_summary["success_rate"] == 90.0
            assert exec_summary["active_users"] == 10

    def test_format_report_as_text(self, report_generator):
        """Test formatting report as plain text."""
        sample_report = {
            "report_type": "daily_summary",
            "report_date": "2023-01-01",
            "generated_at": "2023-01-01T12:00:00Z",
            "executive_summary": {
                "total_backtests": 100,
                "success_rate": 85.0,
                "avg_return": 5.5,
                "active_users": 15,
                "data_quality": 92.0,
                "critical_alerts": 2,
            },
            "backtest_performance": {
                "successful_backtests": 85,
                "total_backtests": 100,
                "avg_sharpe": 1.2,
                "avg_drawdown": 8.5,
                "total_trades": 1500,
                "top_strategies": [
                    {"name": "Strategy A", "avg_return": 7.2},
                    {"name": "Strategy B", "avg_return": 6.8},
                ],
            },
            "data_pipeline": {
                "total_records": 10000,
                "high_quality_percentage": 92.0,
                "data_gaps_count": 3,
            },
            "alerts_summary": {
                "total_alerts": 15,
                "by_severity": {"critical": 2, "high": 5},
                "recent_critical": [],
            },
            "trends": {"backtests_change": 10.5, "success_rate_change": -2.1},
        }

        text_report = report_generator.format_report_as_text(sample_report)

        assert "Daily Summary Report" in text_report
        assert "Total Backtests: 100" in text_report
        assert "Success Rate: 85.0%" in text_report
        assert "Strategy A: 7.20% return" in text_report


class TestMonitoringEndpoints:
    """Test monitoring API endpoints."""

    def test_get_backtest_metrics(self):
        """Test backtest metrics endpoint."""
        with patch(
            "src.monitoring.business_metrics.BusinessMetricsCollector.collect_backtest_metrics"
        ) as mock_collect:
            mock_collect.return_value = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "periods": {"24h": {"total_backtests": 50}},
            }

            response = client.get("/monitoring/business-metrics/backtests")
            assert response.status_code == 200
            data = response.json()
            assert "timestamp" in data
            assert "periods" in data

    def test_create_alert(self):
        """Test alert creation endpoint."""
        alert_data = {
            "alert_type": "system_health",
            "severity": "high",
            "title": "Test Alert",
            "message": "This is a test alert",
            "source": "test_system",
            "metadata": {"test": "value"},
        }

        with patch(
            "src.monitoring.alerting.AlertingSystem.create_alert"
        ) as mock_create:
            mock_create.return_value = True

            response = client.post("/monitoring/alerts", json=alert_data)
            assert response.status_code == 200
            data = response.json()
            assert "alert_id" in data
            assert data["message"] == "Alert created successfully"

    def test_get_alerts(self):
        """Test alerts retrieval endpoint."""
        with patch("src.monitoring.alerting.AlertingSystem.get_alerts") as mock_get:
            mock_get.return_value = [
                {
                    "alert_id": "test_123",
                    "severity": "high",
                    "title": "Test Alert",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]

            response = client.get("/monitoring/alerts?severity=high&hours=24")
            assert response.status_code == 200
            data = response.json()
            assert data["total_alerts"] == 1
            assert len(data["alerts"]) == 1

    def test_get_daily_report(self):
        """Test daily report endpoint."""
        with patch(
            "src.monitoring.reporting.ReportGenerator.generate_daily_summary"
        ) as mock_report:
            mock_report.return_value = {
                "report_type": "daily_summary",
                "executive_summary": {"total_backtests": 100},
            }

            response = client.get("/monitoring/reports/daily")
            assert response.status_code == 200
            data = response.json()
            assert data["report_type"] == "daily_summary"

    def test_monitoring_dashboard(self):
        """Test monitoring dashboard endpoint."""
        with (
            patch(
                "src.monitoring.business_metrics.BusinessMetricsCollector.collect_backtest_metrics"
            ) as mock_backtest,
            patch(
                "src.monitoring.business_metrics.BusinessMetricsCollector.collect_data_pipeline_metrics"
            ) as mock_pipeline,
            patch(
                "src.monitoring.business_metrics.BusinessMetricsCollector.collect_user_activity_metrics"
            ) as mock_activity,
            patch("src.monitoring.alerting.AlertingSystem.get_alerts") as mock_alerts,
        ):
            mock_backtest.return_value = {"periods": {"24h": {"total_backtests": 50}}}
            mock_pipeline.return_value = {
                "quality_summary": {"high_quality_percentage": 95}
            }
            mock_activity.return_value = {"user_activity": {"active_users_24h": 10}}
            mock_alerts.return_value = [{"severity": "high"}]

            response = client.get("/monitoring/dashboard")
            assert response.status_code == 200
            data = response.json()
            assert "timestamp" in data
            assert "backtest_metrics" in data
            assert "alert_summary" in data


if __name__ == "__main__":
    pytest.main([__file__])
