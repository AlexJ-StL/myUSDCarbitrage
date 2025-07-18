"""Business metrics monitoring and alerting system."""

import asyncio
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import redis
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""

    SYSTEM_HEALTH = "system_health"
    DATA_PIPELINE = "data_pipeline"
    STRATEGY_PERFORMANCE = "strategy_performance"
    BUSINESS_METRIC = "business_metric"
    SECURITY = "security"


class Alert:
    """Represents an alert with metadata."""

    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize alert."""
        self.alert_type = alert_type
        self.severity = severity
        self.title = title
        self.message = message
        self.source = source
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
        self.alert_id = f"{alert_type.value}_{int(self.timestamp.timestamp())}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class AlertingSystem:
    """Comprehensive alerting system for business metrics and system events."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        smtp_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize alerting system."""
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
        self.smtp_config = smtp_config or {}
        self.alert_retention_hours = 168  # 7 days

        # Alert thresholds
        self.thresholds = {
            "cpu_usage": 85.0,
            "memory_usage": 90.0,
            "disk_usage": 95.0,
            "database_response_time": 5000,  # ms
            "redis_response_time": 2000,  # ms
            "strategy_drawdown": 0.15,  # 15%
            "data_gap_hours": 2,
            "failed_backtests_per_hour": 10,
        }

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    async def create_alert(self, alert: Alert) -> bool:
        """Create and store a new alert."""
        try:
            # Store alert in Redis
            alert_key = f"alert:{alert.alert_id}"
            import json

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.setex(
                    alert_key,
                    self.alert_retention_hours * 3600,
                    json.dumps(alert.to_dict(), default=str),
                ),
            )

            # Add to alert index
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zadd(
                    "alerts_index", {alert_key: alert.timestamp.timestamp()}
                ),
            )

            # Add to severity-specific index
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zadd(
                    f"alerts_severity:{alert.severity.value}",
                    {alert_key: alert.timestamp.timestamp()},
                ),
            )

            # Send alert notifications
            await self._send_alert_notifications(alert)

            logger.info(f"Created alert: {alert.alert_id} - {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return False

    async def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve alerts based on filters."""
        try:
            import time
            import json

            start_time = time.time() - (hours * 3600)
            end_time = time.time()

            # Choose appropriate index
            if severity:
                index_key = f"alerts_severity:{severity.value}"
            else:
                index_key = "alerts_index"

            # Get alert keys from index
            alert_keys = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zrangebyscore(
                    index_key,
                    start_time,
                    end_time,
                    start=0,
                    num=limit,
                    desc=True,  # Most recent first
                ),
            )

            if not alert_keys:
                return []

            # Get alert data
            alerts = []
            for key in alert_keys:
                alert_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                if alert_data:
                    try:
                        alert_dict = json.loads(alert_data)

                        # Filter by alert type if specified
                        if (
                            alert_type
                            and alert_dict.get("alert_type") != alert_type.value
                        ):
                            continue

                        alerts.append(alert_dict)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode alert data for key: {key}")

            return alerts

        except Exception as e:
            logger.error(f"Failed to retrieve alerts: {e}")
            return []

    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications via configured channels."""
        try:
            # Send email notification for high/critical alerts
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                await self._send_email_alert(alert)

            # Log all alerts
            logger.warning(
                f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}"
            )

        except Exception as e:
            logger.error(f"Failed to send alert notifications: {e}")

    async def _send_email_alert(self, alert: Alert):
        """Send email alert notification."""
        if not self.smtp_config:
            logger.debug("SMTP not configured, skipping email alert")
            return

        try:
            # Create email message
            msg = MIMEMultipart()
            msg["From"] = self.smtp_config.get("from_email", "alerts@usdcarbitrage.com")
            msg["To"] = self.smtp_config.get("to_email", "admin@usdcarbitrage.com")
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Email body
            body = f"""
Alert Details:
- Type: {alert.alert_type.value}
- Severity: {alert.severity.value}
- Source: {alert.source}
- Time: {alert.timestamp.isoformat()}

Message:
{alert.message}

Metadata:
{alert.metadata}

Alert ID: {alert.alert_id}
            """

            msg.attach(MIMEText(body, "plain"))

            # Send email
            server = smtplib.SMTP(
                self.smtp_config.get("smtp_server", "localhost"),
                self.smtp_config.get("smtp_port", 587),
            )

            if self.smtp_config.get("use_tls", True):
                server.starttls()

            if self.smtp_config.get("username") and self.smtp_config.get("password"):
                server.login(self.smtp_config["username"], self.smtp_config["password"])

            server.send_message(msg)
            server.quit()

            logger.info(f"Email alert sent for: {alert.alert_id}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def check_system_health_alerts(self, health_data: Dict[str, Any]):
        """Check system health metrics and create alerts if needed."""
        try:
            # Check system resources
            system_data = health_data.get("system", {})

            # CPU usage alert
            cpu_usage = system_data.get("cpu", {}).get("usage_percent", 0)
            if cpu_usage > self.thresholds["cpu_usage"]:
                await self.create_alert(
                    Alert(
                        alert_type=AlertType.SYSTEM_HEALTH,
                        severity=AlertSeverity.HIGH
                        if cpu_usage > 95
                        else AlertSeverity.MEDIUM,
                        title="High CPU Usage",
                        message=f"CPU usage is {cpu_usage:.1f}%, exceeding threshold of {self.thresholds['cpu_usage']}%",
                        source="system_monitor",
                        metadata={
                            "cpu_usage": cpu_usage,
                            "threshold": self.thresholds["cpu_usage"],
                        },
                    )
                )

            # Memory usage alert
            memory_usage = system_data.get("memory", {}).get("usage_percent", 0)
            if memory_usage > self.thresholds["memory_usage"]:
                await self.create_alert(
                    Alert(
                        alert_type=AlertType.SYSTEM_HEALTH,
                        severity=AlertSeverity.HIGH
                        if memory_usage > 95
                        else AlertSeverity.MEDIUM,
                        title="High Memory Usage",
                        message=f"Memory usage is {memory_usage:.1f}%, exceeding threshold of {self.thresholds['memory_usage']}%",
                        source="system_monitor",
                        metadata={
                            "memory_usage": memory_usage,
                            "threshold": self.thresholds["memory_usage"],
                        },
                    )
                )

            # Disk usage alert
            disk_usage = system_data.get("disk", {}).get("usage_percent", 0)
            if disk_usage > self.thresholds["disk_usage"]:
                await self.create_alert(
                    Alert(
                        alert_type=AlertType.SYSTEM_HEALTH,
                        severity=AlertSeverity.CRITICAL
                        if disk_usage > 98
                        else AlertSeverity.HIGH,
                        title="High Disk Usage",
                        message=f"Disk usage is {disk_usage:.1f}%, exceeding threshold of {self.thresholds['disk_usage']}%",
                        source="system_monitor",
                        metadata={
                            "disk_usage": disk_usage,
                            "threshold": self.thresholds["disk_usage"],
                        },
                    )
                )

            # Database response time alert
            db_data = health_data.get("database", {})
            db_response_time = db_data.get("response_time_ms", 0)
            if db_response_time > self.thresholds["database_response_time"]:
                await self.create_alert(
                    Alert(
                        alert_type=AlertType.SYSTEM_HEALTH,
                        severity=AlertSeverity.HIGH,
                        title="Slow Database Response",
                        message=f"Database response time is {db_response_time:.1f}ms, exceeding threshold of {self.thresholds['database_response_time']}ms",
                        source="database_monitor",
                        metadata={
                            "response_time_ms": db_response_time,
                            "threshold": self.thresholds["database_response_time"],
                        },
                    )
                )

            # Redis response time alert
            redis_data = health_data.get("redis", {})
            redis_response_time = redis_data.get("response_time_ms", 0)
            if redis_response_time > self.thresholds["redis_response_time"]:
                await self.create_alert(
                    Alert(
                        alert_type=AlertType.SYSTEM_HEALTH,
                        severity=AlertSeverity.MEDIUM,
                        title="Slow Redis Response",
                        message=f"Redis response time is {redis_response_time:.1f}ms, exceeding threshold of {self.thresholds['redis_response_time']}ms",
                        source="redis_monitor",
                        metadata={
                            "response_time_ms": redis_response_time,
                            "threshold": self.thresholds["redis_response_time"],
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Failed to check system health alerts: {e}")

    async def check_data_pipeline_alerts(self, db: Session):
        """Check data pipeline health and create alerts for issues."""
        try:
            # Check for data gaps
            gap_query = text("""
                SELECT 
                    exchange,
                    symbol,
                    timeframe,
                    MAX(timestamp) as last_update,
                    EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/3600 as hours_since_update
                FROM market_data 
                GROUP BY exchange, symbol, timeframe
                HAVING EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/3600 > :threshold
            """)

            gaps = db.execute(
                gap_query, {"threshold": self.thresholds["data_gap_hours"]}
            ).fetchall()

            for gap in gaps:
                await self.create_alert(
                    Alert(
                        alert_type=AlertType.DATA_PIPELINE,
                        severity=AlertSeverity.HIGH
                        if gap.hours_since_update > 6
                        else AlertSeverity.MEDIUM,
                        title="Data Pipeline Gap Detected",
                        message=f"No data received for {gap.exchange} {gap.symbol} {gap.timeframe} for {gap.hours_since_update:.1f} hours",
                        source="data_pipeline_monitor",
                        metadata={
                            "exchange": gap.exchange,
                            "symbol": gap.symbol,
                            "timeframe": gap.timeframe,
                            "hours_since_update": gap.hours_since_update,
                            "last_update": str(gap.last_update),
                        },
                    )
                )

            # Check for data quality issues
            quality_query = text("""
                SELECT 
                    exchange,
                    symbol,
                    COUNT(*) as low_quality_count,
                    AVG(quality_score) as avg_quality_score
                FROM market_data 
                WHERE created_at > NOW() - INTERVAL '1 hour'
                AND quality_score < 0.8
                GROUP BY exchange, symbol
                HAVING COUNT(*) > 10
            """)

            quality_issues = db.execute(quality_query).fetchall()

            for issue in quality_issues:
                await self.create_alert(
                    Alert(
                        alert_type=AlertType.DATA_PIPELINE,
                        severity=AlertSeverity.MEDIUM,
                        title="Data Quality Issues Detected",
                        message=f"Found {issue.low_quality_count} low-quality records for {issue.exchange} {issue.symbol} in the last hour",
                        source="data_quality_monitor",
                        metadata={
                            "exchange": issue.exchange,
                            "symbol": issue.symbol,
                            "low_quality_count": issue.low_quality_count,
                            "avg_quality_score": float(issue.avg_quality_score),
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Failed to check data pipeline alerts: {e}")

    async def check_strategy_performance_alerts(self, db: Session):
        """Check strategy performance and create alerts for deviations."""
        try:
            # Check for strategies with high drawdown
            drawdown_query = text("""
                SELECT 
                    s.name as strategy_name,
                    br.max_drawdown,
                    br.created_at,
                    br.total_return
                FROM backtest_results br
                JOIN strategies s ON br.strategy_id = s.id
                WHERE br.created_at > NOW() - INTERVAL '24 hours'
                AND br.max_drawdown > :threshold
                ORDER BY br.max_drawdown DESC
            """)

            high_drawdowns = db.execute(
                drawdown_query, {"threshold": self.thresholds["strategy_drawdown"]}
            ).fetchall()

            for result in high_drawdowns:
                await self.create_alert(
                    Alert(
                        alert_type=AlertType.STRATEGY_PERFORMANCE,
                        severity=AlertSeverity.HIGH
                        if result.max_drawdown > 0.25
                        else AlertSeverity.MEDIUM,
                        title="High Strategy Drawdown",
                        message=f"Strategy '{result.strategy_name}' has drawdown of {result.max_drawdown:.1%}, exceeding threshold of {self.thresholds['strategy_drawdown']:.1%}",
                        source="strategy_monitor",
                        metadata={
                            "strategy_name": result.strategy_name,
                            "max_drawdown": float(result.max_drawdown),
                            "total_return": float(result.total_return),
                            "backtest_date": str(result.created_at),
                        },
                    )
                )

            # Check for failed backtests
            failed_query = text("""
                SELECT 
                    COUNT(*) as failed_count
                FROM backtest_results br
                WHERE br.created_at > NOW() - INTERVAL '1 hour'
                AND br.total_return IS NULL
            """)

            failed_result = db.execute(failed_query).scalar()

            if (
                failed_result
                and failed_result > self.thresholds["failed_backtests_per_hour"]
            ):
                await self.create_alert(
                    Alert(
                        alert_type=AlertType.STRATEGY_PERFORMANCE,
                        severity=AlertSeverity.HIGH,
                        title="High Backtest Failure Rate",
                        message=f"Found {failed_result} failed backtests in the last hour, exceeding threshold of {self.thresholds['failed_backtests_per_hour']}",
                        source="backtest_monitor",
                        metadata={
                            "failed_count": failed_result,
                            "threshold": self.thresholds["failed_backtests_per_hour"],
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Failed to check strategy performance alerts: {e}")

    async def cleanup_old_alerts(self):
        """Clean up old alerts beyond retention period."""
        try:
            import time

            cutoff_time = time.time() - (self.alert_retention_hours * 3600)

            # Clean up from main index
            removed_count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zremrangebyscore(
                    "alerts_index", 0, cutoff_time
                ),
            )

            # Clean up from severity indexes
            for severity in AlertSeverity:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.redis_client.zremrangebyscore(
                        f"alerts_severity:{severity.value}", 0, cutoff_time
                    ),
                )

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old alerts")

        except Exception as e:
            logger.error(f"Failed to cleanup old alerts: {e}")


# Global alerting system instance
alerting_system = AlertingSystem()


async def get_alerting_system() -> AlertingSystem:
    """Dependency to get alerting system instance."""
    return alerting_system
