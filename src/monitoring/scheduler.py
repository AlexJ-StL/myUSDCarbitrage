"""Background task scheduler for monitoring and metrics collection."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from ..api.database import SessionLocal
from .metrics_collector import MetricsCollector
from .service_recovery import ServiceRecoveryManager
from .business_metrics import BusinessMetricsCollector
from .alerting import AlertingSystem
from .health_checks import ServiceHealthChecker
from .log_aggregation import LogAggregator
from .log_aggregation import LogAggregator

logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """Schedules and manages background monitoring tasks."""

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        recovery_manager: Optional[ServiceRecoveryManager] = None,
        business_metrics: Optional[BusinessMetricsCollector] = None,
        alerting_system: Optional[AlertingSystem] = None,
        health_checker: Optional[ServiceHealthChecker] = None,
        log_aggregator: Optional[LogAggregator] = None,
    ):
        """Initialize monitoring scheduler."""
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.recovery_manager = recovery_manager or ServiceRecoveryManager()
        self.business_metrics = business_metrics or BusinessMetricsCollector()
        self.alerting_system = alerting_system or AlertingSystem()
        self.health_checker = health_checker or ServiceHealthChecker()
        self.log_aggregator = LogAggregator()
        self.running = False
        self.tasks = []

    async def collect_metrics_task(self):
        """Background task to collect metrics periodically."""
        while self.running:
            try:
                # Create database session
                db = SessionLocal()
                try:
                    await self.metrics_collector.collect_and_store_all_metrics(db)
                    logger.debug("Successfully collected and stored metrics")
                except Exception as e:
                    logger.error(f"Failed to collect metrics: {e}")
                finally:
                    db.close()

                # Wait 60 seconds before next collection
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                logger.info("Metrics collection task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in metrics collection task: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def service_recovery_task(self):
        """Background task to check and recover services periodically."""
        while self.running:
            try:
                results = await self.recovery_manager.check_and_recover_services()

                # Log any recovery actions
                for service_name, service_result in results.get("services", {}).items():
                    recovery = service_result.get("recovery", {})
                    if (
                        recovery.get("success")
                        and recovery.get("reason") != "not_needed"
                    ):
                        logger.info(f"Successfully recovered service: {service_name}")
                    elif (
                        not recovery.get("success")
                        and recovery.get("reason") != "not_needed"
                    ):
                        logger.warning(
                            f"Failed to recover service {service_name}: {recovery.get('reason')}"
                        )

                # Wait 5 minutes before next check
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                logger.info("Service recovery task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in service recovery task: {e}")
                await asyncio.sleep(300)  # Wait before retrying

    async def cleanup_old_metrics_task(self):
        """Background task to clean up old metrics data."""
        while self.running:
            try:
                # Clean up metrics older than retention period
                import time

                cutoff_time = time.time() - (
                    self.metrics_collector.metrics_retention_hours * 3600
                )

                # Clean up from Redis
                for metric_type in ["system", "database", "redis"]:
                    try:
                        removed_count = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.metrics_collector.redis_client.zremrangebyscore(
                                f"metrics_index:{metric_type}", 0, cutoff_time
                            ),
                        )
                        if removed_count > 0:
                            logger.debug(
                                f"Cleaned up {removed_count} old {metric_type} metrics"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {metric_type} metrics: {e}")

                # Wait 1 hour before next cleanup
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                logger.info("Metrics cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in metrics cleanup task: {e}")
                await asyncio.sleep(3600)  # Wait before retrying

    async def business_metrics_task(self):
        """Background task to collect business metrics periodically."""
        while self.running:
            try:
                # Create database session
                db = SessionLocal()
                try:
                    await self.business_metrics.collect_and_store_all_business_metrics(
                        db
                    )
                    logger.debug("Successfully collected and stored business metrics")
                except Exception as e:
                    logger.error(f"Failed to collect business metrics: {e}")
                finally:
                    db.close()

                # Wait 5 minutes before next collection
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                logger.info("Business metrics collection task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in business metrics task: {e}")
                await asyncio.sleep(300)  # Wait before retrying

    async def alerting_task(self):
        """Background task to check for alerts periodically."""
        while self.running:
            try:
                # Create database session
                db = SessionLocal()
                try:
                    # Get current system health
                    health_status = (
                        await self.health_checker.comprehensive_health_check(db)
                    )

                    # Check for various types of alerts
                    await self.alerting_system.check_system_health_alerts(
                        health_status.get("checks", {})
                    )
                    await self.alerting_system.check_data_pipeline_alerts(db)
                    await self.alerting_system.check_strategy_performance_alerts(db)

                    # Clean up old alerts
                    await self.alerting_system.cleanup_old_alerts()

                    logger.debug("Successfully completed alert checks")
                except Exception as e:
                    logger.error(f"Failed to check alerts: {e}")
                finally:
                    db.close()

                # Wait 2 minutes before next check
                await asyncio.sleep(120)

            except asyncio.CancelledError:
                logger.info("Alerting task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in alerting task: {e}")
                await asyncio.sleep(120)  # Wait before retrying

    async def start(self):
        """Start all monitoring background tasks."""
        if self.running:
            logger.warning("Monitoring scheduler is already running")
            return

        self.running = True
        logger.info("Starting monitoring scheduler...")

        # Create and start background tasks
        self.tasks = [
            asyncio.create_task(self.collect_metrics_task()),
            asyncio.create_task(self.service_recovery_task()),
            asyncio.create_task(self.cleanup_old_metrics_task()),
            asyncio.create_task(self.business_metrics_task()),
            asyncio.create_task(self.alerting_task()),
        ]

        logger.info(f"Started {len(self.tasks)} monitoring tasks")

    async def stop(self):
        """Stop all monitoring background tasks."""
        if not self.running:
            logger.warning("Monitoring scheduler is not running")
            return

        logger.info("Stopping monitoring scheduler...")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        self.tasks.clear()
        logger.info("Monitoring scheduler stopped")

    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        return self.running and any(not task.done() for task in self.tasks)


# Global scheduler instance
monitoring_scheduler = MonitoringScheduler()


async def start_monitoring():
    """Start the monitoring scheduler."""
    await monitoring_scheduler.start()


async def stop_monitoring():
    """Stop the monitoring scheduler."""
    await monitoring_scheduler.stop()


def get_monitoring_scheduler() -> MonitoringScheduler:
    """Get the monitoring scheduler instance."""
    return monitoring_scheduler
