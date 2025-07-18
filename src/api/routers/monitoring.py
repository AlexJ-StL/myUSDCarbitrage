"""Business metrics and alerting API endpoints."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ...monitoring.alerting import (
    AlertingSystem,
    Alert,
    AlertType,
    AlertSeverity,
    get_alerting_system,
)
from ...monitoring.business_metrics import (
    BusinessMetricsCollector,
    get_business_metrics_collector,
)
from ...monitoring.reporting import ReportGenerator, get_report_generator

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


class AlertCreate(BaseModel):
    """Model for creating alerts."""

    alert_type: str
    severity: str
    title: str
    message: str
    source: str
    metadata: Optional[Dict[str, Any]] = None


class AlertResponse(BaseModel):
    """Model for alert responses."""

    alert_id: str
    alert_type: str
    severity: str
    title: str
    message: str
    source: str
    metadata: Dict[str, Any]
    timestamp: str


@router.get("/business-metrics/backtests")
async def get_backtest_metrics(
    db: Session = Depends(get_db),
    collector: BusinessMetricsCollector = Depends(get_business_metrics_collector),
):
    """Get current backtesting performance metrics."""
    try:
        metrics = await collector.collect_backtest_metrics(db)
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to collect backtest metrics: {str(e)}"
        )


@router.get("/business-metrics/data-pipeline")
async def get_data_pipeline_metrics(
    db: Session = Depends(get_db),
    collector: BusinessMetricsCollector = Depends(get_business_metrics_collector),
):
    """Get data pipeline performance metrics."""
    try:
        metrics = await collector.collect_data_pipeline_metrics(db)
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to collect data pipeline metrics: {str(e)}"
        )


@router.get("/business-metrics/user-activity")
async def get_user_activity_metrics(
    db: Session = Depends(get_db),
    collector: BusinessMetricsCollector = Depends(get_business_metrics_collector),
):
    """Get user activity and API usage metrics."""
    try:
        metrics = await collector.collect_user_activity_metrics(db)
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to collect user activity metrics: {str(e)}"
        )


@router.get("/business-metrics/history")
async def get_business_metrics_history(
    metric_type: str = Query(
        ..., description="Type of metrics (backtests, data_pipeline, user_activity)"
    ),
    days: int = Query(7, description="Number of days of history", ge=1, le=30),
    limit: int = Query(100, description="Maximum number of data points", ge=1, le=500),
    collector: BusinessMetricsCollector = Depends(get_business_metrics_collector),
):
    """Get historical business metrics."""
    try:
        history = await collector.get_business_metrics_history(
            metric_type=metric_type, days=days, limit=limit
        )

        return {
            "metric_type": metric_type,
            "days": days,
            "data_points": len(history),
            "data": history,
            "query_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve business metrics history: {str(e)}",
        )


@router.post("/alerts")
async def create_alert(
    alert_data: AlertCreate,
    alerting_system: AlertingSystem = Depends(get_alerting_system),
):
    """Create a new alert."""
    try:
        # Validate alert type and severity
        try:
            alert_type = AlertType(alert_data.alert_type)
            severity = AlertSeverity(alert_data.severity)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid alert type or severity: {str(e)}"
            )

        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            title=alert_data.title,
            message=alert_data.message,
            source=alert_data.source,
            metadata=alert_data.metadata,
        )

        success = await alerting_system.create_alert(alert)

        if success:
            return {
                "message": "Alert created successfully",
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create alert")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")


@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = Query(
        None, description="Filter by severity (low, medium, high, critical)"
    ),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    hours: int = Query(24, description="Number of hours to look back", ge=1, le=168),
    limit: int = Query(100, description="Maximum number of alerts", ge=1, le=500),
    alerting_system: AlertingSystem = Depends(get_alerting_system),
):
    """Get alerts with optional filtering."""
    try:
        # Validate parameters
        severity_enum = None
        if severity:
            try:
                severity_enum = AlertSeverity(severity)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid severity: {severity}"
                )

        alert_type_enum = None
        if alert_type:
            try:
                alert_type_enum = AlertType(alert_type)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid alert type: {alert_type}"
                )

        alerts = await alerting_system.get_alerts(
            severity=severity_enum, alert_type=alert_type_enum, hours=hours, limit=limit
        )

        return {
            "total_alerts": len(alerts),
            "filters": {
                "severity": severity,
                "alert_type": alert_type,
                "hours": hours,
            },
            "alerts": alerts,
            "query_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve alerts: {str(e)}"
        )


@router.get("/alerts/summary")
async def get_alerts_summary(
    hours: int = Query(24, description="Number of hours to analyze", ge=1, le=168),
    alerting_system: AlertingSystem = Depends(get_alerting_system),
):
    """Get alerts summary with statistics."""
    try:
        alerts = await alerting_system.get_alerts(hours=hours, limit=1000)

        # Calculate summary statistics
        by_severity = {}
        by_type = {}
        by_source = {}

        for alert in alerts:
            severity = alert.get("severity", "unknown")
            alert_type = alert.get("alert_type", "unknown")
            source = alert.get("source", "unknown")

            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1

        # Get recent critical alerts
        recent_critical = [
            alert for alert in alerts[:10] if alert.get("severity") == "critical"
        ]

        return {
            "period_hours": hours,
            "total_alerts": len(alerts),
            "by_severity": by_severity,
            "by_type": by_type,
            "by_source": by_source,
            "recent_critical": recent_critical,
            "summary_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate alerts summary: {str(e)}"
        )


@router.get("/reports/daily")
async def get_daily_report(
    db: Session = Depends(get_db),
    report_generator: ReportGenerator = Depends(get_report_generator),
):
    """Generate daily summary report."""
    try:
        report = await report_generator.generate_daily_summary(db)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate daily report: {str(e)}"
        )


@router.get("/reports/weekly")
async def get_weekly_report(
    db: Session = Depends(get_db),
    report_generator: ReportGenerator = Depends(get_report_generator),
):
    """Generate weekly summary report."""
    try:
        report = await report_generator.generate_weekly_summary(db)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate weekly report: {str(e)}"
        )


@router.get("/reports/daily/text")
async def get_daily_report_text(
    db: Session = Depends(get_db),
    report_generator: ReportGenerator = Depends(get_report_generator),
):
    """Generate daily summary report as plain text."""
    try:
        report = await report_generator.generate_daily_summary(db)
        text_report = report_generator.format_report_as_text(report)

        return {
            "report_type": "daily_summary",
            "format": "text",
            "content": text_report,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate daily text report: {str(e)}"
        )


@router.get("/reports/weekly/text")
async def get_weekly_report_text(
    db: Session = Depends(get_db),
    report_generator: ReportGenerator = Depends(get_report_generator),
):
    """Generate weekly summary report as plain text."""
    try:
        report = await report_generator.generate_weekly_summary(db)
        text_report = report_generator.format_report_as_text(report)

        return {
            "report_type": "weekly_summary",
            "format": "text",
            "content": text_report,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate weekly text report: {str(e)}"
        )


@router.post("/check-alerts")
async def trigger_alert_checks(
    db: Session = Depends(get_db),
    alerting_system: AlertingSystem = Depends(get_alerting_system),
):
    """Manually trigger alert checks for all systems."""
    try:
        # This would typically be called by the scheduler
        # but can be triggered manually for testing

        from ...monitoring.health_checks import health_checker

        # Get current health status
        health_status = await health_checker.comprehensive_health_check(db)

        # Check for alerts
        await alerting_system.check_system_health_alerts(
            health_status.get("checks", {})
        )
        await alerting_system.check_data_pipeline_alerts(db)
        await alerting_system.check_strategy_performance_alerts(db)

        return {
            "message": "Alert checks completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to trigger alert checks: {str(e)}"
        )


@router.get("/dashboard")
async def get_monitoring_dashboard(
    db: Session = Depends(get_db),
    collector: BusinessMetricsCollector = Depends(get_business_metrics_collector),
    alerting_system: AlertingSystem = Depends(get_alerting_system),
):
    """Get comprehensive monitoring dashboard data."""
    try:
        # Collect all dashboard data concurrently
        import asyncio

        (
            backtest_metrics,
            pipeline_metrics,
            activity_metrics,
            recent_alerts,
        ) = await asyncio.gather(
            collector.collect_backtest_metrics(db),
            collector.collect_data_pipeline_metrics(db),
            collector.collect_user_activity_metrics(db),
            alerting_system.get_alerts(hours=24, limit=50),
            return_exceptions=True,
        )

        # Handle any exceptions
        dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backtest_metrics": backtest_metrics
            if not isinstance(backtest_metrics, Exception)
            else {"error": str(backtest_metrics)},
            "pipeline_metrics": pipeline_metrics
            if not isinstance(pipeline_metrics, Exception)
            else {"error": str(pipeline_metrics)},
            "activity_metrics": activity_metrics
            if not isinstance(activity_metrics, Exception)
            else {"error": str(activity_metrics)},
            "recent_alerts": recent_alerts
            if not isinstance(recent_alerts, Exception)
            else [],
            "alert_summary": {
                "total": len(recent_alerts)
                if not isinstance(recent_alerts, Exception)
                else 0,
                "critical": len([
                    a
                    for a in (
                        recent_alerts
                        if not isinstance(recent_alerts, Exception)
                        else []
                    )
                    if a.get("severity") == "critical"
                ]),
                "high": len([
                    a
                    for a in (
                        recent_alerts
                        if not isinstance(recent_alerts, Exception)
                        else []
                    )
                    if a.get("severity") == "high"
                ]),
            },
        }

        return dashboard_data

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get monitoring dashboard: {str(e)}"
        )
