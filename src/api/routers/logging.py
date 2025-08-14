"""Logging and error tracking API endpoints."""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...monitoring.log_aggregation import LogAggregator, get_log_aggregator
from ...monitoring.logging_config import ErrorTracker, get_error_tracker
from ...monitoring.performance_profiler import (
    PerformanceProfiler,
    get_performance_profiler,
)
from ...monitoring.log_analysis import LogAnalyzer, get_log_analyzer
from ...monitoring.performance_profiler import (
    PerformanceProfiler,
    get_performance_profiler,
)

router = APIRouter(prefix="/logging", tags=["logging"])


class LogSearchRequest(BaseModel):
    """Model for log search requests."""

    query: Optional[str] = None
    log_level: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 100
    offset: int = 0


class ErrorTrackingRequest(BaseModel):
    """Model for error tracking requests."""

    error_type: str
    error_message: str
    context: str
    metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


@router.post("/search")
async def search_logs(
    search_request: LogSearchRequest,
    log_aggregator: LogAggregator = Depends(get_log_aggregator),
):
    """Search logs with various filters."""
    try:
        results = await log_aggregator.search_logs(
            query=search_request.query,
            log_level=search_request.log_level,
            start_time=search_request.start_time,
            end_time=search_request.end_time,
            limit=search_request.limit,
            offset=search_request.offset,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search logs: {str(e)}")


@router.get("/search")
async def search_logs_get(
    query: Optional[str] = Query(None, description="Search query"),
    log_level: Optional[str] = Query(
        None, description="Log level filter (debug, info, warning, error, critical)"
    ),
    hours: int = Query(24, description="Hours to look back", ge=1, le=168),
    limit: int = Query(100, description="Maximum number of logs", ge=1, le=1000),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    log_aggregator: LogAggregator = Depends(get_log_aggregator),
):
    """Search logs using GET parameters."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        results = await log_aggregator.search_logs(
            query=query,
            log_level=log_level,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search logs: {str(e)}")


@router.get("/statistics")
async def get_log_statistics(
    hours: int = Query(24, description="Hours to analyze", ge=1, le=168),
    log_aggregator: LogAggregator = Depends(get_log_aggregator),
):
    """Get log statistics for the specified time period."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        stats = await log_aggregator.get_log_statistics(
            start_time=start_time, end_time=end_time
        )
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get log statistics: {str(e)}"
        )


@router.get("/export")
async def export_logs(
    format_type: str = Query("json", description="Export format (json, csv, txt)"),
    query: Optional[str] = Query(None, description="Search query"),
    log_level: Optional[str] = Query(None, description="Log level filter"),
    hours: int = Query(24, description="Hours to look back", ge=1, le=168),
    limit: int = Query(1000, description="Maximum number of logs", ge=1, le=5000),
    log_aggregator: LogAggregator = Depends(get_log_aggregator),
):
    """Export logs in specified format."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        exported_data = await log_aggregator.export_logs(
            format_type=format_type,
            query=query,
            log_level=log_level,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        # Set appropriate content type
        if format_type.lower() == "json":
            media_type = "application/json"
            filename = f"logs_{start_time.strftime('%Y%m%d_%H%M')}.json"
        elif format_type.lower() == "csv":
            media_type = "text/csv"
            filename = f"logs_{start_time.strftime('%Y%m%d_%H%M')}.csv"
        else:
            media_type = "text/plain"
            filename = f"logs_{start_time.strftime('%Y%m%d_%H%M')}.txt"

        from fastapi.responses import Response

        return Response(
            content=exported_data,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export logs: {str(e)}")


@router.post("/errors/track")
async def track_error(
    error_request: ErrorTrackingRequest,
    error_tracker: ErrorTracker = Depends(get_error_tracker),
):
    """Track an error manually."""
    try:
        # Create a mock exception for tracking
        class TrackedError(Exception):
            pass

        error = TrackedError(error_request.error_message)
        error.__class__.__name__ = error_request.error_type

        error_id = error_tracker.track_error(
            error=error,
            context=error_request.context,
            metadata=error_request.metadata,
            user_id=error_request.user_id,
        )

        return {
            "message": "Error tracked successfully",
            "error_id": error_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track error: {str(e)}")


@router.get("/errors/statistics")
async def get_error_statistics(
    hours: int = Query(24, description="Hours to analyze", ge=1, le=168),
    error_tracker: ErrorTracker = Depends(get_error_tracker),
):
    """Get error statistics for the specified time period."""
    try:
        stats = error_tracker.get_error_statistics(hours=hours)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get error statistics: {str(e)}"
        )


@router.get("/performance/profiles")
async def get_performance_profiles(
    operation: Optional[str] = Query(None, description="Filter by operation name"),
    hours: int = Query(24, description="Hours to look back", ge=1, le=168),
    limit: int = Query(100, description="Maximum number of profiles", ge=1, le=500),
    profiler: PerformanceProfiler = Depends(get_performance_profiler),
):
    """Get performance profiles with optional filtering."""
    try:
        profiles = await profiler.get_performance_profiles(
            operation=operation, hours=hours, limit=limit
        )

        return {
            "operation": operation,
            "hours": hours,
            "total_profiles": len(profiles),
            "profiles": profiles,
            "query_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance profiles: {str(e)}"
        )


@router.get("/performance/bottlenecks")
async def analyze_bottlenecks(
    hours: int = Query(24, description="Hours to analyze", ge=1, le=168),
    profiler: PerformanceProfiler = Depends(get_performance_profiler),
):
    """Analyze performance data to identify bottlenecks."""
    try:
        analysis = await profiler.analyze_bottlenecks(hours=hours)
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze bottlenecks: {str(e)}"
        )


@router.get("/performance/trends/{operation}")
async def get_operation_trends(
    operation: str,
    days: int = Query(7, description="Days to analyze", ge=1, le=30),
    profiler: PerformanceProfiler = Depends(get_performance_profiler),
):
    """Get performance trends for a specific operation."""
    try:
        trends = await profiler.get_operation_trends(operation=operation, days=days)
        return trends
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get operation trends: {str(e)}"
        )


@router.get("/performance/summary")
async def get_performance_summary(
    hours: int = Query(24, description="Hours to analyze", ge=1, le=168),
    profiler: PerformanceProfiler = Depends(get_performance_profiler),
):
    """Get performance summary with key metrics."""
    try:
        profiles = await profiler.get_performance_profiles(hours=hours, limit=1000)

        if not profiles:
            return {
                "message": "No performance data available",
                "hours": hours,
                "total_profiles": 0,
            }

        # Calculate summary statistics
        execution_times = [p.get("execution_time_ms", 0) for p in profiles]
        memory_usage = [p.get("memory_used_mb", 0) for p in profiles]
        cpu_times = [p.get("cpu_time_seconds", 0) for p in profiles]

        # Group by operation
        operations = {}
        for profile in profiles:
            op_name = profile.get("operation", "unknown")
            if op_name not in operations:
                operations[op_name] = 0
            operations[op_name] += 1

        top_operations = sorted(operations.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        summary = {
            "period_hours": hours,
            "total_profiles": len(profiles),
            "total_operations": len(operations),
            "performance_metrics": {
                "avg_execution_time_ms": sum(execution_times) / len(execution_times),
                "max_execution_time_ms": max(execution_times),
                "min_execution_time_ms": min(execution_times),
                "avg_memory_usage_mb": sum(memory_usage) / len(memory_usage),
                "max_memory_usage_mb": max(memory_usage),
                "total_cpu_time_seconds": sum(cpu_times),
            },
            "top_operations": [
                {"operation": op, "call_count": count} for op, count in top_operations
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return summary
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance summary: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_old_logs(
    retention_days: int = Query(7, description="Days to retain logs", ge=1, le=30),
    log_aggregator: LogAggregator = Depends(get_log_aggregator),
):
    """Clean up old logs beyond retention period."""
    try:
        removed_count = await log_aggregator.cleanup_old_logs(
            retention_days=retention_days
        )

        return {
            "message": f"Cleanup completed",
            "removed_logs": removed_count,
            "retention_days": retention_days,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup logs: {str(e)}")


@router.get("/analysis/error-patterns")
async def analyze_error_patterns(
    hours: int = Query(24, description="Hours to analyze", ge=1, le=168),
    min_occurrences: int = Query(
        3, description="Minimum occurrences to consider a pattern", ge=1, le=100
    ),
    log_analyzer: LogAnalyzer = Depends(get_log_analyzer),
):
    """Analyze error logs to identify patterns and recurring issues."""
    try:
        analysis = await log_analyzer.analyze_error_patterns(
            hours=hours, min_occurrences=min_occurrences
        )
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze error patterns: {str(e)}"
        )


@router.get("/analysis/volume-anomalies")
async def analyze_log_volume_anomalies(
    days: int = Query(7, description="Days to analyze", ge=1, le=30),
    anomaly_threshold: float = Query(
        2.0, description="Anomaly detection threshold", ge=1.1, le=10.0
    ),
    log_analyzer: LogAnalyzer = Depends(get_log_analyzer),
):
    """Analyze log volume to detect anomalies."""
    try:
        analysis = await log_analyzer.analyze_log_volume_anomalies(
            days=days, anomaly_threshold=anomaly_threshold
        )
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze log volume anomalies: {str(e)}"
        )


@router.get("/analysis/error-chains")
async def identify_error_chains(
    hours: int = Query(24, description="Hours to analyze", ge=1, le=168),
    log_analyzer: LogAnalyzer = Depends(get_log_analyzer),
):
    """Identify chains of related errors that may indicate cascading failures."""
    try:
        analysis = await log_analyzer.identify_error_chains(hours=hours)
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to identify error chains: {str(e)}"
        )


@router.get("/analysis/insights")
async def generate_log_insights(
    days: int = Query(7, description="Days to analyze", ge=1, le=30),
    log_analyzer: LogAnalyzer = Depends(get_log_analyzer),
):
    """Generate insights from log data."""
    try:
        insights = await log_analyzer.generate_log_insights(days=days)
        return insights
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate log insights: {str(e)}"
        )


@router.get("/dashboard")
async def get_logging_dashboard(
    hours: int = Query(24, description="Hours to analyze", ge=1, le=168),
    log_aggregator: LogAggregator = Depends(get_log_aggregator),
    error_tracker: ErrorTracker = Depends(get_error_tracker),
    log_analyzer: LogAnalyzer = Depends(get_log_analyzer),
):
    """Get comprehensive logging dashboard data."""
    try:
        # Collect all dashboard data concurrently
        import asyncio

        log_stats, error_stats, insights = await asyncio.gather(
            log_aggregator.get_log_statistics(
                start_time=datetime.now(timezone.utc) - timedelta(hours=hours),
                end_time=datetime.now(timezone.utc),
            ),
            error_tracker.get_error_statistics(hours=hours),
            log_analyzer.generate_log_insights(days=hours // 24 or 1),
            return_exceptions=True,
        )

        # Handle any exceptions
        dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "period_hours": hours,
            "log_statistics": log_stats
            if not isinstance(log_stats, Exception)
            else {"error": str(log_stats)},
            "error_statistics": error_stats
            if not isinstance(error_stats, Exception)
            else {"error": str(error_stats)},
            "insights": insights.get("insights", [])
            if not isinstance(insights, Exception)
            else [],
        }

        return dashboard_data

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get logging dashboard: {str(e)}"
        )
