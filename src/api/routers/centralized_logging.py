"""Centralized logging API endpoints."""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from ...monitoring.centralized_logging import (
    get_centralized_logging,
    CentralizedLogging,
)
from ...monitoring.log_rotation import get_log_rotator, LogRotator

router = APIRouter(prefix="/logging/centralized", tags=["centralized_logging"])


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
    centralized_logging: CentralizedLogging = Depends(get_centralized_logging),
):
    """Search logs with various filters."""
    try:
        results = await centralized_logging.search_logs(
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
    centralized_logging: CentralizedLogging = Depends(get_centralized_logging),
):
    """Search logs using GET parameters."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        results = await centralized_logging.search_logs(
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
    centralized_logging: CentralizedLogging = Depends(get_centralized_logging),
):
    """Get log statistics for the specified time period."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        stats = await centralized_logging.get_log_statistics(
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
    centralized_logging: CentralizedLogging = Depends(get_centralized_logging),
):
    """Export logs in specified format."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        exported_data = await centralized_logging.export_logs(
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
    centralized_logging: CentralizedLogging = Depends(get_centralized_logging),
):
    """Track an error manually."""
    try:
        # Create a mock exception for tracking
        class TrackedError(Exception):
            pass

        error = TrackedError(error_request.error_message)
        error.__class__.__name__ = error_request.error_type

        error_id = centralized_logging.track_error(
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


@router.get("/performance/bottlenecks")
async def analyze_bottlenecks(
    hours: int = Query(24, description="Hours to analyze", ge=1, le=168),
    centralized_logging: CentralizedLogging = Depends(get_centralized_logging),
):
    """Analyze performance data to identify bottlenecks."""
    try:
        analysis = await centralized_logging.analyze_bottlenecks(hours=hours)
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze bottlenecks: {str(e)}"
        )


@router.get("/analysis/error-patterns")
async def analyze_error_patterns(
    hours: int = Query(24, description="Hours to analyze", ge=1, le=168),
    min_occurrences: int = Query(
        3, description="Minimum occurrences to consider a pattern", ge=1, le=100
    ),
    centralized_logging: CentralizedLogging = Depends(get_centralized_logging),
):
    """Analyze error logs to identify patterns and recurring issues."""
    try:
        analysis = await centralized_logging.analyze_error_patterns(
            hours=hours, min_occurrences=min_occurrences
        )
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze error patterns: {str(e)}"
        )


@router.get("/insights")
async def generate_log_insights(
    days: int = Query(7, description="Days to analyze", ge=1, le=30),
    centralized_logging: CentralizedLogging = Depends(get_centralized_logging),
):
    """Generate insights from log data."""
    try:
        insights = await centralized_logging.generate_log_insights(days=days)
        return insights
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate log insights: {str(e)}"
        )


@router.get("/dashboard")
async def get_logging_dashboard(
    hours: int = Query(24, description="Hours to analyze", ge=1, le=168),
    centralized_logging: CentralizedLogging = Depends(get_centralized_logging),
):
    """Get comprehensive logging dashboard data."""
    try:
        dashboard_data = await centralized_logging.get_logging_dashboard(hours=hours)
        return dashboard_data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get logging dashboard: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_old_logs(
    retention_days: int = Query(7, description="Days to retain logs", ge=1, le=30),
    centralized_logging: CentralizedLogging = Depends(get_centralized_logging),
):
    """Clean up old logs beyond retention period."""
    try:
        removed_count = await centralized_logging.cleanup_old_logs(
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


@router.post("/rotate")
async def rotate_logs(
    background_tasks: BackgroundTasks,
    log_rotator: LogRotator = Depends(get_log_rotator),
):
    """Manually trigger log rotation."""
    try:
        # Run log rotation in background
        background_tasks.add_task(log_rotator.check_and_rotate_logs)

        return {
            "message": "Log rotation triggered",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rotate logs: {str(e)}")


@router.post("/archive")
async def archive_old_logs(
    days: int = Query(30, description="Days threshold for archiving", ge=1, le=365),
    background_tasks: BackgroundTasks,
    log_rotator: LogRotator = Depends(get_log_rotator),
):
    """Archive logs older than specified days."""
    try:
        # Run archiving in background
        background_tasks.add_task(log_rotator.archive_old_logs, days)

        return {
            "message": f"Archiving logs older than {days} days",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to archive logs: {str(e)}")