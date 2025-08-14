"""Health check and monitoring endpoints."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ...monitoring.health_checks import ServiceHealthChecker, get_health_checker
from ...monitoring.metrics_collector import MetricsCollector, get_metrics_collector
from ...monitoring.service_recovery import ServiceRecoveryManager, get_recovery_manager

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def basic_health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
    }


@router.get("/detailed")
async def detailed_health_check(
    db: Session = Depends(get_db),
    health_checker: ServiceHealthChecker = Depends(get_health_checker),
):
    """Comprehensive health check of all services."""
    try:
        health_status = await health_checker.comprehensive_health_check(db)
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/database")
async def database_health_check(
    db: Session = Depends(get_db),
    health_checker: ServiceHealthChecker = Depends(get_health_checker),
):
    """Database-specific health check."""
    try:
        db_health = await health_checker.check_database_health(db)
        return db_health
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Database health check failed: {str(e)}"
        )


@router.get("/redis")
async def redis_health_check(
    health_checker: ServiceHealthChecker = Depends(get_health_checker),
):
    """Redis-specific health check."""
    try:
        redis_health = await health_checker.check_redis_health()
        return redis_health
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Redis health check failed: {str(e)}"
        )


@router.get("/system")
async def system_health_check(
    health_checker: ServiceHealthChecker = Depends(get_health_checker),
):
    """System resources health check."""
    try:
        import asyncio

        system_health = await asyncio.get_event_loop().run_in_executor(
            None, health_checker.check_system_resources
        )
        return system_health
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"System health check failed: {str(e)}"
        )


@router.get("/dependencies")
async def dependencies_health_check(
    health_checker: ServiceHealthChecker = Depends(get_health_checker),
):
    """External dependencies health check."""
    try:
        deps_health = await health_checker.check_service_dependencies()
        return deps_health
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Dependencies health check failed: {str(e)}"
        )


@router.get("/metrics/current")
async def current_metrics(
    db: Session = Depends(get_db),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
):
    """Get current system metrics."""
    try:
        import asyncio

        # Collect current metrics
        system_metrics, db_metrics, redis_metrics = await asyncio.gather(
            asyncio.get_event_loop().run_in_executor(
                None, metrics_collector.collect_system_metrics
            ),
            metrics_collector.collect_database_metrics(db),
            metrics_collector.collect_redis_metrics(),
            return_exceptions=True,
        )

        return {
            "system": system_metrics
            if not isinstance(system_metrics, Exception)
            else {"error": str(system_metrics)},
            "database": db_metrics
            if not isinstance(db_metrics, Exception)
            else {"error": str(db_metrics)},
            "redis": redis_metrics
            if not isinstance(redis_metrics, Exception)
            else {"error": str(redis_metrics)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to collect current metrics: {str(e)}"
        )


@router.get("/metrics/history")
async def metrics_history(
    metric_type: str = Query(
        ..., description="Type of metrics (system, database, redis)"
    ),
    hours: int = Query(
        1, description="Number of hours of history to retrieve", ge=1, le=24
    ),
    limit: int = Query(100, description="Maximum number of data points", ge=1, le=1000),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
):
    """Get historical metrics data."""
    try:
        import time

        start_time = time.time() - (hours * 3600)
        end_time = time.time()

        history = await metrics_collector.get_metrics_history(
            metric_type=metric_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        return {
            "metric_type": metric_type,
            "hours": hours,
            "data_points": len(history),
            "data": history,
            "query_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve metrics history: {str(e)}"
        )


@router.get("/services")
async def services_status(
    recovery_manager: ServiceRecoveryManager = Depends(get_recovery_manager),
):
    """Get status of all monitored services."""
    try:
        services = ["uvicorn", "celery", "redis", "postgres"]
        status_results = {}

        for service in services:
            status_results[service] = recovery_manager.check_service_health(service)

        return {
            "services": status_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to check services status: {str(e)}"
        )


@router.post("/services/{service_name}/restart")
async def restart_service(
    service_name: str,
    recovery_manager: ServiceRecoveryManager = Depends(get_recovery_manager),
):
    """Manually restart a specific service."""
    try:
        if service_name not in ["uvicorn", "celery"]:
            raise HTTPException(
                status_code=400,
                detail=f"Service '{service_name}' cannot be restarted via API",
            )

        result = await recovery_manager.restart_service(service_name)

        if result["success"]:
            return {
                "message": f"Service '{service_name}' restarted successfully",
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to restart service '{service_name}': {result.get('reason', 'unknown')}",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to restart service: {str(e)}"
        )


@router.post("/recovery/check")
async def check_and_recover_services(
    recovery_manager: ServiceRecoveryManager = Depends(get_recovery_manager),
):
    """Check all services and attempt automatic recovery if needed."""
    try:
        results = await recovery_manager.check_and_recover_services()
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Service recovery check failed: {str(e)}"
        )


@router.get("/liveness")
async def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/readiness")
async def readiness_probe(
    db: Session = Depends(get_db),
    health_checker: ServiceHealthChecker = Depends(get_health_checker),
):
    """Kubernetes readiness probe endpoint."""
    try:
        # Quick health checks for readiness
        db_health = await health_checker.check_database_health(db)
        redis_health = await health_checker.check_redis_health()

        if db_health["status"] == "healthy" and redis_health["status"] == "healthy":
            return {
                "status": "ready",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Readiness check failed: {str(e)}")
