"""Results router for USDC arbitrage API."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..security import get_current_active_user, require_permissions

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/results", tags=["results"])


class BacktestResultFilter(BaseModel):
    """Filter model for backtest results."""

    strategy_id: Optional[int] = None
    status: Optional[str] = None
    start_date_from: Optional[datetime] = None
    start_date_to: Optional[datetime] = None
    end_date_from: Optional[datetime] = None
    end_date_to: Optional[datetime] = None
    created_at_from: Optional[datetime] = None
    created_at_to: Optional[datetime] = None


@router.get("/", response_model=List[models.BacktestResultPydantic])
def list_results(
    strategy_id: Optional[int] = None,
    status: Optional[str] = None,
    start_date_from: Optional[datetime] = None,
    start_date_to: Optional[datetime] = None,
    end_date_from: Optional[datetime] = None,
    end_date_to: Optional[datetime] = None,
    created_at_from: Optional[datetime] = None,
    created_at_to: Optional[datetime] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """List backtest results with filtering and sorting."""
    try:
        # Build query
        query = db.query(models.BacktestResult)

        # Apply filters
        if strategy_id is not None:
            query = query.filter(models.BacktestResult.strategy_id == strategy_id)

        if status is not None:
            query = query.filter(models.BacktestResult.status == status)

        if start_date_from is not None:
            query = query.filter(models.BacktestResult.start_date >= start_date_from)

        if start_date_to is not None:
            query = query.filter(models.BacktestResult.start_date <= start_date_to)

        if end_date_from is not None:
            query = query.filter(models.BacktestResult.end_date >= end_date_from)

        if end_date_to is not None:
            query = query.filter(models.BacktestResult.end_date <= end_date_to)

        if created_at_from is not None:
            query = query.filter(models.BacktestResult.created_at >= created_at_from)

        if created_at_to is not None:
            query = query.filter(models.BacktestResult.created_at <= created_at_to)

        # Apply sorting
        sort_column = getattr(
            models.BacktestResult, sort_by, models.BacktestResult.created_at
        )
        if sort_order.lower() == "asc":
            query = query.order_by(sort_column)
        else:
            query = query.order_by(desc(sort_column))

        # Apply pagination
        results = query.offset(skip).limit(limit).all()

        return results

    except Exception as e:
        logger.error(f"Error listing backtest results: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list backtest results: {str(e)}"
        )


@router.get("/{result_id}", response_model=models.BacktestResultPydantic)
def get_result(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get backtest result by ID."""
    try:
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest result {result_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get backtest result: {str(e)}"
        )


@router.get("/{result_id}/transactions", response_model=List[Dict[str, Any]])
def get_result_transactions(
    result_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get transactions for a backtest result."""
    try:
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        if not result.results or "transactions" not in result.results:
            return []

        transactions = result.results["transactions"]

        # Apply pagination
        start = skip
        end = skip + limit

        return transactions[start:end]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transactions for backtest result {result_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get transactions: {str(e)}"
        )


@router.get("/{result_id}/positions", response_model=List[Dict[str, Any]])
def get_result_positions(
    result_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get position snapshots for a backtest result."""
    try:
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        if not result.results or "positions" not in result.results:
            return []

        positions = result.results["positions"]

        # Apply pagination
        start = skip
        end = skip + limit

        return positions[start:end]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting positions for backtest result {result_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get positions: {str(e)}"
        )


@router.get("/{result_id}/equity_curve", response_model=List[Dict[str, Any]])
def get_result_equity_curve(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get equity curve for a backtest result."""
    try:
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        if not result.results or "equity_curve" not in result.results:
            return []

        return result.results["equity_curve"]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting equity curve for backtest result {result_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get equity curve: {str(e)}"
        )


@router.delete("/{result_id}")
def delete_result(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["delete:backtest", "delete:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Delete backtest result by ID."""
    try:
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        db.delete(result)
        db.commit()

        return {"message": "Backtest result deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting backtest result {result_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete backtest result: {str(e)}"
        )


@router.get("/strategy/{strategy_id}/summary", response_model=Dict[str, Any])
def get_strategy_results_summary(
    strategy_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get summary of backtest results for a strategy."""
    try:
        # Build query
        query = db.query(models.BacktestResult).filter(
            models.BacktestResult.strategy_id == strategy_id,
            models.BacktestResult.status == "completed",
        )

        if start_date:
            query = query.filter(models.BacktestResult.start_date >= start_date)

        if end_date:
            query = query.filter(models.BacktestResult.end_date <= end_date)

        results = query.all()

        if not results:
            return {"strategy_id": strategy_id, "backtest_count": 0, "metrics": {}}

        # Calculate average metrics
        avg_metrics = {}
        for metric in [
            "total_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "cagr",
            "win_rate",
        ]:
            values = [
                result.metrics.get(metric, 0)
                for result in results
                if result.metrics and metric in result.metrics
            ]

            if values:
                avg_metrics[f"avg_{metric}"] = sum(values) / len(values)
                avg_metrics[f"min_{metric}"] = min(values)
                avg_metrics[f"max_{metric}"] = max(values)

        return {
            "strategy_id": strategy_id,
            "backtest_count": len(results),
            "date_range": {
                "earliest": min(r.start_date for r in results),
                "latest": max(r.end_date for r in results),
            },
            "metrics": avg_metrics,
        }

    except Exception as e:
        logger.error(f"Error getting summary for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get strategy summary: {str(e)}"
        )
