"""Data export router for USDC arbitrage API."""

import io
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..security import get_current_active_user, require_permissions

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["data_export"])


class ExportRequest(BaseModel):
    """Request model for data export."""

    data_type: str  # "market_data", "backtest_results", "strategy_comparison"
    format: str  # "json", "csv", "parquet"
    filters: Optional[Dict] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    exchanges: Optional[List[str]] = None
    symbols: Optional[List[str]] = None
    strategy_id: Optional[int] = None
    backtest_id: Optional[int] = None
    limit: Optional[int] = 10000


@router.post("/data")
async def export_data(
    request: ExportRequest,
    current_user: models.User = Depends(require_permissions(["export:data"])),
    db: Session = Depends(get_db),
):
    """Export data in the requested format."""
    try:
        # Validate data type
        valid_data_types = [
            "market_data",
            "backtest_results",
            "strategy_comparison",
            "transactions",
        ]
        if request.data_type not in valid_data_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data_type. Must be one of: {', '.join(valid_data_types)}",
            )

        # Validate format
        valid_formats = ["json", "csv", "parquet"]
        if request.format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Must be one of: {', '.join(valid_formats)}",
            )

        # Get data based on type
        data = await get_export_data(request, db)

        # Convert to DataFrame for easy format conversion
        df = pd.DataFrame(data)

        # Return data in requested format
        if request.format == "json":
            return Response(
                content=df.to_json(orient="records", date_format="iso"),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=export_{request.data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                },
            )

        elif request.format == "csv":
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv",
            )
            response.headers["Content-Disposition"] = (
                f"attachment; filename=export_{request.data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            return response

        elif request.format == "parquet":
            stream = io.BytesIO()
            df.to_parquet(stream, index=False)
            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="application/octet-stream",
            )
            response.headers["Content-Disposition"] = (
                f"attachment; filename=export_{request.data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            )
            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


async def get_export_data(request: ExportRequest, db: Session) -> List[Dict]:
    """Get data for export based on request parameters."""

    if request.data_type == "market_data":
        # Query market data with filters
        query = db.query(models.USDCData)

        if request.start_date:
            query = query.filter(models.USDCData.timestamp >= request.start_date)

        if request.end_date:
            query = query.filter(models.USDCData.timestamp <= request.end_date)

        if request.exchanges:
            query = query.filter(models.USDCData.exchange.in_(request.exchanges))

        # Apply limit
        if request.limit:
            query = query.limit(request.limit)

        # Execute query
        results = query.all()

        # Convert to dict
        return [
            {
                "id": item.id,
                "timestamp": item.timestamp,
                "exchange": item.exchange,
                "price": item.price,
            }
            for item in results
        ]

    elif request.data_type == "backtest_results":
        # Query backtest results with filters
        query = db.query(models.BacktestResult)

        if request.strategy_id:
            query = query.filter(
                models.BacktestResult.strategy_id == request.strategy_id
            )

        if request.backtest_id:
            query = query.filter(models.BacktestResult.id == request.backtest_id)

        if request.start_date:
            query = query.filter(models.BacktestResult.start_date >= request.start_date)

        if request.end_date:
            query = query.filter(models.BacktestResult.end_date <= request.end_date)

        # Apply limit
        if request.limit:
            query = query.limit(request.limit)

        # Execute query
        results = query.all()

        # Convert to dict
        return [
            {
                "id": item.id,
                "strategy_id": item.strategy_id,
                "start_date": item.start_date,
                "end_date": item.end_date,
                "parameters": item.parameters,
                "results": item.results,
                "metrics": item.metrics,
                "status": item.status,
                "created_at": item.created_at,
                "completed_at": item.completed_at,
            }
            for item in results
        ]

    elif request.data_type == "strategy_comparison":
        # Query strategy comparisons with filters
        query = db.query(models.StrategyComparison)

        if request.start_date:
            query = query.filter(
                models.StrategyComparison.start_date >= request.start_date
            )

        if request.end_date:
            query = query.filter(models.StrategyComparison.end_date <= request.end_date)

        # Apply limit
        if request.limit:
            query = query.limit(request.limit)

        # Execute query
        results = query.all()

        # Convert to dict
        return [
            {
                "id": item.id,
                "name": item.name,
                "strategy_ids": item.strategy_ids,
                "start_date": item.start_date,
                "end_date": item.end_date,
                "parameters": item.parameters,
                "results": item.results,
                "created_at": item.created_at,
            }
            for item in results
        ]

    elif request.data_type == "transactions":
        # Query backtest transactions with filters
        if not request.backtest_id:
            raise HTTPException(
                status_code=400, detail="backtest_id is required for transaction export"
            )

        # Get backtest result
        backtest = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == request.backtest_id)
            .first()
        )

        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")

        # Extract transactions from results
        if not backtest.results or "transactions" not in backtest.results:
            return []

        return backtest.results["transactions"]

    # Default empty response
    return []


@router.get("/backtest/{backtest_id}")
async def export_backtest_results(
    backtest_id: int,
    format: str = Query("json", description="Export format (json, csv, parquet)"),
    include_transactions: bool = Query(
        False, description="Include transaction details"
    ),
    include_positions: bool = Query(False, description="Include position snapshots"),
    current_user: models.User = Depends(require_permissions(["export:data"])),
    db: Session = Depends(get_db),
):
    """Export backtest results in the requested format."""
    try:
        # Validate format
        valid_formats = ["json", "csv", "parquet"]
        if format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Must be one of: {', '.join(valid_formats)}",
            )

        # Get backtest result
        backtest = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == backtest_id)
            .first()
        )

        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")

        # Prepare data for export
        export_data = {
            "id": backtest.id,
            "strategy_id": backtest.strategy_id,
            "start_date": backtest.start_date,
            "end_date": backtest.end_date,
            "parameters": backtest.parameters,
            "metrics": backtest.metrics,
            "status": backtest.status,
            "created_at": backtest.created_at,
            "completed_at": backtest.completed_at,
        }

        # Include additional data if requested
        if (
            include_transactions
            and backtest.results
            and "transactions" in backtest.results
        ):
            export_data["transactions"] = backtest.results["transactions"]

        if include_positions and backtest.results and "positions" in backtest.results:
            export_data["positions"] = backtest.results["positions"]

        # Convert to DataFrame for easy format conversion
        df = pd.json_normalize(export_data)

        # Return data in requested format
        if format == "json":
            return Response(
                content=json.dumps(export_data, default=str),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=backtest_{backtest_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                },
            )

        elif format == "csv":
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv",
            )
            response.headers["Content-Disposition"] = (
                f"attachment; filename=backtest_{backtest_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            return response

        elif format == "parquet":
            stream = io.BytesIO()
            df.to_parquet(stream, index=False)
            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="application/octet-stream",
            )
            response.headers["Content-Disposition"] = (
                f"attachment; filename=backtest_{backtest_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            )
            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting backtest results: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to export backtest results: {str(e)}"
        )


@router.get("/strategy/{strategy_id}")
async def export_strategy(
    strategy_id: int,
    format: str = Query("json", description="Export format (json, csv, parquet)"),
    include_code: bool = Query(False, description="Include strategy code"),
    include_backtests: bool = Query(False, description="Include backtest results"),
    current_user: models.User = Depends(require_permissions(["export:data"])),
    db: Session = Depends(get_db),
):
    """Export strategy details in the requested format."""
    try:
        # Validate format
        valid_formats = ["json", "csv", "parquet"]
        if format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Must be one of: {', '.join(valid_formats)}",
            )

        # Get strategy
        strategy = (
            db.query(models.Strategy).filter(models.Strategy.id == strategy_id).first()
        )

        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Prepare data for export
        export_data = {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "parameters": strategy.parameters,
            "version": strategy.version,
            "strategy_type": strategy.strategy_type,
            "is_active": strategy.is_active,
            "created_at": strategy.created_at,
            "updated_at": strategy.updated_at,
            "tags": [tag.name for tag in strategy.tags],
        }

        # Include code if requested
        if include_code:
            export_data["code"] = strategy.code

        # Include backtests if requested
        if include_backtests:
            backtests = (
                db.query(models.BacktestResult)
                .filter(models.BacktestResult.strategy_id == strategy_id)
                .all()
            )

            export_data["backtests"] = [
                {
                    "id": backtest.id,
                    "start_date": backtest.start_date,
                    "end_date": backtest.end_date,
                    "metrics": backtest.metrics,
                    "status": backtest.status,
                    "created_at": backtest.created_at,
                }
                for backtest in backtests
            ]

        # Return data in requested format
        if format == "json":
            return Response(
                content=json.dumps(export_data, default=str),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=strategy_{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                },
            )

        elif format == "csv":
            # Convert to DataFrame for CSV export
            df = pd.json_normalize(export_data)
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv",
            )
            response.headers["Content-Disposition"] = (
                f"attachment; filename=strategy_{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            return response

        elif format == "parquet":
            # Convert to DataFrame for Parquet export
            df = pd.json_normalize(export_data)
            stream = io.BytesIO()
            df.to_parquet(stream, index=False)
            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="application/octet-stream",
            )
            response.headers["Content-Disposition"] = (
                f"attachment; filename=strategy_{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            )
            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting strategy: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to export strategy: {str(e)}"
        )
