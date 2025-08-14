"""
API endpoints for on-demand report generation.

This module provides FastAPI endpoints for generating on-demand reports
for arbitrage opportunities and strategy performance.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.database import DBConnector, get_db
from src.reporting.on_demand_report_generator import (
    OnDemandReportGenerator,
    get_on_demand_report_generator,
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["On-Demand Reports"])


class ArbitrageReportRequest(BaseModel):
    """Request model for arbitrage opportunity report."""

    exchanges: List[str] = Field(
        ...,
        min_length=2,
        description="List of exchanges to compare.",
        example=["coinbase", "kraken"],
    )
    symbol: str = Field(
        "USDC/USD", description="The trading symbol to analyze.", example="USDC/USD"
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Start time for the report period (UTC). Defaults to 24 hours ago.",
    )
    end_time: Optional[datetime] = Field(
        None, description="End time for the report period (UTC). Defaults to now."
    )
    threshold: float = Field(
        0.001,
        gt=0,
        description="Minimum percentage difference to be considered an opportunity (e.g., 0.001 for 0.1%).",
        example=0.001,
    )
    output_format: str = Field(
        "html",
        description="Output format (html, json, csv).",
        example="html",
    )


class StrategyReportRequest(BaseModel):
    """Request model for strategy performance report."""

    strategy_id: Optional[int] = Field(
        None,
        description="ID of the strategy to analyze.",
        example=1,
    )
    backtest_id: Optional[int] = Field(
        None,
        description="ID of the specific backtest to analyze.",
        example=1,
    )
    start_date: Optional[datetime] = Field(
        None,
        description="Start date for analysis period.",
    )
    end_date: Optional[datetime] = Field(
        None,
        description="End date for analysis period.",
    )
    include_benchmark: bool = Field(
        False,
        description="Whether to include benchmark comparison.",
        example=False,
    )
    benchmark_symbol: str = Field(
        "BTC/USD",
        description="Symbol to use as benchmark.",
        example="BTC/USD",
    )
    include_sections: Optional[List[str]] = Field(
        None,
        description="Specific sections to include in the report.",
        example=["executive_summary", "performance_metrics", "equity_curve"],
    )
    output_format: str = Field(
        "html",
        description="Output format (html, json, csv).",
        example="html",
    )


def get_db_connector() -> DBConnector:
    """Get database connector for raw SQL queries."""
    # This is a simplified example - in a real implementation,
    # you would get the connection string from environment variables
    connection_string = (
        "postgresql://arb_user:strongpassword@localhost:5432/usdc_arbitrage"
    )
    return DBConnector(connection_string)


@router.post(
    "/reports/arbitrage",
    response_class=Response,
    summary="Generate arbitrage opportunity report",
    description="Generates an on-demand report analyzing arbitrage opportunities across multiple exchanges.",
)
async def create_arbitrage_report(
    request: ArbitrageReportRequest,
    db: Session = Depends(get_db),
    db_connector: DBConnector = Depends(get_db_connector),
):
    """Generate an on-demand report for arbitrage opportunities."""
    try:
        # Create report generator
        report_generator = get_on_demand_report_generator(db, db_connector)

        # Generate report
        report = report_generator.generate_arbitrage_opportunity_report(
            exchanges=request.exchanges,
            symbol=request.symbol,
            start_time=request.start_time,
            end_time=request.end_time,
            threshold=request.threshold,
            output_format=request.output_format,
        )

        # Handle error
        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])

        # Return appropriate response based on content type
        if report["content_type"] == "text/html":
            return HTMLResponse(content=report["content"], status_code=200)
        elif report["content_type"] == "application/json":
            return JSONResponse(content=report["content"], status_code=200)
        elif report["content_type"] == "text/csv":
            return Response(
                content=report["content"],
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=arbitrage_report.csv"
                },
            )
        else:
            return Response(
                content=report["content"], media_type=report["content_type"]
            )

    except Exception as e:
        logger.exception(f"Error generating arbitrage report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate report: {str(e)}"
        )


@router.post(
    "/reports/strategy",
    response_class=Response,
    summary="Generate strategy performance report",
    description="Generates an on-demand report analyzing strategy performance metrics.",
)
async def create_strategy_report(
    request: StrategyReportRequest,
    db: Session = Depends(get_db),
    db_connector: DBConnector = Depends(get_db_connector),
):
    """Generate an on-demand report for strategy performance."""
    try:
        # Validate request
        if not request.strategy_id and not request.backtest_id:
            raise HTTPException(
                status_code=400,
                detail="Either strategy_id or backtest_id must be provided",
            )

        # Create report generator
        report_generator = get_on_demand_report_generator(db, db_connector)

        # Generate report
        report = report_generator.generate_strategy_performance_report(
            strategy_id=request.strategy_id,
            backtest_id=request.backtest_id,
            start_date=request.start_date,
            end_date=request.end_date,
            include_benchmark=request.include_benchmark,
            benchmark_symbol=request.benchmark_symbol,
            include_sections=request.include_sections,
            output_format=request.output_format,
        )

        # Handle error
        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])

        # Return appropriate response based on content type
        if report["content_type"] == "text/html":
            return HTMLResponse(content=report["content"], status_code=200)
        elif report["content_type"] == "application/json":
            return JSONResponse(content=report["content"], status_code=200)
        elif report["content_type"] == "text/csv":
            return Response(
                content=report["content"],
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=strategy_report.csv"
                },
            )
        else:
            return Response(
                content=report["content"], media_type=report["content_type"]
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating strategy report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate report: {str(e)}"
        )


@router.get(
    "/reports/arbitrage",
    response_class=HTMLResponse,
    summary="Generate arbitrage opportunity report (GET)",
    description="Generates an on-demand report analyzing arbitrage opportunities across multiple exchanges.",
)
async def get_arbitrage_report(
    exchanges: List[str] = Query(
        ..., min_length=2, description="List of exchanges to compare"
    ),
    symbol: str = Query("USDC/USD", description="The trading symbol to analyze"),
    start_time: Optional[datetime] = Query(
        None, description="Start time for the report period (UTC)"
    ),
    end_time: Optional[datetime] = Query(
        None, description="End time for the report period (UTC)"
    ),
    threshold: float = Query(0.001, gt=0, description="Minimum percentage difference"),
    output_format: str = Query("html", description="Output format (html, json, csv)"),
    db: Session = Depends(get_db),
    db_connector: DBConnector = Depends(get_db_connector),
):
    """Generate an on-demand report for arbitrage opportunities using GET method."""
    try:
        # Create report generator
        report_generator = get_on_demand_report_generator(db, db_connector)

        # Generate report
        report = report_generator.generate_arbitrage_opportunity_report(
            exchanges=exchanges,
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            threshold=threshold,
            output_format=output_format,
        )

        # Handle error
        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])

        # Return appropriate response based on content type
        if report["content_type"] == "text/html":
            return HTMLResponse(content=report["content"], status_code=200)
        elif report["content_type"] == "application/json":
            return JSONResponse(content=report["content"], status_code=200)
        elif report["content_type"] == "text/csv":
            return Response(
                content=report["content"],
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=arbitrage_report.csv"
                },
            )
        else:
            return Response(
                content=report["content"], media_type=report["content_type"]
            )

    except Exception as e:
        logger.exception(f"Error generating arbitrage report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate report: {str(e)}"
        )


@router.get(
    "/reports/strategy",
    response_class=HTMLResponse,
    summary="Generate strategy performance report (GET)",
    description="Generates an on-demand report analyzing strategy performance metrics.",
)
async def get_strategy_report(
    strategy_id: Optional[int] = Query(
        None, description="ID of the strategy to analyze"
    ),
    backtest_id: Optional[int] = Query(
        None, description="ID of the specific backtest to analyze"
    ),
    start_date: Optional[datetime] = Query(
        None, description="Start date for analysis period"
    ),
    end_date: Optional[datetime] = Query(
        None, description="End date for analysis period"
    ),
    include_benchmark: bool = Query(
        False, description="Whether to include benchmark comparison"
    ),
    benchmark_symbol: str = Query("BTC/USD", description="Symbol to use as benchmark"),
    include_sections: Optional[List[str]] = Query(
        None, description="Specific sections to include"
    ),
    output_format: str = Query("html", description="Output format (html, json, csv)"),
    db: Session = Depends(get_db),
    db_connector: DBConnector = Depends(get_db_connector),
):
    """Generate an on-demand report for strategy performance using GET method."""
    try:
        # Validate request
        if not strategy_id and not backtest_id:
            raise HTTPException(
                status_code=400,
                detail="Either strategy_id or backtest_id must be provided",
            )

        # Create report generator
        report_generator = get_on_demand_report_generator(db, db_connector)

        # Generate report
        report = report_generator.generate_strategy_performance_report(
            strategy_id=strategy_id,
            backtest_id=backtest_id,
            start_date=start_date,
            end_date=end_date,
            include_benchmark=include_benchmark,
            benchmark_symbol=benchmark_symbol,
            include_sections=include_sections,
            output_format=output_format,
        )

        # Handle error
        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])

        # Return appropriate response based on content type
        if report["content_type"] == "text/html":
            return HTMLResponse(content=report["content"], status_code=200)
        elif report["content_type"] == "application/json":
            return JSONResponse(content=report["content"], status_code=200)
        elif report["content_type"] == "text/csv":
            return Response(
                content=report["content"],
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=strategy_report.csv"
                },
            )
        else:
            return Response(
                content=report["content"], media_type=report["content_type"]
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating strategy report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate report: {str(e)}"
        )
