"""
API endpoints for report generation.

This module provides FastAPI endpoints for generating reports.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.reporting.report_generator import generate_arbitrage_report

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reporting"])


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


@router.post("/reports/arbitrage", response_class=HTMLResponse, tags=["Reporting"])
async def create_arbitrage_report(
    request: ArbitrageReportRequest, db: Session = Depends(get_db)
):
    """
    Generates and returns an on-demand HTML report analyzing arbitrage
    opportunities for a given symbol across multiple exchanges.
    """
    end_time = request.end_time or datetime.utcnow()
    start_time = request.start_time or (end_time - timedelta(days=1))

    if start_time >= end_time:
        raise HTTPException(
            status_code=400, detail="start_time must be before end_time."
        )

    try:
        html_report = generate_arbitrage_report(
            db=db,
            start_time=start_time,
            end_time=end_time,
            exchanges=request.exchanges,
            symbol=request.symbol,
            threshold=request.threshold,
        )
        return HTMLResponse(content=html_report, status_code=200)
    except Exception as e:
        # In a production app, this should be logged in detail.
        logger.exception(f"Failed to generate arbitrage report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: Failed to generate report. {str(e)}",
        )
