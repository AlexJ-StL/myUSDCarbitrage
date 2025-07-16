"""Backtest router for USDC arbitrage API."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db

router = APIRouter()


@router.post("/backtest/")
def run_backtest(
    strategy_id: int, start_date: str, end_date: str, db: Session = Depends(get_db)
):
    """Execute a backtest for a given strategy and date range."""
    # TODO: Implement actual backtesting engine
    return {
        "message": f"Backtest for strategy {strategy_id} from {start_date} to {end_date} would run here.",
        "status": "pending",
    }
