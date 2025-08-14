"""Backtest router for USDC arbitrage API."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .websocket import send_backtest_update

from .. import models
from ..backtesting import (
    BacktestEngine,
    ExchangeFeeModel,
    OrderSide,
    PositionSizer,
    PositionSizing,
    PortfolioRebalancer,
    RebalanceFrequency,
    SlippageModel,
)
from ..database import get_db
from ..security import (
    get_current_active_user,
    require_permissions,
)
from ..strategies import get_strategy_by_id

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("api.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

router = APIRouter()


class BacktestRequest(BaseModel):
    """Backtest request model."""

    strategy_id: int = Field(..., description="Strategy ID to backtest")
    start_date: datetime = Field(..., description="Start date for backtest")
    end_date: datetime = Field(..., description="End date for backtest")
    exchanges: List[str] = Field(
        default=["coinbase", "kraken", "binance"],
        description="Exchanges to include in backtest",
    )
    symbols: List[str] = Field(
        default=["USDC/USD"],
        description="Symbols to include in backtest",
    )
    timeframe: str = Field(
        default="1h",
        description="Timeframe for backtest (e.g., 1m, 5m, 1h, 1d)",
    )
    initial_balance: float = Field(
        default=10000.0,
        description="Initial portfolio balance",
        gt=0,
    )
    position_sizing: str = Field(
        default="percent",
        description="Position sizing strategy (fixed, percent, kelly, volatility, risk_parity)",
    )
    position_size: float = Field(
        default=0.02,
        description="Position size (interpretation depends on position_sizing)",
        gt=0,
    )
    rebalance_frequency: str = Field(
        default="monthly",
        description="Portfolio rebalancing frequency (daily, weekly, monthly, quarterly, threshold)",
    )
    rebalance_threshold: float = Field(
        default=0.05,
        description="Threshold for threshold-based rebalancing",
        gt=0,
    )
    include_fees: bool = Field(
        default=True,
        description="Include exchange fees in backtest",
    )
    include_slippage: bool = Field(
        default=True,
        description="Include slippage in backtest",
    )
    strategy_params: Dict[str, Any] = Field(
        default={},
        description="Additional strategy parameters",
    )


class BacktestResponse(BaseModel):
    """Backtest response model."""

    backtest_id: int
    strategy_id: int
    start_date: datetime
    end_date: datetime
    status: str
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime


@router.post("/backtest/", response_model=BacktestResponse)
def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    current_user: models.User = Depends(require_permissions(["create:backtest"])),
    db: Session = Depends(get_db),
):
    """Execute a backtest for a given strategy and parameters."""
    try:
        # Check if strategy exists
        strategy = get_strategy_by_id(request.strategy_id, db)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Create backtest record
        backtest_record = models.BacktestResult(
            strategy_id=request.strategy_id,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=request.dict(),
            status="pending",
        )
        db.add(backtest_record)
        db.commit()
        db.refresh(backtest_record)

        # Initialize backtesting engine with requested parameters
        engine = BacktestEngine(
            db_connection_string="postgresql://postgres:postgres@localhost:5432/usdc_arbitrage",
            initial_balance=request.initial_balance,
        )

        # Configure position sizing
        position_sizing_map = {
            "fixed": PositionSizing.FIXED,
            "percent": PositionSizing.PERCENT,
            "kelly": PositionSizing.KELLY,
            "volatility": PositionSizing.VOLATILITY,
            "risk_parity": PositionSizing.RISK_PARITY,
        }
        position_sizing = position_sizing_map.get(
            request.position_sizing, PositionSizing.PERCENT
        )

        engine.position_sizer = PositionSizer(
            strategy=position_sizing,
            percent_size=request.position_size,
            fixed_size=request.position_size
            if position_sizing == PositionSizing.FIXED
            else 1000.0,
        )

        # Configure rebalancing
        rebalance_frequency_map = {
            "daily": RebalanceFrequency.DAILY,
            "weekly": RebalanceFrequency.WEEKLY,
            "monthly": RebalanceFrequency.MONTHLY,
            "quarterly": RebalanceFrequency.QUARTERLY,
            "threshold": RebalanceFrequency.THRESHOLD,
        }
        rebalance_frequency = rebalance_frequency_map.get(
            request.rebalance_frequency, RebalanceFrequency.MONTHLY
        )

        engine.rebalancer = PortfolioRebalancer(
            frequency=rebalance_frequency,
            threshold=request.rebalance_threshold,
        )

        # Disable fees or slippage if requested
        if not request.include_fees:
            engine.fee_models = {
                exchange: ExchangeFeeModel(exchange, maker_fee=0, taker_fee=0)
                for exchange in request.exchanges
            }

        if not request.include_slippage:
            engine.slippage_model = SlippageModel(base_slippage=0)

        # Run backtest
        try:
            # Get strategy function
            strategy_func = strategy.get_strategy_function()

            # Run backtest
            results = engine.run_backtest(
                strategy_func=strategy_func,
                exchanges=request.exchanges,
                symbols=request.symbols,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date,
                strategy_params=request.strategy_params,
            )

            # Update backtest record with results
            backtest_record.status = "completed"
            backtest_record.results = results
            backtest_record.metrics = results.get("metrics", {})
            backtest_record.completed_at = datetime.now()
            db.commit()

            # Send WebSocket notification
            background_tasks.add_task(
                send_backtest_update,
                backtest_record.id,
                {
                    "status": "completed",
                    "progress": 100.0,
                    "metrics": backtest_record.metrics,
                },
            )

            # Return response
            return BacktestResponse(
                backtest_id=backtest_record.id,
                strategy_id=request.strategy_id,
                start_date=request.start_date,
                end_date=request.end_date,
                status="completed",
                metrics=results.get("metrics"),
                created_at=backtest_record.created_at,
            )

        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            backtest_record.status = "failed"
            backtest_record.error_message = str(e)
            backtest_record.completed_at = datetime.now()
            db.commit()

            # Send WebSocket notification
            background_tasks.add_task(
                send_backtest_update,
                backtest_record.id,
                {"status": "failed", "error": str(e)},
            )

            raise HTTPException(
                status_code=500,
                detail=f"Backtest execution failed: {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest request error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process backtest request: {str(e)}",
        )


@router.get("/backtest/{backtest_id}", response_model=BacktestResponse)
def get_backtest(
    backtest_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get backtest results by ID."""
    backtest = (
        db.query(models.BacktestResult)
        .filter(models.BacktestResult.id == backtest_id)
        .first()
    )

    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    return BacktestResponse(
        backtest_id=backtest.id,
        strategy_id=backtest.strategy_id,
        start_date=backtest.start_date,
        end_date=backtest.end_date,
        status=backtest.status,
        metrics=backtest.metrics,
        created_at=backtest.created_at,
    )


@router.get("/backtest/", response_model=List[BacktestResponse])
def list_backtests(
    strategy_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """List backtests with optional filtering."""
    query = db.query(models.BacktestResult)

    if strategy_id is not None:
        query = query.filter(models.BacktestResult.strategy_id == strategy_id)

    if status is not None:
        query = query.filter(models.BacktestResult.status == status)

    backtests = (
        query.order_by(models.BacktestResult.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        BacktestResponse(
            backtest_id=backtest.id,
            strategy_id=backtest.strategy_id,
            start_date=backtest.start_date,
            end_date=backtest.end_date,
            status=backtest.status,
            metrics=backtest.metrics,
            created_at=backtest.created_at,
        )
        for backtest in backtests
    ]
