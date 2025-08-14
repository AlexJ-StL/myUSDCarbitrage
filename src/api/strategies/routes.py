"""API routes for strategy management."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..backtesting import BacktestEngine
from .comparison import StrategyComparison
from .manager import StrategyManager
from .types import STRATEGY_TYPES, create_strategy, get_strategy_template

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/strategies", tags=["strategies"])


# Pydantic models for API requests/responses
class StrategyCreateRequest(BaseModel):
    """Request model for creating a strategy."""

    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    code: str = Field(..., description="Strategy code (Python)")
    parameters: Dict[str, Any] = Field(default={}, description="Strategy parameters")
    author: str = Field(..., description="Author name")
    tags: Optional[List[str]] = Field(default=None, description="Strategy tags")
    is_active: bool = Field(default=True, description="Whether strategy is active")
    strategy_type: str = Field(default="custom", description="Strategy type")


class StrategyUpdateRequest(BaseModel):
    """Request model for updating a strategy."""

    name: Optional[str] = Field(default=None, description="Strategy name")
    description: Optional[str] = Field(default=None, description="Strategy description")
    code: Optional[str] = Field(default=None, description="Strategy code")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Strategy parameters"
    )
    author: str = Field(default="system", description="Author of the change")
    commit_message: str = Field(default="", description="Commit message")
    tags: Optional[List[str]] = Field(default=None, description="Strategy tags")
    is_active: Optional[bool] = Field(
        default=None, description="Whether strategy is active"
    )
    create_new_version: bool = Field(default=True, description="Create new version")


class StrategyComparisonRequest(BaseModel):
    """Request model for strategy comparison."""

    strategy_ids: List[int] = Field(..., description="Strategy IDs to compare")
    start_date: datetime = Field(..., description="Start date for comparison")
    end_date: datetime = Field(..., description="End date for comparison")
    exchanges: Optional[List[str]] = Field(
        default=None, description="Exchanges to test on"
    )
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to test on")
    timeframe: str = Field(default="1h", description="Timeframe for testing")
    initial_capital: float = Field(default=10000.0, description="Initial capital")


class ABTestCreateRequest(BaseModel):
    """Request model for creating an A/B test."""

    strategy_a_id: int = Field(..., description="Strategy A ID")
    strategy_b_id: int = Field(..., description="Strategy B ID")
    test_name: str = Field(..., description="A/B test name")
    description: str = Field(..., description="A/B test description")
    allocation_ratio: float = Field(
        default=0.5, description="Allocation ratio for strategy A"
    )
    start_date: Optional[datetime] = Field(default=None, description="Start date")
    end_date: Optional[datetime] = Field(default=None, description="End date")
    exchanges: Optional[List[str]] = Field(
        default=None, description="Exchanges to test on"
    )
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to test on")


class StrategyTestRequest(BaseModel):
    """Request model for testing a strategy."""

    start_date: datetime = Field(..., description="Start date for test")
    end_date: datetime = Field(..., description="End date for test")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Override parameters"
    )
    exchanges: Optional[List[str]] = Field(
        default=None, description="Exchanges to test on"
    )
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to test on")
    timeframe: str = Field(default="1h", description="Timeframe for testing")


# Dependency to get strategy manager
def get_strategy_manager(db: Session = Depends(get_db)) -> StrategyManager:
    """Get strategy manager instance."""
    # TODO: Initialize BacktestEngine properly
    backtest_engine = None  # BacktestEngine(db)
    return StrategyManager(db, backtest_engine)


# Dependency to get strategy comparison
def get_strategy_comparison(db: Session = Depends(get_db)) -> StrategyComparison:
    """Get strategy comparison instance."""
    # TODO: Initialize BacktestEngine properly
    backtest_engine = None  # BacktestEngine(db)
    return StrategyComparison(db, backtest_engine)


@router.post("/", response_model=models.StrategyPydantic)
async def create_strategy(
    request: StrategyCreateRequest,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Create a new strategy."""
    try:
        strategy = manager.create_strategy(
            name=request.name,
            description=request.description,
            code=request.code,
            parameters=request.parameters,
            author=request.author,
            tags=request.tags,
            is_active=request.is_active,
        )

        # Update strategy type if provided
        if request.strategy_type != "custom":
            strategy.strategy_type = request.strategy_type
            manager.db.commit()
            manager.db.refresh(strategy)

        return strategy
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=List[models.StrategyPydantic])
async def list_strategies(
    active_only: bool = Query(
        default=False, description="Only include active strategies"
    ),
    tag: Optional[str] = Query(default=None, description="Filter by tag"),
    search: Optional[str] = Query(
        default=None, description="Search in name and description"
    ),
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        default=100, ge=1, le=1000, description="Maximum number of records"
    ),
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """List strategies with optional filtering."""
    try:
        strategies = manager.list_strategies(
            active_only=active_only,
            tag=tag,
            search=search,
            skip=skip,
            limit=limit,
        )
        return strategies
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{strategy_id}", response_model=models.StrategyPydantic)
async def get_strategy(
    strategy_id: int,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Get a strategy by ID."""
    try:
        strategy = manager.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{strategy_id}", response_model=models.StrategyPydantic)
async def update_strategy(
    strategy_id: int,
    request: StrategyUpdateRequest,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Update a strategy."""
    try:
        strategy = manager.update_strategy(
            strategy_id=strategy_id,
            name=request.name,
            description=request.description,
            code=request.code,
            parameters=request.parameters,
            author=request.author,
            commit_message=request.commit_message,
            tags=request.tags,
            is_active=request.is_active,
            create_new_version=request.create_new_version,
        )
        return strategy
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{strategy_id}")
async def delete_strategy(
    strategy_id: int,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Delete a strategy."""
    try:
        success = manager.delete_strategy(strategy_id)
        if not success:
            raise HTTPException(status_code=404, detail="Strategy not found")
        return {"message": "Strategy deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{strategy_id}/versions")
async def get_strategy_versions(
    strategy_id: int,
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        default=100, ge=1, le=1000, description="Maximum number of records"
    ),
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Get versions of a strategy."""
    try:
        versions = manager.get_strategy_versions(strategy_id, skip=skip, limit=limit)
        return [
            {
                "id": v.id,
                "version": v.version,
                "created_at": v.created_at,
                "created_by": v.created_by,
                "commit_message": v.commit_message,
            }
            for v in versions
        ]
    except Exception as e:
        logger.error(f"Error getting versions for strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{strategy_id}/versions/{version}")
async def get_strategy_version(
    strategy_id: int,
    version: int,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Get a specific version of a strategy."""
    try:
        version_obj = manager.get_strategy_version(strategy_id, version)
        if not version_obj:
            raise HTTPException(status_code=404, detail="Strategy version not found")

        return {
            "id": version_obj.id,
            "strategy_id": version_obj.strategy_id,
            "version": version_obj.version,
            "code": version_obj.code,
            "parameters": version_obj.parameters,
            "created_at": version_obj.created_at,
            "created_by": version_obj.created_by,
            "commit_message": version_obj.commit_message,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting version {version} for strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{strategy_id}/revert/{version}", response_model=models.StrategyPydantic)
async def revert_strategy_to_version(
    strategy_id: int,
    version: int,
    author: str = Query(..., description="Author of the revert"),
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Revert a strategy to a previous version."""
    try:
        strategy = manager.revert_to_version(strategy_id, version, author)
        return strategy
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error reverting strategy {strategy_id} to version {version}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{strategy_id}/versions/{version1}/compare/{version2}")
async def compare_strategy_versions(
    strategy_id: int,
    version1: int,
    version2: int,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Compare two versions of a strategy."""
    try:
        comparison = manager.compare_versions(strategy_id, version1, version2)
        return comparison
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error comparing versions {version1} and {version2} for strategy {strategy_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{strategy_id}/test")
async def test_strategy(
    strategy_id: int,
    request: StrategyTestRequest,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Test a strategy with backtesting."""
    try:
        if not manager.backtest_engine:
            raise HTTPException(
                status_code=501, detail="Backtesting engine not available"
            )

        results = manager.test_strategy(
            strategy_id=strategy_id,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=request.parameters,
            exchanges=request.exchanges,
            symbols=request.symbols,
            timeframe=request.timeframe,
        )
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{strategy_id}/backtests")
async def get_strategy_backtests(
    strategy_id: int,
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        default=100, ge=1, le=1000, description="Maximum number of records"
    ),
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Get backtest results for a strategy."""
    try:
        backtests = manager.get_strategy_backtests(strategy_id, skip=skip, limit=limit)
        return [
            {
                "id": b.id,
                "start_date": b.start_date,
                "end_date": b.end_date,
                "parameters": b.parameters,
                "metrics": b.metrics,
                "status": b.status,
                "created_at": b.created_at,
                "completed_at": b.completed_at,
            }
            for b in backtests
        ]
    except Exception as e:
        logger.error(f"Error getting backtests for strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{strategy_id}/export")
async def export_strategy(
    strategy_id: int,
    include_history: bool = Query(default=False, description="Include version history"),
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Export a strategy to a portable format."""
    try:
        exported_data = manager.export_strategy(
            strategy_id, include_history=include_history
        )
        return exported_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/import", response_model=models.StrategyPydantic)
async def import_strategy(
    strategy_data: Dict[str, Any],
    author: str = Query(..., description="Author importing the strategy"),
    overwrite: bool = Query(default=False, description="Overwrite existing strategy"),
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Import a strategy from exported data."""
    try:
        strategy = manager.import_strategy(strategy_data, author, overwrite=overwrite)
        return strategy
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error importing strategy: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tags/")
async def get_strategy_tags(
    manager: StrategyManager = Depends(get_strategy_manager),
):
    """Get all strategy tags."""
    try:
        tags = manager.get_strategy_tags()
        return [
            {
                "id": tag.id,
                "name": tag.name,
                "description": tag.description,
                "created_at": tag.created_at,
            }
            for tag in tags
        ]
    except Exception as e:
        logger.error(f"Error getting strategy tags: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/types/")
async def get_strategy_types():
    """Get available strategy types and their templates."""
    try:
        types_info = {}
        for strategy_type in STRATEGY_TYPES:
            types_info[strategy_type] = {
                "name": strategy_type,
                "class": STRATEGY_TYPES[strategy_type].__name__,
                "description": STRATEGY_TYPES[strategy_type].__doc__ or "",
                "default_parameters": get_strategy_template(strategy_type),
            }
        return types_info
    except Exception as e:
        logger.error(f"Error getting strategy types: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/compare")
async def compare_strategies(
    request: StrategyComparisonRequest,
    comparison: StrategyComparison = Depends(get_strategy_comparison),
):
    """Compare multiple strategies."""
    try:
        if not comparison.backtest_engine:
            raise HTTPException(
                status_code=501, detail="Backtesting engine not available"
            )

        results = comparison.compare_strategies(
            strategy_ids=request.strategy_ids,
            start_date=request.start_date,
            end_date=request.end_date,
            exchanges=request.exchanges,
            symbols=request.symbols,
            timeframe=request.timeframe,
            initial_capital=request.initial_capital,
        )
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ab-tests/")
async def create_ab_test(
    request: ABTestCreateRequest,
    comparison: StrategyComparison = Depends(get_strategy_comparison),
):
    """Create an A/B test between two strategies."""
    try:
        ab_test = comparison.create_ab_test(
            strategy_a_id=request.strategy_a_id,
            strategy_b_id=request.strategy_b_id,
            test_name=request.test_name,
            description=request.description,
            allocation_ratio=request.allocation_ratio,
            start_date=request.start_date,
            end_date=request.end_date,
            exchanges=request.exchanges,
            symbols=request.symbols,
        )

        return {
            "id": ab_test.id,
            "name": ab_test.name,
            "description": ab_test.description,
            "status": ab_test.status,
            "created_at": ab_test.created_at,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ab-tests/{ab_test_id}/run")
async def run_ab_test(
    ab_test_id: int,
    comparison: StrategyComparison = Depends(get_strategy_comparison),
):
    """Run an A/B test."""
    try:
        if not comparison.backtest_engine:
            raise HTTPException(
                status_code=501, detail="Backtesting engine not available"
            )

        results = comparison.run_ab_test(ab_test_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running A/B test {ab_test_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/ab-tests/{ab_test_id}")
async def get_ab_test_results(
    ab_test_id: int,
    comparison: StrategyComparison = Depends(get_strategy_comparison),
):
    """Get A/B test results."""
    try:
        results = comparison.get_ab_test_results(ab_test_id)
        if not results:
            raise HTTPException(status_code=404, detail="A/B test not found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B test results {ab_test_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/ab-tests/")
async def list_ab_tests(
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        default=100, ge=1, le=1000, description="Maximum number of records"
    ),
    comparison: StrategyComparison = Depends(get_strategy_comparison),
):
    """List A/B tests."""
    try:
        ab_tests = comparison.list_ab_tests(skip=skip, limit=limit)
        return ab_tests
    except Exception as e:
        logger.error(f"Error listing A/B tests: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
