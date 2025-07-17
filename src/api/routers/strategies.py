"""API routes for strategy management."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..strategies.comparison import StrategyComparison
from ..strategies.manager import StrategyManager
from ..strategies.types import STRATEGY_TYPES, get_strategy_template

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategies"])


# Pydantic models for API requests/responses
class StrategyCreateRequest(BaseModel):
    """Request model for creating a strategy."""

    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    code: str = Field(..., description="Strategy code")
    parameters: Dict[str, Any] = Field(default={}, description="Strategy parameters")
    tags: Optional[List[str]] = Field(default=None, description="Strategy tags")
    strategy_type: str = Field(default="custom", description="Strategy type")
    is_active: bool = Field(default=True, description="Whether strategy is active")


class StrategyUpdateRequest(BaseModel):
    """Request model for updating a strategy."""

    name: Optional[str] = Field(default=None, description="Strategy name")
    description: Optional[str] = Field(default=None, description="Strategy description")
    code: Optional[str] = Field(default=None, description="Strategy code")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Strategy parameters"
    )
    tags: Optional[List[str]] = Field(default=None, description="Strategy tags")
    is_active: Optional[bool] = Field(
        default=None, description="Whether strategy is active"
    )
    commit_message: str = Field(
        default="", description="Commit message for version control"
    )


class StrategyComparisonRequest(BaseModel):
    """Request model for strategy comparison."""

    strategy_ids: List[int] = Field(..., description="List of strategy IDs to compare")
    start_date: datetime = Field(..., description="Start date for comparison")
    end_date: datetime = Field(..., description="End date for comparison")
    exchanges: Optional[List[str]] = Field(
        default=None, description="Exchanges to test on"
    )
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to test on")
    timeframe: str = Field(default="1h", description="Timeframe for testing")
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    comparison_name: Optional[str] = Field(default=None, description="Comparison name")


class ABTestRequest(BaseModel):
    """Request model for A/B testing."""

    strategy_a_id: int = Field(..., description="Strategy A ID")
    strategy_b_id: int = Field(..., description="Strategy B ID")
    start_date: datetime = Field(..., description="Start date for test")
    end_date: datetime = Field(..., description="End date for test")
    test_name: Optional[str] = Field(default=None, description="Test name")
    exchanges: Optional[List[str]] = Field(
        default=None, description="Exchanges to test on"
    )
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to test on")
    timeframe: str = Field(default="1h", description="Timeframe for testing")
    confidence_level: float = Field(
        default=0.95, description="Confidence level for statistical tests"
    )


# Strategy CRUD endpoints
@router.post("/", response_model=Dict[str, Any])
async def create_strategy(
    request: StrategyCreateRequest,
    author: str = Query("api", description="Author name"),
    db: Session = Depends(get_db),
):
    """Create a new strategy."""
    try:
        manager = StrategyManager(db)
        strategy = manager.create_strategy(
            name=request.name,
            description=request.description,
            code=request.code,
            parameters=request.parameters,
            author=author,
            tags=request.tags,
            is_active=request.is_active,
        )

        # Update strategy type if provided
        if request.strategy_type != "custom":
            strategy.strategy_type = request.strategy_type
            db.commit()

        return {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "version": strategy.version,
            "strategy_type": strategy.strategy_type,
            "is_active": strategy.is_active,
            "created_at": strategy.created_at,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=List[Dict[str, Any]])
async def list_strategies(
    active_only: bool = Query(False, description="Only include active strategies"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    db: Session = Depends(get_db),
):
    """List strategies with optional filtering."""
    try:
        manager = StrategyManager(db)
        strategies = manager.list_strategies(
            active_only=active_only,
            tag=tag,
            search=search,
            skip=skip,
            limit=limit,
        )

        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "version": s.version,
                "strategy_type": s.strategy_type,
                "is_active": s.is_active,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in strategies
        ]
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{strategy_id}", response_model=Dict[str, Any])
async def get_strategy(
    strategy_id: int,
    include_code: bool = Query(False, description="Include strategy code"),
    db: Session = Depends(get_db),
):
    """Get strategy by ID."""
    try:
        manager = StrategyManager(db)
        strategy = manager.get_strategy(strategy_id)

        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Get tags
        tags = [tag.name for tag in strategy.tags]

        result = {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "parameters": strategy.parameters,
            "version": strategy.version,
            "strategy_type": strategy.strategy_type,
            "is_active": strategy.is_active,
            "created_at": strategy.created_at,
            "updated_at": strategy.updated_at,
            "tags": tags,
        }

        if include_code:
            result["code"] = strategy.code

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{strategy_id}", response_model=Dict[str, Any])
async def update_strategy(
    strategy_id: int,
    request: StrategyUpdateRequest,
    author: str = Query("api", description="Author name"),
    db: Session = Depends(get_db),
):
    """Update an existing strategy."""
    try:
        manager = StrategyManager(db)
        strategy = manager.update_strategy(
            strategy_id=strategy_id,
            name=request.name,
            description=request.description,
            code=request.code,
            parameters=request.parameters,
            author=author,
            commit_message=request.commit_message,
            tags=request.tags,
            is_active=request.is_active,
        )

        return {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "version": strategy.version,
            "strategy_type": strategy.strategy_type,
            "is_active": strategy.is_active,
            "updated_at": strategy.updated_at,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{strategy_id}")
async def delete_strategy(
    strategy_id: int,
    db: Session = Depends(get_db),
):
    """Delete a strategy."""
    try:
        manager = StrategyManager(db)
        success = manager.delete_strategy(strategy_id)

        if not success:
            raise HTTPException(status_code=404, detail="Strategy not found")

        return {"message": "Strategy deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Strategy version endpoints
@router.get("/{strategy_id}/versions", response_model=List[Dict[str, Any]])
async def get_strategy_versions(
    strategy_id: int,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    db: Session = Depends(get_db),
):
    """Get versions of a strategy."""
    try:
        manager = StrategyManager(db)
        versions = manager.get_strategy_versions(strategy_id, skip=skip, limit=limit)

        return [
            {
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


@router.get("/{strategy_id}/versions/{version}", response_model=Dict[str, Any])
async def get_strategy_version(
    strategy_id: int,
    version: int,
    include_code: bool = Query(False, description="Include strategy code"),
    db: Session = Depends(get_db),
):
    """Get specific version of a strategy."""
    try:
        manager = StrategyManager(db)
        version_obj = manager.get_strategy_version(strategy_id, version)

        if not version_obj:
            raise HTTPException(status_code=404, detail="Strategy version not found")

        result = {
            "version": version_obj.version,
            "parameters": version_obj.parameters,
            "created_at": version_obj.created_at,
            "created_by": version_obj.created_by,
            "commit_message": version_obj.commit_message,
        }

        if include_code:
            result["code"] = version_obj.code

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting version {version} for strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{strategy_id}/revert/{version}", response_model=Dict[str, Any])
async def revert_strategy_to_version(
    strategy_id: int,
    version: int,
    author: str = Query("api", description="Author name"),
    db: Session = Depends(get_db),
):
    """Revert strategy to a previous version."""
    try:
        manager = StrategyManager(db)
        strategy = manager.revert_to_version(strategy_id, version, author)

        return {
            "id": strategy.id,
            "name": strategy.name,
            "version": strategy.version,
            "updated_at": strategy.updated_at,
            "message": f"Reverted to version {version}",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error reverting strategy {strategy_id} to version {version}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{strategy_id}/compare-versions/{version1}/{version2}",
    response_model=Dict[str, Any],
)
async def compare_strategy_versions(
    strategy_id: int,
    version1: int,
    version2: int,
    db: Session = Depends(get_db),
):
    """Compare two versions of a strategy."""
    try:
        manager = StrategyManager(db)
        comparison = manager.compare_versions(strategy_id, version1, version2)

        return comparison
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error comparing versions {version1} and {version2} for strategy {strategy_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error")


# Strategy export/import endpoints
@router.get("/{strategy_id}/export", response_model=Dict[str, Any])
async def export_strategy(
    strategy_id: int,
    include_history: bool = Query(False, description="Include version history"),
    db: Session = Depends(get_db),
):
    """Export strategy to a portable format."""
    try:
        manager = StrategyManager(db)
        exported_data = manager.export_strategy(strategy_id, include_history)

        return exported_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/import", response_model=Dict[str, Any])
async def import_strategy(
    strategy_data: Dict[str, Any],
    author: str = Query("api", description="Author name"),
    overwrite: bool = Query(False, description="Overwrite existing strategy"),
    db: Session = Depends(get_db),
):
    """Import strategy from exported data."""
    try:
        manager = StrategyManager(db)
        strategy = manager.import_strategy(strategy_data, author, overwrite)

        return {
            "id": strategy.id,
            "name": strategy.name,
            "version": strategy.version,
            "message": "Strategy imported successfully",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error importing strategy: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Strategy comparison endpoints
@router.post("/compare", response_model=Dict[str, Any])
async def compare_strategies(
    request: StrategyComparisonRequest,
    db: Session = Depends(get_db),
):
    """Compare multiple strategies using backtesting."""
    try:
        from ..backtesting import BacktestEngine

        backtest_engine = BacktestEngine(db)
        comparison = StrategyComparison(db, backtest_engine)

        result = comparison.compare_strategies(
            strategy_ids=request.strategy_ids,
            start_date=request.start_date,
            end_date=request.end_date,
            exchanges=request.exchanges,
            symbols=request.symbols,
            timeframe=request.timeframe,
            initial_capital=request.initial_capital,
            comparison_name=request.comparison_name,
        )

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ab-test", response_model=Dict[str, Any])
async def run_ab_test(
    request: ABTestRequest,
    db: Session = Depends(get_db),
):
    """Run A/B test between two strategies."""
    try:
        from ..backtesting import BacktestEngine

        backtest_engine = BacktestEngine(db)
        comparison = StrategyComparison(db, backtest_engine)

        result = comparison.run_ab_test(
            strategy_a_id=request.strategy_a_id,
            strategy_b_id=request.strategy_b_id,
            start_date=request.start_date,
            end_date=request.end_date,
            test_name=request.test_name,
            exchanges=request.exchanges,
            symbols=request.symbols,
            timeframe=request.timeframe,
            confidence_level=request.confidence_level,
        )

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error running A/B test: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Strategy tags endpoints
@router.get("/tags/", response_model=List[Dict[str, Any]])
async def get_strategy_tags(db: Session = Depends(get_db)):
    """Get all strategy tags."""
    try:
        manager = StrategyManager(db)
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


# Strategy types endpoints
@router.get("/types/", response_model=Dict[str, List[str]])
async def get_strategy_types():
    """Get available strategy types."""
    return {
        "strategy_types": list(STRATEGY_TYPES.keys()),
        "descriptions": {
            "arbitrage": "Exploits price differences between exchanges",
            "trend_following": "Follows price momentum using moving averages",
            "mean_reversion": "Trades price reversals to the mean",
            "volatility_breakout": "Trades breakouts from price ranges",
        },
    }


@router.get("/types/{strategy_type}/template", response_model=Dict[str, Any])
async def get_strategy_template(strategy_type: str):
    """Get default parameters template for a strategy type."""
    try:
        template = get_strategy_template(strategy_type)
        return {
            "strategy_type": strategy_type,
            "parameters": template,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting template for strategy type {strategy_type}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Comparison history endpoints
@router.get("/comparisons/", response_model=List[Dict[str, Any]])
async def get_comparison_history(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    db: Session = Depends(get_db),
):
    """Get history of strategy comparisons."""
    try:
        from ..backtesting import BacktestEngine

        backtest_engine = BacktestEngine(db)
        comparison = StrategyComparison(db, backtest_engine)

        comparisons = comparison.get_comparison_history(skip=skip, limit=limit)

        return [
            {
                "id": c.id,
                "name": c.name,
                "strategy_ids": c.strategy_ids,
                "start_date": c.start_date,
                "end_date": c.end_date,
                "created_at": c.created_at,
            }
            for c in comparisons
        ]
    except Exception as e:
        logger.error(f"Error getting comparison history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/ab-tests/", response_model=List[Dict[str, Any]])
async def get_ab_test_history(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    db: Session = Depends(get_db),
):
    """Get history of A/B tests."""
    try:
        from ..backtesting import BacktestEngine

        backtest_engine = BacktestEngine(db)
        comparison = StrategyComparison(db, backtest_engine)

        ab_tests = comparison.get_ab_test_history(skip=skip, limit=limit)

        return [
            {
                "id": test.id,
                "name": test.name,
                "strategy_a_id": test.strategy_a_id,
                "strategy_b_id": test.strategy_b_id,
                "start_date": test.start_date,
                "end_date": test.end_date,
                "created_at": test.created_at,
            }
            for test in ab_tests
        ]
    except Exception as e:
        logger.error(f"Error getting A/B test history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
