"""Example of versioned API router."""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..versioning import APIVersion, VersionedAPIRouter, get_api_version


# Define models for different API versions
class StrategyV1(BaseModel):
    """Strategy model for API v1."""

    id: int
    name: str
    description: str
    is_active: bool


class StrategyV2(BaseModel):
    """Strategy model for API v2."""

    id: int
    name: str
    description: str
    is_active: bool
    version: int
    tags: List[str] = []
    parameters: Dict[str, str] = {}
    created_at: str
    updated_at: str


# Create versioned router
versioned_router = VersionedAPIRouter()

# V1 router
router_v1 = versioned_router.get_router(APIVersion.V1)


@router_v1.get("/strategies/{strategy_id}", response_model=StrategyV1)
def get_strategy_v1(strategy_id: int):
    """Get strategy by ID (API v1)."""
    # This would normally query the database
    return {
        "id": strategy_id,
        "name": "Example Strategy",
        "description": "This is an example strategy",
        "is_active": True,
    }


@router_v1.get("/strategies/", response_model=List[StrategyV1])
def list_strategies_v1(
    active_only: bool = Query(False, description="Only include active strategies"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """List strategies (API v1)."""
    # This would normally query the database
    return [
        {
            "id": 1,
            "name": "Example Strategy 1",
            "description": "This is an example strategy",
            "is_active": True,
        },
        {
            "id": 2,
            "name": "Example Strategy 2",
            "description": "This is another example strategy",
            "is_active": False,
        },
    ]


# V2 router
router_v2 = versioned_router.get_router(APIVersion.V2)


@router_v2.get("/strategies/{strategy_id}", response_model=StrategyV2)
def get_strategy_v2(strategy_id: int):
    """Get strategy by ID (API v2)."""
    # This would normally query the database
    return {
        "id": strategy_id,
        "name": "Example Strategy",
        "description": "This is an example strategy",
        "is_active": True,
        "version": 1,
        "tags": ["arbitrage", "usdc"],
        "parameters": {"threshold": "0.01", "window_size": "24"},
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-02T00:00:00Z",
    }


@router_v2.get("/strategies/", response_model=List[StrategyV2])
def list_strategies_v2(
    active_only: bool = Query(False, description="Only include active strategies"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """List strategies (API v2)."""
    # This would normally query the database
    return [
        {
            "id": 1,
            "name": "Example Strategy 1",
            "description": "This is an example strategy",
            "is_active": True,
            "version": 1,
            "tags": ["arbitrage", "usdc"],
            "parameters": {"threshold": "0.01", "window_size": "24"},
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
        },
        {
            "id": 2,
            "name": "Example Strategy 2",
            "description": "This is another example strategy",
            "is_active": False,
            "version": 2,
            "tags": ["mean_reversion"],
            "parameters": {"z_score": "2.0", "lookback": "48"},
            "created_at": "2025-01-03T00:00:00Z",
            "updated_at": "2025-01-04T00:00:00Z",
        },
    ]
