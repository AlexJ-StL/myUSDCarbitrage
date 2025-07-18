"""API versioning utilities."""

from enum import Enum
from typing import Callable, Dict, Optional, Type

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel


class APIVersion(str, Enum):
    """API version enum."""

    V1 = "1.0"
    V2 = "2.0"

    @classmethod
    def default(cls) -> "APIVersion":
        """Get default API version."""
        return cls.V1

    @classmethod
    def latest(cls) -> "APIVersion":
        """Get latest API version."""
        return cls.V2


def get_api_version(
    accept_version: Optional[str] = Header(None, description="API version"),
) -> APIVersion:
    """Get API version from Accept-Version header."""
    if not accept_version:
        return APIVersion.default()

    try:
        return APIVersion(accept_version)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid API version: {accept_version}. Available versions: {[v.value for v in APIVersion]}",
        )


class VersionedAPIRouter:
    """Router that supports API versioning."""

    def __init__(self):
        self.routers: Dict[APIVersion, APIRouter] = {}

    def get_router(self, version: APIVersion) -> APIRouter:
        """Get router for specific version."""
        if version not in self.routers:
            self.routers[version] = APIRouter()
        return self.routers[version]

    def include_in_app(self, app: FastAPI, prefix: str = ""):
        """Include all versioned routers in FastAPI app."""
        for version, router in self.routers.items():
            app.include_router(
                router,
                prefix=f"/api/v{version.value.replace('.', '_')}{prefix}",
                dependencies=[Depends(lambda v=version: get_api_version() == v)],
            )


def version_response_model(
    v1_model: Type[BaseModel], v2_model: Optional[Type[BaseModel]] = None
) -> Callable:
    """Dependency for versioned response models."""

    def _get_model(version: APIVersion = Depends(get_api_version)) -> Type[BaseModel]:
        if version == APIVersion.V1 or not v2_model:
            return v1_model
        return v2_model

    return _get_model
