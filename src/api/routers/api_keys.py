"""API key management endpoints."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..api_keys import (
    APIKeyCreate,
    APIKeyResponse,
    APIKeyService,
    APIKeyUpdate,
    APIKeyWithSecret,
    format_api_key_response,
    get_api_key_service,
)
from ..audit_logging import AuditEventType, AuditLogger
from ..database import get_db
from ..models import User
from ..security import get_current_active_user, require_permissions

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


@router.post("/", response_model=APIKeyWithSecret, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(require_permissions(["manage:api_keys"])),
    api_key_service: APIKeyService = Depends(get_api_key_service),
    db: Session = Depends(get_db),
):
    """Create a new API key (admin only)."""
    try:
        api_key = api_key_service.create_api_key(key_data, created_by=current_user.id)

        # Log the API key creation
        audit_logger = AuditLogger(db)
        audit_logger.log_api_key_event(
            event_type=AuditEventType.API_KEY_CREATED,
            api_key_id=api_key.id,
            api_key_name=api_key.name,
            details={
                "permissions": key_data.permissions,
                "rate_limit": key_data.rate_limit,
                "expires_in_days": key_data.expires_in_days,
                "created_by": current_user.username,
            },
        )

        return api_key
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}",
        )


@router.get("/", response_model=List[APIKeyResponse])
async def list_api_keys(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    current_user: User = Depends(require_permissions(["manage:api_keys"])),
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """List all API keys (admin only)."""
    api_keys = api_key_service.list_api_keys()

    # Apply pagination
    paginated_keys = api_keys[skip : skip + limit]

    return [format_api_key_response(key) for key in paginated_keys]


@router.get("/my", response_model=List[APIKeyResponse])
async def list_my_api_keys(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    current_user: User = Depends(get_current_active_user),
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """List API keys created by the current user."""
    api_keys = api_key_service.list_api_keys(created_by=current_user.id)

    # Apply pagination
    paginated_keys = api_keys[skip : skip + limit]

    return [format_api_key_response(key) for key in paginated_keys]


@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: int,
    current_user: User = Depends(require_permissions(["manage:api_keys"])),
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """Get API key details (admin only)."""
    api_key = api_key_service.get_api_key_by_id(key_id)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    return format_api_key_response(api_key)


@router.put("/{key_id}", response_model=APIKeyResponse)
async def update_api_key(
    key_id: int,
    update_data: APIKeyUpdate,
    current_user: User = Depends(require_permissions(["manage:api_keys"])),
    api_key_service: APIKeyService = Depends(get_api_key_service),
    db: Session = Depends(get_db),
):
    """Update API key (admin only)."""
    api_key = api_key_service.update_api_key(key_id, update_data)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    # Log the API key update
    audit_logger = AuditLogger(db)
    audit_logger.log_api_key_event(
        event_type=AuditEventType.API_KEY_CREATED,  # Using created as there's no updated event
        api_key_id=api_key.id,
        api_key_name=api_key.name,
        details={
            "action": "updated",
            "updated_by": current_user.username,
            "changes": update_data.dict(exclude_unset=True),
        },
    )

    return format_api_key_response(api_key)


@router.delete("/{key_id}")
async def delete_api_key(
    key_id: int,
    current_user: User = Depends(require_permissions(["manage:api_keys"])),
    api_key_service: APIKeyService = Depends(get_api_key_service),
    db: Session = Depends(get_db),
):
    """Delete API key (admin only)."""
    api_key = api_key_service.get_api_key_by_id(key_id)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    # Log the API key deletion before deleting
    audit_logger = AuditLogger(db)
    audit_logger.log_api_key_event(
        event_type=AuditEventType.API_KEY_REVOKED,
        api_key_id=api_key.id,
        api_key_name=api_key.name,
        details={
            "action": "deleted",
            "deleted_by": current_user.username,
        },
    )

    success = api_key_service.delete_api_key(key_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key",
        )

    return {"message": f"API key '{api_key.name}' deleted successfully"}


@router.post("/{key_id}/revoke")
async def revoke_api_key(
    key_id: int,
    current_user: User = Depends(require_permissions(["manage:api_keys"])),
    api_key_service: APIKeyService = Depends(get_api_key_service),
    db: Session = Depends(get_db),
):
    """Revoke API key (disable without deleting)."""
    api_key = api_key_service.get_api_key_by_id(key_id)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    if not api_key.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API key is already revoked",
        )

    # Revoke the key
    update_data = APIKeyUpdate(is_active=False)
    api_key = api_key_service.update_api_key(key_id, update_data)

    # Log the API key revocation
    audit_logger = AuditLogger(db)
    audit_logger.log_api_key_event(
        event_type=AuditEventType.API_KEY_REVOKED,
        api_key_id=api_key.id,
        api_key_name=api_key.name,
        details={
            "action": "revoked",
            "revoked_by": current_user.username,
        },
    )

    return {"message": f"API key '{api_key.name}' revoked successfully"}


@router.post("/{key_id}/activate")
async def activate_api_key(
    key_id: int,
    current_user: User = Depends(require_permissions(["manage:api_keys"])),
    api_key_service: APIKeyService = Depends(get_api_key_service),
    db: Session = Depends(get_db),
):
    """Activate a revoked API key."""
    api_key = api_key_service.get_api_key_by_id(key_id)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    if api_key.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API key is already active",
        )

    # Activate the key
    update_data = APIKeyUpdate(is_active=True)
    api_key = api_key_service.update_api_key(key_id, update_data)

    # Log the API key activation
    audit_logger = AuditLogger(db)
    audit_logger.log_api_key_event(
        event_type=AuditEventType.API_KEY_CREATED,  # Using created as there's no activated event
        api_key_id=api_key.id,
        api_key_name=api_key.name,
        details={
            "action": "activated",
            "activated_by": current_user.username,
        },
    )

    return {"message": f"API key '{api_key.name}' activated successfully"}


@router.get("/{key_id}/usage")
async def get_api_key_usage(
    key_id: int,
    current_user: User = Depends(require_permissions(["manage:api_keys"])),
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """Get API key usage statistics."""
    api_key = api_key_service.get_api_key_by_id(key_id)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    return {
        "api_key_id": api_key.id,
        "name": api_key.name,
        "usage_count": api_key.usage_count,
        "last_used_at": api_key.last_used_at,
        "rate_limit": api_key.rate_limit,
        "is_active": api_key.is_active,
        "expires_at": api_key.expires_at,
    }
