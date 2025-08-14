"""Admin interface for user and role management."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (
    Role,
    RoleCreate,
    RoleResponse,
    User,
    UserCreate,
    UserResponse,
    UserRole,
)
from ..security import (
    SecurityService,
    get_current_active_user,
    get_security_service,
    require_permissions,
    require_roles,
)

router = APIRouter(prefix="/admin", tags=["admin"])


# Pydantic models for admin operations
class UserUpdateRequest(BaseModel):
    """Request model for updating user information."""

    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = Field(
        None, pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class RoleUpdateRequest(BaseModel):
    """Request model for updating role information."""

    name: Optional[str] = Field(None, min_length=1, max_length=50)
    description: Optional[str] = None
    permissions: Optional[List[str]] = None


class UserRoleAssignment(BaseModel):
    """Request model for user-role assignments."""

    user_id: int
    role_id: int


class BulkUserOperation(BaseModel):
    """Request model for bulk user operations."""

    user_ids: List[int]
    operation: str = Field(..., pattern="^(activate|deactivate|delete)$")


class SystemStats(BaseModel):
    """Response model for system statistics."""

    total_users: int
    active_users: int
    total_roles: int
    total_permissions: int
    recent_logins: int


# User Management Endpoints
@router.get("/users", response_model=List[UserResponse])
async def list_all_users(
    active_only: bool = Query(False, description="Only show active users"),
    search: Optional[str] = Query(None, description="Search by username or email"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    current_user: User = Depends(require_permissions(["manage:users"])),
    db: Session = Depends(get_db),
):
    """List all users with filtering options (admin only)."""
    query = db.query(User)

    if active_only:
        query = query.filter(User.is_active == True)

    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (User.username.ilike(search_pattern)) | (User.email.ilike(search_pattern))
        )

    users = query.offset(skip).limit(limit).all()
    return [UserResponse.model_validate(user) for user in users]


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user_details(
    user_id: int,
    current_user: User = Depends(require_permissions(["manage:users"])),
    security_service: SecurityService = Depends(get_security_service),
):
    """Get detailed user information (admin only)."""
    user = security_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return UserResponse.model_validate(user)


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    update_data: UserUpdateRequest,
    current_user: User = Depends(require_permissions(["manage:users"])),
    security_service: SecurityService = Depends(get_security_service),
    db: Session = Depends(get_db),
):
    """Update user information (admin only)."""
    user = security_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Update fields if provided
    if update_data.username is not None:
        # Check if username is already taken
        existing_user = security_service.get_user_by_username(update_data.username)
        if existing_user and existing_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken"
            )
        user.username = update_data.username

    if update_data.email is not None:
        # Check if email is already taken
        existing_user = security_service.get_user_by_email(update_data.email)
        if existing_user and existing_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email already taken"
            )
        user.email = update_data.email

    if update_data.is_active is not None:
        user.is_active = update_data.is_active

    if update_data.is_verified is not None:
        user.is_verified = update_data.is_verified

    user.updated_at = datetime.now()
    db.commit()
    db.refresh(user)

    return UserResponse.model_validate(user)


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_permissions(["manage:users"])),
    security_service: SecurityService = Depends(get_security_service),
    db: Session = Depends(get_db),
):
    """Delete a user (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    user = security_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Remove user role assignments
    db.query(UserRole).filter(UserRole.user_id == user_id).delete()

    # Delete user
    db.delete(user)
    db.commit()

    return {"message": f"User {user.username} deleted successfully"}


@router.post("/users/bulk-operation")
async def bulk_user_operation(
    operation_data: BulkUserOperation,
    current_user: User = Depends(require_permissions(["manage:users"])),
    db: Session = Depends(get_db),
):
    """Perform bulk operations on users (admin only)."""
    # Prevent self-modification in bulk operations
    if current_user.id in operation_data.user_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot perform bulk operations on your own account",
        )

    users = db.query(User).filter(User.id.in_(operation_data.user_ids)).all()

    if len(users) != len(operation_data.user_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Some users not found"
        )

    if operation_data.operation == "activate":
        for user in users:
            user.is_active = True
        message = f"Activated {len(users)} users"

    elif operation_data.operation == "deactivate":
        for user in users:
            user.is_active = False
        message = f"Deactivated {len(users)} users"

    elif operation_data.operation == "delete":
        # Remove role assignments first
        db.query(UserRole).filter(
            UserRole.user_id.in_(operation_data.user_ids)
        ).delete()
        # Delete users
        for user in users:
            db.delete(user)
        message = f"Deleted {len(users)} users"

    db.commit()
    return {"message": message}


# Role Management Endpoints
@router.get("/roles", response_model=List[RoleResponse])
async def list_all_roles(
    current_user: User = Depends(require_permissions(["manage:roles"])),
    db: Session = Depends(get_db),
):
    """List all roles (admin only)."""
    roles = db.query(Role).all()
    return [RoleResponse.model_validate(role) for role in roles]


@router.post("/roles", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(
    role_data: RoleCreate,
    current_user: User = Depends(require_permissions(["manage:roles"])),
    db: Session = Depends(get_db),
):
    """Create a new role (admin only)."""
    # Check if role already exists
    existing_role = db.query(Role).filter(Role.name == role_data.name).first()
    if existing_role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Role already exists"
        )

    role = Role(
        name=role_data.name,
        description=role_data.description,
        permissions=role_data.permissions,
    )

    db.add(role)
    db.commit()
    db.refresh(role)

    return RoleResponse.model_validate(role)


@router.get("/roles/{role_id}", response_model=RoleResponse)
async def get_role_details(
    role_id: int,
    current_user: User = Depends(require_permissions(["manage:roles"])),
    db: Session = Depends(get_db),
):
    """Get detailed role information (admin only)."""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role not found"
        )

    return RoleResponse.model_validate(role)


@router.put("/roles/{role_id}", response_model=RoleResponse)
async def update_role(
    role_id: int,
    update_data: RoleUpdateRequest,
    current_user: User = Depends(require_permissions(["manage:roles"])),
    db: Session = Depends(get_db),
):
    """Update role information (admin only)."""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role not found"
        )

    # Update fields if provided
    if update_data.name is not None:
        # Check if name is already taken
        existing_role = (
            db.query(Role)
            .filter(Role.name == update_data.name, Role.id != role_id)
            .first()
        )
        if existing_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role name already taken",
            )
        role.name = update_data.name

    if update_data.description is not None:
        role.description = update_data.description

    if update_data.permissions is not None:
        role.permissions = update_data.permissions

    db.commit()
    db.refresh(role)

    return RoleResponse.model_validate(role)


@router.delete("/roles/{role_id}")
async def delete_role(
    role_id: int,
    current_user: User = Depends(require_permissions(["manage:roles"])),
    db: Session = Depends(get_db),
):
    """Delete a role (admin only)."""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role not found"
        )

    # Check if role is assigned to any users
    user_count = db.query(UserRole).filter(UserRole.role_id == role_id).count()
    if user_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete role. It is assigned to {user_count} users.",
        )

    db.delete(role)
    db.commit()

    return {"message": f"Role {role.name} deleted successfully"}


# User-Role Assignment Endpoints
@router.get("/users/{user_id}/roles", response_model=List[RoleResponse])
async def get_user_roles(
    user_id: int,
    current_user: User = Depends(require_permissions(["manage:users"])),
    security_service: SecurityService = Depends(get_security_service),
    db: Session = Depends(get_db),
):
    """Get roles assigned to a user (admin only)."""
    user = security_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    user_roles = db.query(UserRole).join(Role).filter(UserRole.user_id == user_id).all()

    return [RoleResponse.model_validate(ur.role) for ur in user_roles]


@router.post("/users/{user_id}/roles/{role_id}")
async def assign_role_to_user(
    user_id: int,
    role_id: int,
    current_user: User = Depends(require_permissions(["manage:users"])),
    security_service: SecurityService = Depends(get_security_service),
    db: Session = Depends(get_db),
):
    """Assign role to user (admin only)."""
    # Check if user exists
    user = security_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Check if role exists
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role not found"
        )

    # Check if assignment already exists
    existing_assignment = (
        db.query(UserRole)
        .filter(UserRole.user_id == user_id, UserRole.role_id == role_id)
        .first()
    )

    if existing_assignment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role already assigned to user",
        )

    # Create assignment
    user_role = UserRole(user_id=user_id, role_id=role_id)
    db.add(user_role)
    db.commit()

    return {"message": f"Role '{role.name}' assigned to user '{user.username}'"}


@router.delete("/users/{user_id}/roles/{role_id}")
async def remove_role_from_user(
    user_id: int,
    role_id: int,
    current_user: User = Depends(require_permissions(["manage:users"])),
    security_service: SecurityService = Depends(get_security_service),
    db: Session = Depends(get_db),
):
    """Remove role from user (admin only)."""
    # Check if user exists
    user = security_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Check if role exists
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role not found"
        )

    # Find and remove assignment
    assignment = (
        db.query(UserRole)
        .filter(UserRole.user_id == user_id, UserRole.role_id == role_id)
        .first()
    )

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role assignment not found"
        )

    db.delete(assignment)
    db.commit()

    return {"message": f"Role '{role.name}' removed from user '{user.username}'"}


# System Statistics and Monitoring
@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
    current_user: User = Depends(require_permissions(["manage:system"])),
    db: Session = Depends(get_db),
):
    """Get system statistics (admin only)."""
    from datetime import timedelta

    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    total_roles = db.query(Role).count()

    # Count unique permissions across all roles
    all_roles = db.query(Role).all()
    all_permissions = set()
    for role in all_roles:
        if role.permissions:
            all_permissions.update(role.permissions)
    total_permissions = len(all_permissions)

    # Count recent logins (last 24 hours)
    recent_cutoff = datetime.now() - timedelta(hours=24)
    recent_logins = db.query(User).filter(User.last_login >= recent_cutoff).count()

    return SystemStats(
        total_users=total_users,
        active_users=active_users,
        total_roles=total_roles,
        total_permissions=total_permissions,
        recent_logins=recent_logins,
    )


@router.get("/permissions")
async def get_available_permissions(
    current_user: User = Depends(require_permissions(["manage:roles"])),
):
    """Get list of available permissions (admin only)."""
    # Define all available permissions in the system
    permissions = {
        "user_management": [
            "manage:users",
            "read:users",
            "create:user",
            "update:user",
            "delete:user",
        ],
        "role_management": [
            "manage:roles",
            "read:roles",
            "create:role",
            "update:role",
            "delete:role",
        ],
        "strategy_management": [
            "create:strategy",
            "read:strategy",
            "update:strategy",
            "delete:strategy",
            "read:own_strategy",
            "update:own_strategy",
            "delete:own_strategy",
        ],
        "backtest_management": [
            "create:backtest",
            "read:backtest",
            "delete:backtest",
            "read:own_backtest",
            "delete:own_backtest",
        ],
        "data_access": [
            "read:market_data",
            "read:analytics",
            "read:own_data",
            "read:all",
        ],
        "system_management": [
            "manage:system",
            "manage:api_keys",
            "write:all",
            "delete:all",
        ],
    }

    return permissions


@router.get("/audit-log")
async def get_audit_log(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    action: Optional[str] = Query(None, description="Filter by action type"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    current_user: User = Depends(require_permissions(["manage:system"])),
):
    """Get audit log of user actions (admin only)."""
    # This is a placeholder for audit logging functionality
    # In a full implementation, you would have an audit log table
    # that tracks user actions, login attempts, permission changes, etc.

    return {
        "message": "Audit logging not yet implemented",
        "note": "This endpoint would return a list of audit log entries with timestamps, user IDs, actions, and details",
    }
