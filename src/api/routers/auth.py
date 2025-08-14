"""Authentication router with JWT-based user management."""

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (
    Role,
    RoleCreate,
    RoleResponse,
    TokenRefresh,
    TokenResponse,
    User,
    UserCreate,
    UserLogin,
    UserResponse,
)
from ..security import (
    SecurityService,
    get_current_active_user,
    get_security_service,
    require_roles,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register_user(
    user_data: UserCreate,
    security_service: SecurityService = Depends(get_security_service),
):
    """Register a new user."""
    try:
        user = security_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
        )
        return UserResponse.model_validate(user)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}",
        ) from e


@router.post("/login", response_model=TokenResponse)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    security_service: SecurityService = Depends(get_security_service),
):
    """Authenticate user and return JWT tokens."""
    user = security_service.authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create tokens
    access_token_data = {
        "sub": user.username,
        "user_id": user.id,
        "permissions": security_service.get_user_permissions(user.id),
    }

    access_token = security_service.create_access_token(access_token_data)
    refresh_token = security_service.create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800,  # 30 minutes in seconds
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: TokenRefresh,
    security_service: SecurityService = Depends(get_security_service),
):
    """Refresh access token using refresh token."""
    try:
        new_access_token, new_refresh_token = security_service.refresh_access_token(
            token_data.refresh_token
        )

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=1800,  # 30 minutes in seconds
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh token: {str(e)}",
        ) from e


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout_user(
    token_data: TokenRefresh,
    current_user: User = Depends(get_current_active_user),
    security_service: SecurityService = Depends(get_security_service),
):
    """Logout user by blacklisting tokens."""
    try:
        # Get current access token from the request
        # Note: In a real implementation, you'd extract this from the Authorization header
        # For now, we'll revoke the refresh token

        # Revoke refresh token
        security_service.revoke_refresh_token(token_data.refresh_token)

        # In a complete implementation, you'd also blacklist the current access token
        # This would require extracting the JWT from the Authorization header

        return {"message": "Successfully logged out"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to logout: {str(e)}",
        ) from e


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
):
    """Get current user information."""
    return UserResponse.model_validate(current_user)


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: User = Depends(require_roles(["admin"])),
    db: Session = Depends(get_db),
):
    """List all users (admin only)."""
    users = db.query(User).all()
    return [UserResponse.model_validate(user) for user in users]


@router.post("/roles", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(
    role_data: RoleCreate,
    current_user: User = Depends(require_roles(["admin"])),
    db: Session = Depends(get_db),
):
    """Create a new role (admin only)."""
    # Check if role already exists
    existing_role = db.query(Role).filter(Role.name == role_data.name).first()
    if existing_role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role already exists",
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


@router.get("/roles", response_model=List[RoleResponse])
async def list_roles(
    current_user: User = Depends(require_roles(["admin"])),
    db: Session = Depends(get_db),
):
    """List all roles (admin only)."""
    roles = db.query(Role).all()
    return [RoleResponse.model_validate(role) for role in roles]


@router.post("/users/{user_id}/roles/{role_id}", status_code=status.HTTP_200_OK)
async def assign_role_to_user(
    user_id: int,
    role_id: int,
    current_user: User = Depends(require_roles(["admin"])),
    security_service: SecurityService = Depends(get_security_service),
    db: Session = Depends(get_db),
):
    """Assign role to user (admin only)."""
    # Check if user exists
    user = security_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Check if role exists
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role not found",
        )

    # Check if assignment already exists
    from ..models import UserRole

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


@router.delete("/users/{user_id}/roles/{role_id}", status_code=status.HTTP_200_OK)
async def remove_role_from_user(
    user_id: int,
    role_id: int,
    current_user: User = Depends(require_roles(["admin"])),
    security_service: SecurityService = Depends(get_security_service),
    db: Session = Depends(get_db),
):
    """Remove role from user (admin only)."""
    # Check if user exists
    user = security_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Check if role exists
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role not found",
        )

    # Find and remove assignment
    from ..models import UserRole

    assignment = (
        db.query(UserRole)
        .filter(UserRole.user_id == user_id, UserRole.role_id == role_id)
        .first()
    )

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role assignment not found",
        )

    db.delete(assignment)
    db.commit()

    return {"message": f"Role '{role.name}' removed from user '{user.username}'"}


@router.get("/permissions")
async def get_user_permissions(
    current_user: User = Depends(get_current_active_user),
    security_service: SecurityService = Depends(get_security_service),
):
    """Get current user's permissions."""
    permissions = security_service.get_user_permissions(current_user.id)
    return {"permissions": permissions}
