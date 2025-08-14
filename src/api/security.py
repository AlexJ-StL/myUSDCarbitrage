"""Enhanced security module with JWT authentication, token refresh, and blacklisting."""

import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .database import get_db
from .models import (
    RefreshToken,
    Role,
    TokenBlacklist,
    User,
    UserRole,
)

# Load environment variables
load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


class SecurityService:
    """Service class for authentication and authorization operations."""

    def __init__(self, db: Session):
        self.db = db

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token with configurable expiration."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=ACCESS_TOKEN_EXPIRE_MINUTES
            )

        # Add standard JWT claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid.uuid4()),  # JWT ID for blacklisting
            "type": "access",
        })

        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def create_refresh_token(self, user_id: int) -> str:
        """Create and store refresh token."""
        # Generate token
        token_data = {
            "sub": str(user_id),
            "type": "refresh",
            "jti": str(uuid.uuid4()),
            "iat": datetime.now(timezone.utc),
        }

        expires_at = datetime.now(timezone.utc) + timedelta(
            days=REFRESH_TOKEN_EXPIRE_DAYS
        )
        token_data["exp"] = expires_at

        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

        # Store in database
        refresh_token = RefreshToken(
            token=token, user_id=user_id, expires_at=expires_at
        )
        self.db.add(refresh_token)
        self.db.commit()

        return token

    def verify_token(self, token: str, token_type: str = "access") -> dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            # Check token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}",
                )

            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti and self.is_token_blacklisted(jti):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                )

            return payload

        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    def is_token_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted."""
        blacklisted = (
            self.db.query(TokenBlacklist).filter(TokenBlacklist.jti == jti).first()
        )
        return blacklisted is not None

    def blacklist_token(
        self, jti: str, token_type: str, expires_at: datetime, reason: str = "logout"
    ):
        """Add token to blacklist."""
        blacklist_entry = TokenBlacklist(
            jti=jti, token_type=token_type, expires_at=expires_at, reason=reason
        )
        self.db.add(blacklist_entry)
        self.db.commit()

    def revoke_refresh_token(self, token: str):
        """Revoke a refresh token."""
        refresh_token = (
            self.db.query(RefreshToken).filter(RefreshToken.token == token).first()
        )

        if refresh_token:
            refresh_token.is_revoked = True
            refresh_token.revoked_at = datetime.now(timezone.utc)
            self.db.commit()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        user = self.db.query(User).filter(User.username == username).first()

        if not user or not user.is_active:
            return None

        if not self.verify_password(password, user.password_hash):
            return None

        # Update last login
        user.last_login = datetime.now(timezone.utc)
        self.db.commit()

        return user

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.db.query(User).filter(User.username == username).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()

    def create_user(self, username: str, email: str, password: str) -> User:
        """Create a new user."""
        # Check if user already exists
        if self.get_user_by_username(username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered",
            )

        if self.get_user_by_email(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Create user
        hashed_password = self.get_password_hash(password)
        user = User(username=username, email=email, password_hash=hashed_password)

        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        # Assign default role
        self.assign_default_role(user.id)

        return user

    def assign_default_role(self, user_id: int):
        """Assign default 'user' role to new user."""
        # Get or create default user role
        user_role = self.db.query(Role).filter(Role.name == "user").first()
        if not user_role:
            user_role = Role(
                name="user",
                description="Default user role",
                permissions=["read:own_data", "create:backtest", "read:strategies"],
            )
            self.db.add(user_role)
            self.db.commit()
            self.db.refresh(user_role)

        # Assign role to user
        user_role_assignment = UserRole(user_id=user_id, role_id=user_role.id)
        self.db.add(user_role_assignment)
        self.db.commit()

    def get_user_permissions(self, user_id: int) -> List[str]:
        """Get all permissions for a user."""
        permissions = set()

        user_roles = (
            self.db.query(UserRole).join(Role).filter(UserRole.user_id == user_id).all()
        )

        for user_role in user_roles:
            role_permissions = user_role.role.permissions or []
            permissions.update(role_permissions)

        return list(permissions)

    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """Refresh access token using refresh token."""
        # Verify refresh token
        try:
            payload = self.verify_token(refresh_token, "refresh")
        except HTTPException:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

        user_id = int(payload.get("sub"))

        # Check if refresh token exists and is not revoked
        stored_token = (
            self.db.query(RefreshToken)
            .filter(
                RefreshToken.token == refresh_token,
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False,
            )
            .first()
        )

        if not stored_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token not found or revoked",
            )

        # Get user
        user = self.get_user_by_id(user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )

        # Create new tokens
        access_token_data = {
            "sub": user.username,
            "user_id": user.id,
            "permissions": self.get_user_permissions(user.id),
        }

        new_access_token = self.create_access_token(access_token_data)
        new_refresh_token = self.create_refresh_token(user.id)

        # Revoke old refresh token
        self.revoke_refresh_token(refresh_token)

        return new_access_token, new_refresh_token


# Dependency functions
def get_security_service(db: Session = Depends(get_db)) -> SecurityService:
    """Get security service instance."""
    return SecurityService(db)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    security_service: SecurityService = Depends(get_security_service),
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = security_service.verify_token(token, "access")
        user_id = payload.get("user_id")
        if user_id is None:
            raise credentials_exception

        user = security_service.get_user_by_id(user_id)
        if user is None:
            raise credentials_exception

        return user

    except HTTPException:
        raise credentials_exception


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )
    return current_user


def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions."""

    def permission_checker(
        current_user: User = Depends(get_current_active_user),
        security_service: SecurityService = Depends(get_security_service),
    ):
        user_permissions = security_service.get_user_permissions(current_user.id)

        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied. Required: {permission}",
                )

        return current_user

    return permission_checker


def require_roles(required_roles: List[str]):
    """Decorator to require specific roles."""

    def role_checker(
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db),
    ):
        user_roles = (
            db.query(UserRole)
            .join(Role)
            .filter(UserRole.user_id == current_user.id)
            .all()
        )

        user_role_names = [ur.role.name for ur in user_roles]

        for role in required_roles:
            if role not in user_role_names:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required role: {role}",
                )

        return current_user

    return role_checker


def require_resource_access(resource_type: str, permission_type: str = "read"):
    """
    Decorator to require resource-level access control.

    Args:
        resource_type: Type of resource (strategy, backtest, user)
        permission_type: Type of permission (read, write, delete)
    """

    def resource_checker(
        resource_id: int,
        current_user: User = Depends(get_current_active_user),
        security_service: SecurityService = Depends(get_security_service),
        db: Session = Depends(get_db),
    ):
        # Check if user has admin privileges
        user_permissions = security_service.get_user_permissions(current_user.id)

        # Admin users have access to all resources
        if "read:all" in user_permissions or "manage:system" in user_permissions:
            return current_user

        # Check specific resource permissions
        required_permission = f"{permission_type}:{resource_type}"
        own_resource_permission = f"{permission_type}:own_{resource_type}"

        # If user has general permission for this resource type
        if required_permission in user_permissions:
            return current_user

        # Check if user owns the resource and has own_resource permission
        if own_resource_permission in user_permissions:
            if _user_owns_resource(db, current_user.id, resource_type, resource_id):
                return current_user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied. Insufficient permissions for {resource_type} {resource_id}",
        )

    return resource_checker


def _user_owns_resource(
    db: Session, user_id: int, resource_type: str, resource_id: int
) -> bool:
    """Check if user owns a specific resource."""
    if resource_type == "strategy":
        from .models import Strategy

        strategy = db.query(Strategy).filter(Strategy.id == resource_id).first()
        # For now, we'll check if user created the strategy (would need to add created_by field)
        # This is a simplified check - in production you'd have proper ownership tracking
        return True  # Placeholder - implement proper ownership logic

    elif resource_type == "backtest":
        from .models import BacktestResult

        backtest = (
            db.query(BacktestResult).filter(BacktestResult.id == resource_id).first()
        )
        # Similar ownership check for backtests
        return True  # Placeholder - implement proper ownership logic

    elif resource_type == "user":
        # Users can only access their own user record
        return user_id == resource_id

    return False


def create_strategy_access_checker(permission_type: str = "read"):
    """Create a dependency function to check strategy access."""

    def check_strategy_access(
        strategy_id: int,
        current_user: User = Depends(get_current_active_user),
        security_service: SecurityService = Depends(get_security_service),
        db: Session = Depends(get_db),
    ) -> User:
        """Check if user has access to a specific strategy."""
        user_permissions = security_service.get_user_permissions(current_user.id)

        # Admin users have full access
        if "manage:system" in user_permissions or "read:all" in user_permissions:
            return current_user

        # Check specific strategy permissions
        strategy_permission = f"{permission_type}:strategy"
        own_strategy_permission = f"{permission_type}:own_strategy"

        if strategy_permission in user_permissions:
            return current_user

        # Check if user owns the strategy
        if own_strategy_permission in user_permissions:
            from .models import Strategy

            strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
            if strategy:
                # In a full implementation, you'd check actual ownership
                # For now, we'll allow access if user has the own_strategy permission
                return current_user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to strategy {strategy_id}",
        )

    return check_strategy_access


# Convenience functions for common access patterns
def check_strategy_read_access():
    """Dependency to check strategy read access."""
    return create_strategy_access_checker("read")


def check_strategy_write_access():
    """Dependency to check strategy write access."""
    return create_strategy_access_checker("update")


def check_strategy_delete_access():
    """Dependency to check strategy delete access."""
    return create_strategy_access_checker("delete")


def check_backtest_access(
    backtest_id: int,
    permission_type: str = "read",
    current_user: User = Depends(get_current_active_user),
    security_service: SecurityService = Depends(get_security_service),
    db: Session = Depends(get_db),
) -> User:
    """Check if user has access to a specific backtest."""
    user_permissions = security_service.get_user_permissions(current_user.id)

    # Admin users have full access
    if "manage:system" in user_permissions or "read:all" in user_permissions:
        return current_user

    # Check specific backtest permissions
    backtest_permission = f"{permission_type}:backtest"
    own_backtest_permission = f"{permission_type}:own_backtest"

    if backtest_permission in user_permissions:
        return current_user

    # Check if user owns the backtest
    if own_backtest_permission in user_permissions:
        from .models import BacktestResult

        backtest = (
            db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()
        )
        if backtest:
            # In a full implementation, you'd check actual ownership
            # For now, we'll allow access if user has the own_backtest permission
            return current_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Access denied to backtest {backtest_id}",
    )


# Legacy functions for backward compatibility
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Legacy function for password verification."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Legacy function for password hashing."""
    return pwd_context.hash(password)
