"""API key management system for external integrations."""

import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import Session

from .database import Base, get_db


class APIKey(Base):
    """SQLAlchemy model for API keys."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), unique=True, index=True, nullable=False)
    key_prefix = Column(
        String(10), index=True, nullable=False
    )  # First 8 chars for identification
    user_id = Column(Integer, nullable=True)  # Optional user association
    permissions = Column(Text, nullable=True)  # JSON string of permissions
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)
    rate_limit = Column(Integer, default=1000)  # Requests per hour
    created_at = Column(DateTime, default=datetime.now)
    created_by = Column(Integer, nullable=True)  # User who created the key
    description = Column(Text, nullable=True)
    allowed_ips = Column(Text, nullable=True)  # JSON array of allowed IP addresses


class APIKeyCreate(BaseModel):
    """Pydantic model for API key creation."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(None, gt=0, le=365)
    rate_limit: int = Field(default=1000, gt=0, le=10000)
    allowed_ips: Optional[List[str]] = None


class APIKeyResponse(BaseModel):
    """Pydantic model for API key response."""

    id: int
    name: str
    key_prefix: str
    permissions: List[str]
    is_active: bool
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    rate_limit: int
    created_at: datetime
    description: Optional[str]
    allowed_ips: Optional[List[str]]


class APIKeyWithSecret(BaseModel):
    """Pydantic model for API key with secret (only returned on creation)."""

    id: int
    name: str
    api_key: str  # Full API key - only shown once
    key_prefix: str
    permissions: List[str]
    expires_at: Optional[datetime]
    rate_limit: int
    created_at: datetime


class APIKeyUpdate(BaseModel):
    """Pydantic model for API key updates."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None
    rate_limit: Optional[int] = Field(None, gt=0, le=10000)
    allowed_ips: Optional[List[str]] = None


class APIKeyService:
    """Service for managing API keys."""

    def __init__(self, db: Session):
        self.db = db

    def generate_api_key(self) -> tuple[str, str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (full_key, key_hash, key_prefix)
        """
        # Generate a secure random key
        alphabet = string.ascii_letters + string.digits
        key = "ak_" + "".join(secrets.choice(alphabet) for _ in range(40))

        # Create hash for storage
        import hashlib

        key_hash = hashlib.sha256(key.encode()).hexdigest()

        # Get prefix for identification
        key_prefix = key[:8]

        return key, key_hash, key_prefix

    def create_api_key(
        self, key_data: APIKeyCreate, created_by: Optional[int] = None
    ) -> APIKeyWithSecret:
        """Create a new API key."""
        # Generate key
        full_key, key_hash, key_prefix = self.generate_api_key()

        # Calculate expiration
        expires_at = None
        if key_data.expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(
                days=key_data.expires_in_days
            )

        # Create database record
        api_key = APIKey(
            name=key_data.name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            permissions=",".join(key_data.permissions) if key_data.permissions else "",
            expires_at=expires_at,
            rate_limit=key_data.rate_limit,
            created_by=created_by,
            description=key_data.description,
            allowed_ips=",".join(key_data.allowed_ips)
            if key_data.allowed_ips
            else None,
        )

        self.db.add(api_key)
        self.db.commit()
        self.db.refresh(api_key)

        return APIKeyWithSecret(
            id=api_key.id,
            name=api_key.name,
            api_key=full_key,  # Only returned once
            key_prefix=api_key.key_prefix,
            permissions=key_data.permissions,
            expires_at=api_key.expires_at,
            rate_limit=api_key.rate_limit,
            created_at=api_key.created_at,
        )

    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash."""
        return self.db.query(APIKey).filter(APIKey.key_hash == key_hash).first()

    def get_api_key_by_id(self, key_id: int) -> Optional[APIKey]:
        """Get API key by ID."""
        return self.db.query(APIKey).filter(APIKey.id == key_id).first()

    def list_api_keys(self, created_by: Optional[int] = None) -> List[APIKey]:
        """List API keys, optionally filtered by creator."""
        query = self.db.query(APIKey)
        if created_by:
            query = query.filter(APIKey.created_by == created_by)
        return query.all()

    def update_api_key(
        self, key_id: int, update_data: APIKeyUpdate
    ) -> Optional[APIKey]:
        """Update an API key."""
        api_key = self.get_api_key_by_id(key_id)
        if not api_key:
            return None

        if update_data.name is not None:
            api_key.name = update_data.name

        if update_data.description is not None:
            api_key.description = update_data.description

        if update_data.permissions is not None:
            api_key.permissions = ",".join(update_data.permissions)

        if update_data.is_active is not None:
            api_key.is_active = update_data.is_active

        if update_data.rate_limit is not None:
            api_key.rate_limit = update_data.rate_limit

        if update_data.allowed_ips is not None:
            api_key.allowed_ips = (
                ",".join(update_data.allowed_ips) if update_data.allowed_ips else None
            )

        self.db.commit()
        self.db.refresh(api_key)
        return api_key

    def delete_api_key(self, key_id: int) -> bool:
        """Delete an API key."""
        api_key = self.get_api_key_by_id(key_id)
        if not api_key:
            return False

        self.db.delete(api_key)
        self.db.commit()
        return True

    def validate_api_key(self, api_key: str, client_ip: str = None) -> Optional[APIKey]:
        """
        Validate an API key and return the associated record.

        Args:
            api_key: The API key to validate
            client_ip: Client IP address for IP restriction checking

        Returns:
            APIKey record if valid, None otherwise
        """
        import hashlib

        # Hash the provided key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Find the key in database
        db_key = self.get_api_key_by_hash(key_hash)
        if not db_key:
            return None

        # Check if key is active
        if not db_key.is_active:
            return None

        # Check expiration
        if db_key.expires_at and datetime.now(timezone.utc) > db_key.expires_at:
            return None

        # Check IP restrictions
        if db_key.allowed_ips and client_ip:
            allowed_ips = db_key.allowed_ips.split(",")
            if client_ip not in allowed_ips:
                return None

        # Update usage statistics
        db_key.last_used_at = datetime.now(timezone.utc)
        db_key.usage_count += 1
        self.db.commit()

        return db_key

    def get_api_key_permissions(self, api_key: APIKey) -> List[str]:
        """Get permissions for an API key."""
        if not api_key.permissions:
            return []
        return api_key.permissions.split(",")


# Dependency functions
def get_api_key_service(db: Session = Depends(get_db)) -> APIKeyService:
    """Get API key service instance."""
    return APIKeyService(db)


def validate_api_key_header(
    request,
    api_key_service: APIKeyService = Depends(get_api_key_service),
) -> Optional[APIKey]:
    """Validate API key from request headers."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return None

    client_ip = request.client.host
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    return api_key_service.validate_api_key(api_key, client_ip)


def require_api_key(
    request,
    api_key_service: APIKeyService = Depends(get_api_key_service),
) -> APIKey:
    """Require valid API key for endpoint access."""
    api_key = validate_api_key_header(request, api_key_service)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


def require_api_key_permissions(required_permissions: List[str]):
    """Decorator to require specific API key permissions."""

    def permission_checker(
        api_key: APIKey = Depends(require_api_key),
        api_key_service: APIKeyService = Depends(get_api_key_service),
    ):
        key_permissions = api_key_service.get_api_key_permissions(api_key)

        for permission in required_permissions:
            if permission not in key_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"API key missing required permission: {permission}",
                )

        return api_key

    return permission_checker


# Utility functions
def format_api_key_response(api_key: APIKey) -> APIKeyResponse:
    """Format API key for response."""
    permissions = api_key.permissions.split(",") if api_key.permissions else []
    allowed_ips = api_key.allowed_ips.split(",") if api_key.allowed_ips else None

    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        permissions=permissions,
        is_active=api_key.is_active,
        expires_at=api_key.expires_at,
        last_used_at=api_key.last_used_at,
        usage_count=api_key.usage_count,
        rate_limit=api_key.rate_limit,
        created_at=api_key.created_at,
        description=api_key.description,
        allowed_ips=allowed_ips,
    )
