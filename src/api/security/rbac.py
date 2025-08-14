"""Role-Based Access Control (RBAC) utilities."""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from .. import models


def check_strategy_read_access(
    user: models.User, strategy_id: int, db: Session
) -> bool:
    """Check if user has read access to a strategy."""
    # Get user permissions
    user_permissions = []
    for user_role in user.roles:
        user_permissions.extend(user_role.role.permissions)

    # Admin can read all strategies
    if "admin" in user_permissions or "read:strategy" in user_permissions:
        return True

    # Check if user owns the strategy
    if "read:own_strategy" in user_permissions:
        strategy = (
            db.query(models.Strategy).filter(models.Strategy.id == strategy_id).first()
        )

        if strategy and strategy.created_by == user.id:
            return True

    # Access denied
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not enough permissions to read this strategy",
    )


def check_strategy_write_access(
    user: models.User, strategy_id: int, db: Session
) -> bool:
    """Check if user has write access to a strategy."""
    # Get user permissions
    user_permissions = []
    for user_role in user.roles:
        user_permissions.extend(user_role.role.permissions)

    # Admin can write all strategies
    if "admin" in user_permissions or "update:strategy" in user_permissions:
        return True

    # Check if user owns the strategy
    if "update:own_strategy" in user_permissions:
        strategy = (
            db.query(models.Strategy).filter(models.Strategy.id == strategy_id).first()
        )

        if strategy and strategy.created_by == user.id:
            return True

    # Access denied
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not enough permissions to update this strategy",
    )


def check_strategy_delete_access(
    user: models.User, strategy_id: int, db: Session
) -> bool:
    """Check if user has delete access to a strategy."""
    # Get user permissions
    user_permissions = []
    for user_role in user.roles:
        user_permissions.extend(user_role.role.permissions)

    # Admin can delete all strategies
    if "admin" in user_permissions or "delete:strategy" in user_permissions:
        return True

    # Check if user owns the strategy
    if "delete:own_strategy" in user_permissions:
        strategy = (
            db.query(models.Strategy).filter(models.Strategy.id == strategy_id).first()
        )

        if strategy and strategy.created_by == user.id:
            return True

    # Access denied
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not enough permissions to delete this strategy",
    )
