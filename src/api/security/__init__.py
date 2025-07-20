"""Security module for USDC arbitrage API."""

# Import from the main security.py file
from ..security import SecurityService

# Import specific functions from security modules
from .jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_active_user,
    get_current_user,
    require_permissions,
    verify_password,
    get_password_hash,
)

from .rbac import (
    check_strategy_read_access,
    check_strategy_write_access,
    check_strategy_delete_access,
)

from .websocket_auth import get_current_user_ws

__all__ = [
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "get_current_active_user",
    "get_current_user",
    "require_permissions",
    "verify_password",
    "get_password_hash",
    "check_strategy_read_access",
    "check_strategy_write_access",
    "check_strategy_delete_access",
    "get_current_user_ws",
    "SecurityService",
]
