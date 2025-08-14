"""WebSocket authentication utilities."""

import logging
from typing import Optional

from fastapi import WebSocket
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from .jwt import ALGORITHM, SECRET_KEY

# Configure logging
logger = logging.getLogger(__name__)


async def get_current_user_ws(
    websocket: WebSocket, db: Session
) -> Optional[models.User]:
    """Authenticate WebSocket connection using JWT token."""
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            # Try getting token from headers
            auth_header = websocket.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            logger.warning("No authentication token provided for WebSocket connection")
            return None

        # Verify token
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if not username:
                logger.warning("Invalid token payload: missing subject")
                return None

            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti:
                blacklisted = (
                    db.query(models.TokenBlacklist)
                    .filter(models.TokenBlacklist.jti == jti)
                    .first()
                )

                if blacklisted:
                    logger.warning(f"Blacklisted token used: {jti}")
                    return None

            # Get user from database
            user = (
                db.query(models.User)
                .filter(models.User.username == username, models.User.is_active == True)
                .first()
            )

            if not user:
                logger.warning(f"User not found or inactive: {username}")
                return None

            return user

        except JWTError as e:
            logger.warning(f"JWT validation error: {e}")
            return None

    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return None
