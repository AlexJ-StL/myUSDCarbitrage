"""Security service module for USDC arbitrage API."""

from typing import Optional
from sqlalchemy.orm import Session
from ..security import SecurityService as MainSecurityService


class SecurityService(MainSecurityService):
    """Security service wrapper for the main SecurityService."""

    def __init__(self, db: Session):
        super().__init__(db)
