from .database import DBConnector
from .security import get_current_user
from fastapi import Depends, HTTPException, status


def get_database():
    connection_string = (
        "postgresql://arb_user:strongpassword@localhost:5432/usdc_arbitrage"
    )
    db = DBConnector(connection_string)
    try:
        yield db
    finally:
        db.disconnect()


def get_admin_user(
    current_user: str = Depends(get_current_user),
    db: DBConnector = Depends(get_database),
):
    if "admin" not in db.get_user_roles(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )
    return current_user
