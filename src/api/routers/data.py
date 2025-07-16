"""Data router for USDC arbitrage API."""

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db

router = APIRouter()


@router.get("/data/", response_model=List[models.USDCDataPydantic])
def read_usdc_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve USDC data with pagination."""
    usdc_data = db.query(models.USDCData).offset(skip).limit(limit).all()
    return usdc_data
