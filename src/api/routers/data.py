
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models
from ..database import SessionLocal
from typing import List

router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/data/", response_model=List[models.USDCData])
def read_usdc_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    usdc_data = db.query(models.USDCData).offset(skip).limit(limit).all()
    return usdc_data
