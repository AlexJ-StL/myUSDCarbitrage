
from fastapi import APIRouter, Depends
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

@router.get("/results/", response_model=List[models.BacktestResultPydantic])
def read_results(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    results = db.query(models.BacktestResult).offset(skip).limit(limit).all()
    return results
