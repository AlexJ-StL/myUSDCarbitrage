

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models
from ..database import SessionLocal

router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/results/", response_model=list[models.BacktestResultPydantic])
def read_results(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    results = db.query(models.BacktestResult).offset(skip).limit(limit).all()
    return results
