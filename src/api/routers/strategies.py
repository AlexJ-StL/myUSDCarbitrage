

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

@router.post("/strategies/", response_model=models.StrategyPydantic)
def create_strategy(strategy: models.StrategyPydantic, db: Session = Depends(get_db)):
    db_strategy = models.Strategy(**strategy.dict())
    db.add(db_strategy)
    db.commit()
    db.refresh(db_strategy)
    return db_strategy

@router.get("/strategies/", response_model=list[models.StrategyPydantic])
def read_strategies(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    strategies = db.query(models.Strategy).offset(skip).limit(limit).all()
    return strategies
