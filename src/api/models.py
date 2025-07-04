
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from .database import Base
from pydantic import BaseModel, ConfigDict
from datetime import datetime

class USDCData(Base):
    __tablename__ = "usdc_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    exchange = Column(String, index=True)
    price = Column(Float)

class USDCDataPydantic(BaseModel):
    id: int
    timestamp: datetime
    exchange: str
    price: float

    model_config = ConfigDict(from_attributes=True)

class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    parameters = Column(JSON)

class StrategyPydantic(BaseModel):
    id: int
    name: str
    description: str
    parameters: dict

    model_config = ConfigDict(from_attributes=True)

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    results = Column(JSON)

class BacktestResultPydantic(BaseModel):
    id: int
    strategy_id: int
    start_date: datetime
    end_date: datetime
    results: dict

    model_config = ConfigDict(from_attributes=True)
