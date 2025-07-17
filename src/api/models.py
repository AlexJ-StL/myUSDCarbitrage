"""Data models for the USDC arbitrage application."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import JSON, Column, DateTime, Float, Integer, String

from .database import Base


class USDCData(Base):
    """SQLAlchemy model for USDC price data."""

    __tablename__ = "usdc_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    exchange = Column(String, index=True)
    price = Column(Float)


class USDCDataPydantic(BaseModel):
    """Pydantic model for USDC data API responses."""

    id: int
    timestamp: datetime
    exchange: str
    price: float

    model_config = ConfigDict(from_attributes=True)


class Strategy(Base):
    """SQLAlchemy model for trading strategies."""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    parameters = Column(JSON)


class StrategyPydantic(BaseModel):
    """Pydantic model for strategy API responses."""

    id: int
    name: str
    description: str
    parameters: dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class BacktestResult(Base):
    """SQLAlchemy model for backtest results."""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    results = Column(JSON)


class BacktestResultPydantic(BaseModel):
    """Pydantic model for backtest result API responses."""

    id: int
    strategy_id: int
    start_date: datetime
    end_date: datetime
    results: dict[str, Any]

    model_config = ConfigDict(from_attributes=True)
