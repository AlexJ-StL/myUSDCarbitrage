
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from .database import Base

class USDCData(Base):
    __tablename__ = "usdc_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    exchange = Column(String, index=True)
    price = Column(Float)

class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    parameters = Column(JSON)

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    results = Column(JSON)
