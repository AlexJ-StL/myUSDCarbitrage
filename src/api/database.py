"""Database configuration and connection management for USDC arbitrage application."""

import os
from collections.abc import Generator
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")
    DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class OHLCVData(Base):
    __tablename__ = "market_data"
    id = Column(Integer, primary_key=True, index=True)
    exchange = Column(String, index=True)
    symbol = Column(String, index=True)
    timeframe = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DBConnector:
    """Database connector for raw SQL queries and operations."""

    def __init__(self, connection_string: str):
        """Initialize database connector with connection string."""
        self.connection_string = connection_string
        self.engine = create_engine(self.connection_string)

    def get_ohlcv_data(
        self, exchange: str, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """Retrieve OHLCV data for given exchange, symbol, and timeframe."""
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
        ORDER BY timestamp
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text(query),
                    {"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
                )
                columns = ["timestamp", "open", "high", "low", "close", "volume"]
                df = pd.DataFrame(result.fetchall(), columns=columns)
                return df
        except Exception as e:
            raise e

    def get_user_roles(self, username: str) -> list[str]:
        """Get user roles for authorization."""
        query = """
        SELECT r.name as role
        FROM user_roles ur
        JOIN roles r ON ur.role_id = r.id
        JOIN users u ON ur.user_id = u.id
        WHERE u.username = :username
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), {"username": username})
                roles = [row[0] for row in result.fetchall()]
                return roles
        except Exception as e:
            raise e


class Database:
    """Database operations using SQLAlchemy ORM."""

    def __init__(self):
        """Initialize database connection."""
        self.db = SessionLocal()

    def insert_data(
        self, exchange: str, symbol: str, timeframe: str, data: list
    ) -> None:
        """Insert OHLCV data into database."""
        try:
            for entry in data:
                ohlcv_data = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": datetime.fromtimestamp(entry[0] / 1000),
                    "open": entry[1],
                    "high": entry[2],
                    "low": entry[3],
                    "close": entry[4],
                    "volume": entry[5],
                }
                ohlcv_entry = OHLCVData(**ohlcv_data)
                self.db.add(ohlcv_entry)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise e

    def close(self) -> None:
        """Close database connection."""
        self.db.close()
