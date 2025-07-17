"""Database configuration and connection management for USDC arbitrage application."""

import logging
import os
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Boolean,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("database.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

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
    inserted_at = Column(DateTime, default=datetime.now(UTC))
    updated_at = Column(DateTime, default=datetime.now(UTC), onupdate=datetime.now(UTC))
    backfilled = Column(Boolean, default=False)
    source_exchange = Column(String, nullable=True)


# Create indexes for common queries
# Index for timestamp + exchange + symbol + timeframe combination
# Index for inserted_at for conflict resolution


class DataGap(Base):
    __tablename__ = "data_gaps"
    id = Column(Integer, primary_key=True, index=True)
    exchange = Column(String, index=True)
    symbol = Column(String, index=True)
    timeframe = Column(String, index=True)
    start_timestamp = Column(DateTime, index=True)
    end_timestamp = Column(DateTime, index=True)
    gap_duration = Column(Integer)  # Duration in seconds
    detected_at = Column(DateTime, default=datetime.now(UTC))
    filled = Column(Boolean, default=False)
    filled_at = Column(DateTime, nullable=True)
    filled_from_exchange = Column(String, nullable=True)


class DataValidationIssue(Base):
    __tablename__ = "data_validation_issues"
    id = Column(Integer, primary_key=True, index=True)
    exchange = Column(String, index=True)
    symbol = Column(String, index=True)
    timeframe = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    issue_type = Column(String, index=True)
    severity = Column(String, index=True)
    message = Column(String)
    detected_at = Column(DateTime, default=datetime.now(UTC))
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    resolution_method = Column(String, nullable=True)


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
            logger.error(f"Error retrieving OHLCV data: {e}")
            raise e

    def get_ohlcv_data_range(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Retrieve OHLCV data for a specific date range."""
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE exchange = :exchange 
        AND symbol = :symbol 
        AND timeframe = :timeframe
        AND timestamp BETWEEN :start_date AND :end_date
        ORDER BY timestamp
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text(query),
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                )
                columns = ["timestamp", "open", "high", "low", "close", "volume"]
                df = pd.DataFrame(result.fetchall(), columns=columns)
                return df
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data range: {e}")
            raise e

    def get_last_timestamp(
        self, exchange: str, symbol: str, timeframe: str
    ) -> Optional[int]:
        """Get the last timestamp for a given exchange, symbol, and timeframe."""
        query = """
        SELECT EXTRACT(EPOCH FROM MAX(timestamp)) * 1000 as last_timestamp
        FROM market_data
        WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text(query),
                    {"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
                )
                row = result.fetchone()
                return int(row[0]) if row and row[0] else None
        except Exception as e:
            logger.error(f"Error retrieving last timestamp: {e}")
            return None

    def get_duplicate_timestamps(
        self, exchange: str, symbol: str, timeframe: str
    ) -> List[int]:
        """Get timestamps with duplicate entries."""
        query = """
        SELECT EXTRACT(EPOCH FROM timestamp) * 1000 as ts
        FROM market_data
        WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
        GROUP BY timestamp
        HAVING COUNT(*) > 1
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text(query),
                    {"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
                )
                return [int(row[0]) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Error retrieving duplicate timestamps: {e}")
            return []

    def get_records_by_timestamp(
        self, exchange: str, symbol: str, timeframe: str, timestamp: int
    ) -> List[Dict[str, Any]]:
        """Get all records for a specific timestamp."""
        query = """
        SELECT id, open, high, low, close, volume, 
               EXTRACT(EPOCH FROM timestamp) * 1000 as timestamp,
               inserted_at, updated_at, backfilled, source_exchange
        FROM market_data
        WHERE exchange = :exchange 
        AND symbol = :symbol 
        AND timeframe = :timeframe
        AND EXTRACT(EPOCH FROM timestamp) * 1000 = :timestamp
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text(query),
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": timestamp,
                    },
                )
                columns = [
                    "id",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "timestamp",
                    "inserted_at",
                    "updated_at",
                    "backfilled",
                    "source_exchange",
                ]
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Error retrieving records by timestamp: {e}")
            return []

    def get_records_by_timestamps(
        self, exchange: str, symbol: str, timeframe: str, timestamps: List[int]
    ) -> List[Dict[str, Any]]:
        """Get records for multiple timestamps."""
        if not timestamps:
            return []

        placeholders = ", ".join([f":{i}" for i in range(len(timestamps))])
        query = f"""
        SELECT id, open, high, low, close, volume, 
               EXTRACT(EPOCH FROM timestamp) * 1000 as timestamp
        FROM market_data
        WHERE exchange = :exchange 
        AND symbol = :symbol 
        AND timeframe = :timeframe
        AND EXTRACT(EPOCH FROM timestamp) * 1000 IN ({placeholders})
        """
        try:
            with self.engine.connect() as connection:
                params = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                }
                for i, ts in enumerate(timestamps):
                    params[str(i)] = ts

                result = connection.execute(text(query), params)
                columns = ["id", "open", "high", "low", "close", "volume", "timestamp"]
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Error retrieving records by timestamps: {e}")
            return []

    def get_missing_timestamps(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[int]:
        """Get missing timestamps in a date range based on expected timeframe intervals."""
        # First, determine the timeframe duration in seconds
        timeframe_durations = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }

        if timeframe not in timeframe_durations:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return []

        duration = timeframe_durations[timeframe]

        # Generate expected timestamps
        expected_timestamps = []
        current = start_date
        while current <= end_date:
            expected_timestamps.append(int(current.timestamp() * 1000))
            current += timedelta(seconds=duration)

        # Get actual timestamps
        query = """
        SELECT EXTRACT(EPOCH FROM timestamp) * 1000 as ts
        FROM market_data
        WHERE exchange = :exchange 
        AND symbol = :symbol 
        AND timeframe = :timeframe
        AND timestamp BETWEEN :start_date AND :end_date
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text(query),
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                )
                actual_timestamps = [int(row[0]) for row in result.fetchall()]

                # Find missing timestamps
                missing = [
                    ts for ts in expected_timestamps if ts not in actual_timestamps
                ]
                return missing
        except Exception as e:
            logger.error(f"Error retrieving missing timestamps: {e}")
            return []

    def delete_records_by_timestamp(
        self, exchange: str, symbol: str, timeframe: str, timestamp: int
    ) -> int:
        """Delete all records for a specific timestamp."""
        query = """
        DELETE FROM market_data
        WHERE exchange = :exchange 
        AND symbol = :symbol 
        AND timeframe = :timeframe
        AND EXTRACT(EPOCH FROM timestamp) * 1000 = :timestamp
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text(query),
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": timestamp,
                    },
                )
                connection.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"Error deleting records by timestamp: {e}")
            return 0

    def insert_single_record(
        self, exchange: str, symbol: str, timeframe: str, record: Dict[str, Any]
    ) -> bool:
        """Insert a single OHLCV record."""
        query = """
        INSERT INTO market_data 
        (exchange, symbol, timeframe, timestamp, open, high, low, close, volume, 
         backfilled, source_exchange)
        VALUES 
        (:exchange, :symbol, :timeframe, 
         TO_TIMESTAMP(:timestamp / 1000), :open, :high, :low, :close, :volume,
         :backfilled, :source_exchange)
        """
        try:
            with self.engine.connect() as connection:
                params = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": record["timestamp"],
                    "open": record["open"],
                    "high": record["high"],
                    "low": record["low"],
                    "close": record["close"],
                    "volume": record["volume"],
                    "backfilled": record.get("backfilled", False),
                    "source_exchange": record.get("source_exchange", None),
                }
                connection.execute(text(query), params)
                connection.commit()
                return True
        except Exception as e:
            logger.error(f"Error inserting single record: {e}")
            return False

    def insert_data(
        self, exchange: str, symbol: str, timeframe: str, data: List[List[float]]
    ) -> int:
        """Insert OHLCV data into database with conflict handling."""
        if not data:
            return 0

        # Use a transaction to ensure atomicity
        inserted_count = 0
        try:
            with self.engine.begin() as connection:
                for entry in data:
                    # Check if record already exists
                    check_query = """
                    SELECT id FROM market_data
                    WHERE exchange = :exchange 
                    AND symbol = :symbol 
                    AND timeframe = :timeframe
                    AND timestamp = TO_TIMESTAMP(:timestamp / 1000)
                    """
                    result = connection.execute(
                        text(check_query),
                        {
                            "exchange": exchange,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "timestamp": entry[0],
                        },
                    )

                    if result.fetchone():
                        # Record exists, skip
                        continue

                    # Insert new record
                    insert_query = """
                    INSERT INTO market_data 
                    (exchange, symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES 
                    (:exchange, :symbol, :timeframe, 
                     TO_TIMESTAMP(:timestamp / 1000), :open, :high, :low, :close, :volume)
                    """
                    connection.execute(
                        text(insert_query),
                        {
                            "exchange": exchange,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "timestamp": entry[0],
                            "open": entry[1],
                            "high": entry[2],
                            "low": entry[3],
                            "close": entry[4],
                            "volume": entry[5],
                        },
                    )
                    inserted_count += 1

            return inserted_count
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return 0

    def record_data_gap(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_timestamp: datetime,
        end_timestamp: datetime,
    ) -> bool:
        """Record a detected data gap."""
        query = """
        INSERT INTO data_gaps 
        (exchange, symbol, timeframe, start_timestamp, end_timestamp, gap_duration)
        VALUES 
        (:exchange, :symbol, :timeframe, :start_timestamp, :end_timestamp, 
         EXTRACT(EPOCH FROM (:end_timestamp - :start_timestamp)))
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(
                    text(query),
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "start_timestamp": start_timestamp,
                        "end_timestamp": end_timestamp,
                    },
                )
                connection.commit()
                return True
        except Exception as e:
            logger.error(f"Error recording data gap: {e}")
            return False

    def mark_gap_as_filled(self, gap_id: int, filled_from_exchange: str) -> bool:
        """Mark a data gap as filled."""
        query = """
        UPDATE data_gaps
        SET filled = TRUE, filled_at = NOW(), filled_from_exchange = :filled_from_exchange
        WHERE id = :gap_id
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(
                    text(query),
                    {
                        "gap_id": gap_id,
                        "filled_from_exchange": filled_from_exchange,
                    },
                )
                connection.commit()
                return True
        except Exception as e:
            logger.error(f"Error marking gap as filled: {e}")
            return False

    def record_validation_issue(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        issue_type: str,
        severity: str,
        message: str,
    ) -> bool:
        """Record a data validation issue."""
        query = """
        INSERT INTO data_validation_issues 
        (exchange, symbol, timeframe, timestamp, issue_type, severity, message)
        VALUES 
        (:exchange, :symbol, :timeframe, :timestamp, :issue_type, :severity, :message)
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(
                    text(query),
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": timestamp,
                        "issue_type": issue_type,
                        "severity": severity,
                        "message": message,
                    },
                )
                connection.commit()
                return True
        except Exception as e:
            logger.error(f"Error recording validation issue: {e}")
            return False

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
            logger.error(f"Error retrieving user roles: {e}")
            return []


class Database:
    """Database operations using SQLAlchemy ORM."""

    def __init__(self):
        """Initialize database connection."""
        self.db = SessionLocal()

    def insert_data(
        self, exchange: str, symbol: str, timeframe: str, data: list
    ) -> int:
        """Insert OHLCV data into database."""
        inserted_count = 0
        try:
            for entry in data:
                # Check if record already exists
                existing = (
                    self.db.query(OHLCVData)
                    .filter(
                        OHLCVData.exchange == exchange,
                        OHLCVData.symbol == symbol,
                        OHLCVData.timeframe == timeframe,
                        OHLCVData.timestamp
                        == datetime.fromtimestamp(entry[0] / 1000, UTC),
                    )
                    .first()
                )

                if existing:
                    # Skip existing records
                    continue

                ohlcv_data = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": datetime.fromtimestamp(entry[0] / 1000, UTC),
                    "open": entry[1],
                    "high": entry[2],
                    "low": entry[3],
                    "close": entry[4],
                    "volume": entry[5],
                }
                ohlcv_entry = OHLCVData(**ohlcv_data)
                self.db.add(ohlcv_entry)
                inserted_count += 1

            self.db.commit()
            return inserted_count
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error inserting data: {e}")
            raise e

    def close(self) -> None:
        """Close database connection."""
        self.db.close()
