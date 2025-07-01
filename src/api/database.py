from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
import pandas as pd

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
Base = declarative_base()


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


class DBConnector:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
        self.engine = create_engine(self.connection_string)

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_ohlcv_data(self, exchange, symbol, timeframe):
        query = """
        SELECT timestamp, open_price, high_price, low_price, close_price, volume
        FROM ohlcv_data
        WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text(query),
                    {"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
                )
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            raise e

    def get_user_roles(self, username):
        query = """
        SELECT role
        FROM user_roles
        WHERE username = :username
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), {"username": username})
                roles = [row["role"] for row in result.fetchall()]
                return roles
        except Exception as e:
            raise e

    def disconnect(self):
        pass


class Database:
    def __init__(self):
        self.db = SessionLocal()

    def insert_data(self, exchange, symbol, timeframe, data):
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

    def close(self):
        self.db.close()
