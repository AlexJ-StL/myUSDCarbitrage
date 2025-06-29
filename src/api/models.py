from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import List, Optional
import re

class OHLCRequest(BaseModel):
    exchange: str = Field(..., regex=r"^[a-zA-Z0-9_]{1,20}$")
    symbol: str = Field(..., regex=r"^[A-Z]{3,10}\/[A-Z]{3,10}$")
    timeframe: str = Field(..., regex=r"^(1m|5m|15m|1h|4h|1d)$")
    start_date: datetime
    end_date: datetime

    @validator('end_date')
    def validate_dates(cls, end_date, values):
        start_date = values.get('start_date')
        if start_date and end_date <= start_date:
            raise ValueError("End date must be after start date")
        return end_date

class BacktestRequest(BaseModel):
    strategy_name: str
    parameters: dict = Field(
        ...,
        example={
            "buy_threshold": 0.995,
            "sell_threshold": 1.005,
            "initial_capital": 10000
        }
    )
    data_settings: OHLCRequest
