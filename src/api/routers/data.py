from fastapi import APIRouter, Depends
from ..dependencies import get_database, get_current_user
from ..models import OHLCRequest
import pandas as pd

router = APIRouter()

@router.post("/ohlc", summary="Get historical OHLC data")
async def get_ohlc(
    request: OHLCRequest,
    db=Depends(get_database),
    user=Depends(get_current_user)
):
    try:
        df = db.get_ohlcv_data(
            request.exchange,
            request.symbol,
            request.timeframe,
            request.start_date,
            request.end_date
        )
        return {
            "exchange": request.exchange,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "data": df.to_dict(orient="records"),
            "count": len(df)
        }
    except Exception as e:
        return {"error": str(e)}
