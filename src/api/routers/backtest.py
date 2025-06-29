from fastapi import APIRouter, Depends
from ..dependencies import get_database, get_current_user
from ..models import BacktestRequest
from .strategies import get_strategy_function

router = APIRouter()

@router.post("/run", summary="Execute a new backtest")
async def run_backtest(
    request: BacktestRequest,
    db=Depends(get_database),
    user=Depends(get_current_user)
):
    try:
        # Get strategy function
        strategy_fn = get_strategy_function(request.strategy_name)

        # Fetch data
        df = db.get_ohlcv_data(
            request.data_settings.exchange,
            request.data_settings.symbol,
            request.data_settings.timeframe,
            request.data_settings.start_date,
            request.data_settings.end_date
        )

        # Execute backtest
        results = strategy_fn(df, **request.parameters)

        # Save results to DB
        backtest_id = db.save_backtest_results(
            user,
            request.strategy_name,
            request.parameters,
            results
        )

        return {
            "status": "completed",
            "backtest_id": backtest_id,
            "results": results
        }
    except Exception as e:
        return {"error": str(e)}
