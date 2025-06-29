from fastapi import APIRouter, Depends
from ..dependencies import get_database, get_current_user

router = APIRouter()

def generate_equity_curve(results):
    """Generate Plotly JSON for equity curve"""
    # Implementation would go here
    return {}

@router.get("/{backtest_id}", summary="Get backtest results")
async def get_results(
    backtest_id: int,
    db=Depends(get_database),
    user=Depends(get_current_user)
):
    try:
        results = db.get_backtest_results(backtest_id, user)

        # Generate visualizations (example)
        visualization = generate_equity_curve(results)

        return {
            "results": results,
            "visualization": visualization
        }
    except Exception as e:
        return {"error": str(e)}
