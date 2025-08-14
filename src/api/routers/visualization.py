"""Visualization router for USDC arbitrage API."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..security import get_current_active_user, require_permissions
from ..visualization import (
    PerformanceVisualization,
    PortfolioAnalytics,
    RiskAnalysis,
    StrategyComparison,
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualization", tags=["visualization"])


@router.get("/performance/{result_id}")
async def get_performance_visualization(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get performance visualization for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Create visualization
        visualizer = PerformanceVisualization()
        dashboard = visualizer.create_performance_dashboard(result.__dict__)

        return dashboard

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error creating performance visualization for result {result_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to create visualization: {str(e)}"
        )


@router.get("/portfolio/{result_id}")
async def get_portfolio_visualization(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get portfolio analytics visualization for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Create visualization
        visualizer = PortfolioAnalytics()
        dashboard = visualizer.create_portfolio_dashboard(result.__dict__)

        return dashboard

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error creating portfolio visualization for result {result_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to create visualization: {str(e)}"
        )


@router.get("/risk/{result_id}")
async def get_risk_visualization(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get risk analysis visualization for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Create visualization
        visualizer = RiskAnalysis()
        dashboard = visualizer.create_risk_dashboard(result.__dict__)

        return dashboard

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating risk visualization for result {result_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create visualization: {str(e)}"
        )


@router.get("/comparison")
async def get_comparison_visualization(
    result_ids: List[int] = Query(
        ..., description="List of backtest result IDs to compare"
    ),
    benchmark_id: Optional[int] = Query(
        None, description="Optional benchmark result ID"
    ),
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get strategy comparison visualization for multiple backtest results."""
    try:
        # Get backtest results
        strategy_results = []
        for result_id in result_ids:
            result = (
                db.query(models.BacktestResult)
                .filter(models.BacktestResult.id == result_id)
                .first()
            )

            if not result:
                raise HTTPException(
                    status_code=404, detail=f"Backtest result {result_id} not found"
                )

            # Get strategy name
            strategy = (
                db.query(models.Strategy)
                .filter(models.Strategy.id == result.strategy_id)
                .first()
            )

            strategy_name = (
                strategy.name if strategy else f"Strategy {result.strategy_id}"
            )

            # Add to results list
            strategy_results.append({
                "name": strategy_name,
                "id": result.id,
                "strategy_id": result.strategy_id,
                "equity_curve": result.results.get("equity_curve", [])
                if result.results
                else [],
                "metrics": result.metrics or {},
            })

        # Get benchmark if provided
        benchmark_results = None
        if benchmark_id:
            benchmark = (
                db.query(models.BacktestResult)
                .filter(models.BacktestResult.id == benchmark_id)
                .first()
            )

            if not benchmark:
                raise HTTPException(
                    status_code=404, detail=f"Benchmark result {benchmark_id} not found"
                )

            # Get strategy name
            strategy = (
                db.query(models.Strategy)
                .filter(models.Strategy.id == benchmark.strategy_id)
                .first()
            )

            strategy_name = (
                strategy.name if strategy else f"Benchmark {benchmark.strategy_id}"
            )

            benchmark_results = [
                {
                    "name": f"Benchmark: {strategy_name}",
                    "id": benchmark.id,
                    "strategy_id": benchmark.strategy_id,
                    "data": benchmark.results.get("equity_curve", [])
                    if benchmark.results
                    else [],
                    "metrics": benchmark.metrics or {},
                }
            ]

        # Create visualization
        visualizer = StrategyComparison()
        dashboard = visualizer.create_comparison_dashboard(
            strategy_results, benchmark_results
        )

        return dashboard

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating comparison visualization: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create visualization: {str(e)}"
        )


@router.get("/chart/equity_curve/{result_id}")
async def get_equity_curve_chart(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get equity curve chart for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get equity curve data
        equity_curve_data = (
            result.results.get("equity_curve", []) if result.results else []
        )

        if not equity_curve_data:
            raise HTTPException(status_code=404, detail="No equity curve data found")

        # Create visualization
        visualizer = PerformanceVisualization()
        chart = visualizer.create_equity_curve_chart(equity_curve_data)

        return chart

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating equity curve chart for result {result_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create chart: {str(e)}")


@router.get("/chart/drawdown/{result_id}")
async def get_drawdown_chart(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get drawdown chart for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get equity curve data
        equity_curve_data = (
            result.results.get("equity_curve", []) if result.results else []
        )

        if not equity_curve_data:
            raise HTTPException(status_code=404, detail="No equity curve data found")

        # Create visualization
        visualizer = RiskAnalysis()
        chart = visualizer.create_drawdown_analysis_chart(equity_curve_data)

        return chart

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating drawdown chart for result {result_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create chart: {str(e)}")


@router.get("/chart/monthly_returns/{result_id}")
async def get_monthly_returns_chart(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get monthly returns heatmap for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get equity curve data
        equity_curve_data = (
            result.results.get("equity_curve", []) if result.results else []
        )

        if not equity_curve_data:
            raise HTTPException(status_code=404, detail="No equity curve data found")

        # Create visualization
        visualizer = PerformanceVisualization()
        chart = visualizer.create_monthly_returns_heatmap(equity_curve_data)

        return chart

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error creating monthly returns chart for result {result_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Failed to create chart: {str(e)}")


@router.get("/chart/trade_analysis/{result_id}")
async def get_trade_analysis_chart(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get trade analysis chart for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get trades data
        trades_data = result.results.get("transactions", []) if result.results else []

        if not trades_data:
            raise HTTPException(status_code=404, detail="No trade data found")

        # Create visualization
        visualizer = PerformanceVisualization()
        chart = visualizer.create_trade_analysis_chart(trades_data)

        return chart

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating trade analysis chart for result {result_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create chart: {str(e)}")


@router.get("/chart/portfolio_composition/{result_id}")
async def get_portfolio_composition_chart(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get portfolio composition chart for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get positions data
        positions_data = result.results.get("positions", []) if result.results else []

        if not positions_data:
            raise HTTPException(status_code=404, detail="No position data found")

        # Create visualization
        visualizer = PortfolioAnalytics()
        chart = visualizer.create_portfolio_composition_chart(positions_data)

        return chart

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error creating portfolio composition chart for result {result_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Failed to create chart: {str(e)}")


@router.get("/chart/var_cvar/{result_id}")
async def get_var_cvar_chart(
    result_id: int,
    confidence_levels: List[float] = Query(
        default=[0.95, 0.99], description="Confidence levels for VaR/CVaR"
    ),
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get VaR and CVaR chart for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get equity curve data
        equity_curve_data = (
            result.results.get("equity_curve", []) if result.results else []
        )

        if not equity_curve_data:
            raise HTTPException(status_code=404, detail="No equity curve data found")

        # Calculate returns
        df = pd.DataFrame(equity_curve_data)
        df["returns"] = df["equity"].pct_change().fillna(0)
        returns = df["returns"].tolist()

        # Create visualization
        visualizer = RiskAnalysis()
        chart = visualizer.create_var_cvar_chart(returns, confidence_levels)

        return chart

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating VaR/CVaR chart for result {result_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create chart: {str(e)}")


@router.get("/chart/stress_test/{result_id}")
async def get_stress_test_chart(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get stress testing chart for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get equity curve data
        equity_curve_data = (
            result.results.get("equity_curve", []) if result.results else []
        )

        if not equity_curve_data:
            raise HTTPException(status_code=404, detail="No equity curve data found")

        # Create visualization
        visualizer = RiskAnalysis()
        stress_results = visualizer.perform_stress_testing(equity_curve_data)

        if "error" in stress_results:
            raise HTTPException(status_code=400, detail=stress_results["error"])

        chart = visualizer.create_stress_test_chart(stress_results)

        return chart

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating stress test chart for result {result_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create chart: {str(e)}")


@router.get("/chart/regime_analysis/{result_id}")
async def get_regime_analysis_chart(
    result_id: int,
    n_regimes: int = Query(default=3, description="Number of market regimes to detect"),
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get market regime analysis chart for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get equity curve data
        equity_curve_data = (
            result.results.get("equity_curve", []) if result.results else []
        )

        if not equity_curve_data:
            raise HTTPException(status_code=404, detail="No equity curve data found")

        # Create visualization
        visualizer = RiskAnalysis()
        regime_results = visualizer.detect_market_regimes(equity_curve_data, n_regimes)

        if "error" in regime_results:
            raise HTTPException(status_code=400, detail=regime_results["error"])

        chart = visualizer.create_regime_analysis_chart(regime_results)

        return chart

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error creating regime analysis chart for result {result_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Failed to create chart: {str(e)}")


@router.get("/chart/risk_attribution/{result_id}")
async def get_risk_attribution_chart(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get portfolio risk attribution chart for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get positions data
        positions_data = result.results.get("positions", []) if result.results else []

        if not positions_data:
            raise HTTPException(status_code=404, detail="No position data found")

        # Extract returns data for risk attribution
        returns_data = {}
        for pos in positions_data:
            if "symbol" in pos and "returns" in pos:
                returns_data[pos["symbol"]] = pos["returns"]

        if not returns_data:
            raise HTTPException(
                status_code=404, detail="No returns data found for risk attribution"
            )

        # Create visualization
        visualizer = RiskAnalysis()
        attribution_results = visualizer.calculate_portfolio_risk_attribution(
            positions_data, returns_data
        )

        if "error" in attribution_results:
            raise HTTPException(status_code=400, detail=attribution_results["error"])

        chart = visualizer.create_risk_attribution_chart(attribution_results)

        return chart

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error creating risk attribution chart for result {result_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Failed to create chart: {str(e)}")


@router.get("/analytics/var_cvar/{result_id}")
async def get_var_cvar_analytics(
    result_id: int,
    confidence_levels: List[float] = Query(
        default=[0.95, 0.99], description="Confidence levels for VaR/CVaR"
    ),
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get VaR and CVaR analytics for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get equity curve data
        equity_curve_data = (
            result.results.get("equity_curve", []) if result.results else []
        )

        if not equity_curve_data:
            raise HTTPException(status_code=404, detail="No equity curve data found")

        # Calculate returns
        df = pd.DataFrame(equity_curve_data)
        df["returns"] = df["equity"].pct_change().fillna(0)
        returns = df["returns"].tolist()

        # Calculate VaR/CVaR
        visualizer = RiskAnalysis()
        analytics = visualizer.calculate_var_cvar(returns, confidence_levels)

        return analytics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error calculating VaR/CVaR analytics for result {result_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate analytics: {str(e)}"
        )


@router.get("/analytics/stress_test/{result_id}")
async def get_stress_test_analytics(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get stress testing analytics for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get equity curve data
        equity_curve_data = (
            result.results.get("equity_curve", []) if result.results else []
        )

        if not equity_curve_data:
            raise HTTPException(status_code=404, detail="No equity curve data found")

        # Perform stress testing
        visualizer = RiskAnalysis()
        analytics = visualizer.perform_stress_testing(equity_curve_data)

        return analytics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error performing stress test analytics for result {result_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to perform analytics: {str(e)}"
        )


@router.get("/analytics/regime_analysis/{result_id}")
async def get_regime_analysis_analytics(
    result_id: int,
    n_regimes: int = Query(default=3, description="Number of market regimes to detect"),
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get market regime analysis analytics for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get equity curve data
        equity_curve_data = (
            result.results.get("equity_curve", []) if result.results else []
        )

        if not equity_curve_data:
            raise HTTPException(status_code=404, detail="No equity curve data found")

        # Perform regime analysis
        visualizer = RiskAnalysis()
        analytics = visualizer.detect_market_regimes(equity_curve_data, n_regimes)

        return analytics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing regime analysis for result {result_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform analytics: {str(e)}"
        )


@router.get("/analytics/risk_attribution/{result_id}")
async def get_risk_attribution_analytics(
    result_id: int,
    current_user: models.User = Depends(
        require_permissions(["read:backtest", "read:own_backtest"])
    ),
    db: Session = Depends(get_db),
):
    """Get portfolio risk attribution analytics for a backtest result."""
    try:
        # Get backtest result
        result = (
            db.query(models.BacktestResult)
            .filter(models.BacktestResult.id == result_id)
            .first()
        )

        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")

        # Get positions data
        positions_data = result.results.get("positions", []) if result.results else []

        if not positions_data:
            raise HTTPException(status_code=404, detail="No position data found")

        # Extract returns data for risk attribution
        returns_data = {}
        for pos in positions_data:
            if "symbol" in pos and "returns" in pos:
                returns_data[pos["symbol"]] = pos["returns"]

        if not returns_data:
            raise HTTPException(
                status_code=404, detail="No returns data found for risk attribution"
            )

        # Calculate risk attribution
        visualizer = RiskAnalysis()
        analytics = visualizer.calculate_portfolio_risk_attribution(
            positions_data, returns_data
        )

        return analytics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating risk attribution for result {result_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate analytics: {str(e)}"
        )
