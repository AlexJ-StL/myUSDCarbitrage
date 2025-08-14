"""Backtesting package for USDC arbitrage strategies."""

from .engine import (
    BacktestEngine,
    ExchangeFeeModel,
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    PortfolioRebalancer,
    Position,
    PositionSizer,
    PositionSizing,
    RebalanceFrequency,
    SlippageModel,
)
from .metrics import PerformanceMetrics
from .optimization import OptimizationResult, WalkForwardOptimizer

__all__ = [
    "BacktestEngine",
    "ExchangeFeeModel",
    "OptimizationResult",
    "Order",
    "OrderSide",
    "OrderType",
    "PerformanceMetrics",
    "Portfolio",
    "PortfolioRebalancer",
    "Position",
    "PositionSizer",
    "PositionSizing",
    "RebalanceFrequency",
    "SlippageModel",
    "WalkForwardOptimizer",
]
