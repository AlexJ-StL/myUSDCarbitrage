"""Strategy management module for USDC arbitrage application."""

from .comparison import StrategyComparison
from .manager import StrategyManager, get_strategy_by_id
from .routes import router
from .types import (
    STRATEGY_TYPES,
    ArbitrageStrategy,
    BaseStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
    create_strategy,
    get_strategy_template,
)

__all__ = [
    "StrategyManager",
    "StrategyComparison",
    "BaseStrategy",
    "ArbitrageStrategy",
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "STRATEGY_TYPES",
    "create_strategy",
    "get_strategy_template",
    "get_strategy_by_id",
    "router",
]
