"""Strategy management module for USDC arbitrage application."""

from .comparison import StrategyComparison
from .manager import StrategyManager
from .routes import router
from .types import (
    STRATEGY_TYPES,
    ArbitrageStrategy,
    BaseStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
    VolatilityBreakoutStrategy,
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
    "VolatilityBreakoutStrategy",
    "STRATEGY_TYPES",
    "create_strategy",
    "get_strategy_template",
    "router",
]
