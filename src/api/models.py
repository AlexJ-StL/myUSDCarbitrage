"""Data models for the USDC arbitrage application."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .database import Base


class USDCData(Base):
    """SQLAlchemy model for USDC price data."""

    __tablename__ = "usdc_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    exchange = Column(String, index=True)
    price = Column(Float)


class USDCDataPydantic(BaseModel):
    """Pydantic model for USDC data API responses."""

    id: int
    timestamp: datetime
    exchange: str
    price: float

    model_config = ConfigDict(from_attributes=True)


class Strategy(Base):
    """SQLAlchemy model for trading strategies."""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    parameters = Column(JSON)
    code = Column(Text)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_active = Column(Boolean, default=True)
    strategy_type = Column(
        String, default="custom"
    )  # custom, arbitrage, trend_following, etc.

    # Relationships
    backtests = relationship("BacktestResult", back_populates="strategy")
    tags = relationship(
        "StrategyTag",
        secondary="strategy_tag_associations",
        back_populates="strategies",
    )
    versions = relationship("StrategyVersion", back_populates="strategy")

    def get_strategy_function(self):
        """Get the strategy function from the code."""
        try:
            # Create a local namespace
            namespace = {}

            # Execute the strategy code in the namespace
            exec(self.code, globals(), namespace)

            # Return the strategy function
            return namespace.get("strategy")
        except Exception as e:
            raise ValueError(f"Failed to load strategy function: {e}")


class StrategyPydantic(BaseModel):
    """Pydantic model for strategy API responses."""

    id: int
    name: str
    description: str
    parameters: Dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime
    is_active: bool
    code: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class StrategyVersion(Base):
    """SQLAlchemy model for strategy versions."""

    __tablename__ = "strategy_versions"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    version = Column(Integer)
    code = Column(Text)
    parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)
    created_by = Column(String)
    commit_message = Column(String)

    # Relationships
    strategy = relationship("Strategy", back_populates="versions")


class BacktestResult(Base):
    """SQLAlchemy model for backtest results."""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    parameters = Column(JSON)
    results = Column(JSON)
    metrics = Column(JSON)
    status = Column(String, default="pending")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    strategy = relationship("Strategy", back_populates="backtests")


class BacktestResultPydantic(BaseModel):
    """Pydantic model for backtest result API responses."""

    id: int
    strategy_id: int
    start_date: datetime
    end_date: datetime
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class Position(BaseModel):
    """Pydantic model for position data."""

    exchange: str
    symbol: str
    amount: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    fees_paid: float


class Order(BaseModel):
    """Pydantic model for order data."""

    exchange: str
    symbol: str
    order_type: str
    side: str
    amount: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime
    executed_price: Optional[float] = None
    executed_amount: float = 0.0
    fee: float = 0.0
    status: str = "open"
    execution_time: Optional[datetime] = None
    slippage: float = 0.0


class Transaction(BaseModel):
    """Pydantic model for transaction data."""

    exchange: str
    symbol: str
    side: str
    amount: float
    price: float
    fee: float
    timestamp: datetime
    value: float


class PortfolioSnapshot(BaseModel):
    """Pydantic model for portfolio snapshot."""

    timestamp: datetime
    cash: float
    positions: Dict[str, Position]
    equity: float


class BacktestMetrics(BaseModel):
    """Pydantic model for backtest metrics."""

    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    cagr: float
    win_rate: float
    final_equity: float
    initial_equity: float
    trade_count: int
    annual_volatility: float


class BacktestRequest(BaseModel):
    """Pydantic model for backtest request."""

    strategy_id: int = Field(..., description="Strategy ID to backtest")
    start_date: datetime = Field(..., description="Start date for backtest")
    end_date: datetime = Field(..., description="End date for backtest")
    exchanges: List[str] = Field(
        default=["coinbase", "kraken", "binance"],
        description="Exchanges to include in backtest",
    )
    symbols: List[str] = Field(
        default=["USDC/USD"],
        description="Symbols to include in backtest",
    )
    timeframe: str = Field(
        default="1h",
        description="Timeframe for backtest (e.g., 1m, 5m, 1h, 1d)",
    )
    initial_balance: float = Field(
        default=10000.0,
        description="Initial portfolio balance",
        gt=0,
    )
    position_sizing: str = Field(
        default="percent",
        description="Position sizing strategy (fixed, percent, kelly, volatility, risk_parity)",
    )
    position_size: float = Field(
        default=0.02,
        description="Position size (interpretation depends on position_sizing)",
        gt=0,
    )
    rebalance_frequency: str = Field(
        default="monthly",
        description="Portfolio rebalancing frequency (daily, weekly, monthly, quarterly, threshold)",
    )
    rebalance_threshold: float = Field(
        default=0.05,
        description="Threshold for threshold-based rebalancing",
        gt=0,
    )
    include_fees: bool = Field(
        default=True,
        description="Include exchange fees in backtest",
    )
    include_slippage: bool = Field(
        default=True,
        description="Include slippage in backtest",
    )
    strategy_params: Dict[str, Any] = Field(
        default={},
        description="Additional strategy parameters",
    )


class BacktestResponse(BaseModel):
    """Pydantic model for backtest response."""

    backtest_id: int
    strategy_id: int
    start_date: datetime
    end_date: datetime
    status: str
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class StrategyTag(Base):
    """SQLAlchemy model for strategy tags."""

    __tablename__ = "strategy_tags"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    strategies = relationship(
        "Strategy", secondary="strategy_tag_associations", back_populates="tags"
    )


class StrategyTagAssociation(Base):
    """SQLAlchemy model for strategy-tag associations."""

    __tablename__ = "strategy_tag_associations"

    strategy_id = Column(Integer, ForeignKey("strategies.id"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("strategy_tags.id"), primary_key=True)
    created_at = Column(DateTime, default=datetime.now)


class StrategyComparison(Base):
    """SQLAlchemy model for strategy comparisons."""

    __tablename__ = "strategy_comparisons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    strategy_ids = Column(JSON)  # List of strategy IDs
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    parameters = Column(JSON)
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)


class ABTest(Base):
    """SQLAlchemy model for A/B tests between strategies."""

    __tablename__ = "ab_tests"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    strategy_a_id = Column(Integer, ForeignKey("strategies.id"))
    strategy_b_id = Column(Integer, ForeignKey("strategies.id"))
    allocation_ratio = Column(Float, default=0.5)  # Allocation for strategy A
    start_date = Column(DateTime)
    end_date = Column(DateTime, nullable=True)
    exchanges = Column(JSON)  # List of exchanges
    symbols = Column(JSON)  # List of symbols
    status = Column(String, default="created")  # created, running, completed, failed
    results = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)

    # Relationships
    strategy_a = relationship("Strategy", foreign_keys=[strategy_a_id])
    strategy_b = relationship("Strategy", foreign_keys=[strategy_b_id])


# Authentication and User Management Models
class User(Base):
    """SQLAlchemy model for users."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    roles = relationship("UserRole", back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user")


class Role(Base):
    """SQLAlchemy model for roles."""

    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    permissions = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    users = relationship("UserRole", back_populates="role")


class UserRole(Base):
    """SQLAlchemy model for user-role associations."""

    __tablename__ = "user_roles"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id"), primary_key=True)
    assigned_at = Column(DateTime, default=datetime.now)

    # Relationships
    user = relationship("User", back_populates="roles")
    role = relationship("Role", back_populates="users")


class RefreshToken(Base):
    """SQLAlchemy model for refresh tokens."""

    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    revoked_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="refresh_tokens")


class TokenBlacklist(Base):
    """SQLAlchemy model for blacklisted tokens."""

    __tablename__ = "token_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(255), unique=True, index=True, nullable=False)  # JWT ID
    token_type = Column(String(20), nullable=False)  # 'access' or 'refresh'
    expires_at = Column(DateTime, nullable=False)
    blacklisted_at = Column(DateTime, default=datetime.now)
    reason = Column(String(100), nullable=True)  # logout, revoked, etc.


# Pydantic models for authentication
class UserCreate(BaseModel):
    """Pydantic model for user creation."""

    username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_]+$")
    email: str = Field(..., pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=8, max_length=100)


class UserLogin(BaseModel):
    """Pydantic model for user login."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=1, max_length=100)


class UserResponse(BaseModel):
    """Pydantic model for user response."""

    id: int
    username: str
    email: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    """Pydantic model for token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefresh(BaseModel):
    """Pydantic model for token refresh."""

    refresh_token: str


class RoleCreate(BaseModel):
    """Pydantic model for role creation."""

    name: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)


class RoleResponse(BaseModel):
    """Pydantic model for role response."""

    id: int
    name: str
    description: Optional[str] = None
    permissions: List[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Import audit logging and API key models to ensure they're included in the database schema
from .audit_logging import AuditLog
from .api_keys import APIKey
