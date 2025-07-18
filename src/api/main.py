"""Main FastAPI application for USDC arbitrage backtesting system."""

from fastapi import FastAPI

from .database import Base, engine
from .routers import auth, backtest, data, results, strategies

# Note: Database tables should be created using the init_auth.py script
# Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="USDC Arbitrage Backtesting API",
    description="A comprehensive backtesting system for USDC arbitrage strategies",
    version="0.1.0",
)

app.include_router(auth.router)
app.include_router(data.router)
app.include_router(strategies.router)
app.include_router(backtest.router)
app.include_router(results.router)


@app.get("/")
def read_root():
    """Root endpoint returning API welcome message."""
    return {"message": "Welcome to the myUSDCarbitrage API"}
