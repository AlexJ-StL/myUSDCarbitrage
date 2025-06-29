from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from .routers import data, backtest, strategies, results
from .dependencies import get_database

app = FastAPI(
    title="USDC Arbitrage API",
    description="Backtesting and strategy execution API",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/data", tags=["Market Data"])
app.include_router(backtest.router, prefix="/backtest", tags=["Backtesting"])
app.include_router(strategies.router, prefix="/strategies", tags=["Strategies"])
app.include_router(results.router, prefix="/results", tags=["Results"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "USDC Arbitrage API running"}
