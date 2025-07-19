from fastapi import FastAPI

from .routers import reporting

# Other routers would be imported here as they are built
# from .routers import strategies, backtests, auth

app = FastAPI(
    title="USDC Arbitrage Backtesting API",
    description="An API for backtesting and analyzing USDC arbitrage strategies.",
    version="1.0.0",
)

# Include Routers
app.include_router(reporting.router, prefix="/api/v1")
# app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
# app.include_router(strategies.router, prefix="/api/v1", tags=["Strategies"])


@app.get("/")
async def root():
    return {"message": "Welcome to the USDC Arbitrage API"}
