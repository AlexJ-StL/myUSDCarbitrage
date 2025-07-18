"""Main FastAPI application for USDC arbitrage backtesting system."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .audit_logging import AuditLoggingMiddleware
from .database import Base, engine, get_db
from .encryption import EncryptionMiddleware
from .rate_limiting import EnhancedRateLimiter, RateLimitMiddleware
from .routers import admin, api_keys, auth, backtest, data, results, strategies


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Starting USDC Arbitrage Backtesting API...")

    # Create database tables if they don't exist
    # Note: In production, use proper database migrations
    Base.metadata.create_all(bind=engine)

    yield

    # Shutdown
    print("Shutting down USDC Arbitrage Backtesting API...")


app = FastAPI(
    title="USDC Arbitrage Backtesting API",
    description="A comprehensive backtesting system for USDC arbitrage strategies with advanced security features",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize security components
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
rate_limiter = EnhancedRateLimiter(redis_url=redis_url)

# Add security middlewares
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
app.add_middleware(EncryptionMiddleware)
app.add_middleware(AuditLoggingMiddleware, db_session_factory=get_db)

# Include routers
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(api_keys.router)
app.include_router(data.router)
app.include_router(strategies.router)
app.include_router(backtest.router)
app.include_router(results.router)


@app.get("/")
def read_root():
    """Root endpoint returning API welcome message."""
    return {"message": "Welcome to the myUSDCarbitrage API"}


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": "2025-01-17T00:00:00Z",
    }


@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    """Custom handler for rate limit exceptions."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "type": "rate_limit_error",
        },
        headers={
            "Retry-After": "60",
            "X-RateLimit-Reset": str(int(exc.detail.get("retry_after", 60))),
        },
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc):
    """Custom handler for internal server errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "type": "internal_error",
        },
    )
