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
from .routers import (
    admin,
    api_keys,
    auth,
    backtest,
    centralized_logging,
    data,
    data_export,
    health,
    logging as logging_router,
    monitoring,
    results,
    strategies,
    versioned_example,
    websocket,
)
from ..monitoring.scheduler import start_monitoring, stop_monitoring
from ..monitoring.centralized_logging import (
    initialize_logging,
    create_exception_handler,
)
from ..monitoring.db_query_logging import setup_query_logging
from ..monitoring.log_rotation import configure_log_rotation


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Starting USDC Arbitrage Backtesting API...")

    # Setup centralized logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    log_file = os.getenv("LOG_FILE", "logs/app.log")

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Initialize centralized logging system
    initialize_logging(
        log_level=log_level,
        redis_url=redis_url,
        log_file=log_file,
    )
    print(f"Centralized logging initialized with level {log_level}")

    # Configure log rotation
    configure_log_rotation(
        log_dir=os.path.dirname(log_file),
        max_size_mb=100,
        backup_count=10,
        compress=True,
        archive_dir=os.path.join(os.path.dirname(log_file), "archive"),
    )
    print("Log rotation configured")

    # Create database tables if they don't exist
    # Note: In production, use proper database migrations
    Base.metadata.create_all(bind=engine)

    # Setup database query logging
    setup_query_logging(engine)
    print("Database query logging configured")

    # Start monitoring scheduler
    await start_monitoring()
    print("Monitoring scheduler started")

    yield

    # Shutdown
    print("Shutting down USDC Arbitrage Backtesting API...")

    # Stop monitoring scheduler
    await stop_monitoring()
    print("Monitoring scheduler stopped")


app = FastAPI(
    title="USDC Arbitrage Backtesting API",
    description="A comprehensive backtesting system for USDC arbitrage strategies with advanced security features",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    openapi_tags=[
        {
            "name": "auth",
            "description": "Authentication and authorization operations",
        },
        {
            "name": "strategies",
            "description": "Operations with trading strategies",
        },
        {
            "name": "backtest",
            "description": "Backtest execution and management",
        },
        {
            "name": "results",
            "description": "Backtest results and analysis",
        },
        {
            "name": "data",
            "description": "Market data operations",
        },
        {
            "name": "data_export",
            "description": "Data export in multiple formats",
        },
        {
            "name": "monitoring",
            "description": "System monitoring and health checks",
        },
        {
            "name": "admin",
            "description": "Administrative operations",
        },
    ],
    contact={
        "name": "USDC Arbitrage Team",
        "url": "https://github.com/myUSDCarbitrage",
        "email": "support@myusdcarbitrage.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
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
# Include regular routers
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(api_keys.router)
app.include_router(health.router)
app.include_router(monitoring.router)
app.include_router(logging_router.router)
app.include_router(centralized_logging.router)
app.include_router(data.router)
app.include_router(data_export.router)
app.include_router(strategies.router)
app.include_router(backtest.router)
app.include_router(results.router)
app.include_router(websocket.router)

# Include versioned routers
versioned_example.versioned_router.include_in_app(app)


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
    # Use the centralized error tracking system
    from ..monitoring.centralized_logging import track_error

    error_id = track_error(
        error=exc,
        context=f"{request.method} {request.url.path}",
        metadata={
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_host": request.client.host if request.client else None,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "error_id": error_id,
            "type": "internal_error",
        },
    )
