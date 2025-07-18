"""Rate limiting middleware with Redis-based counters."""

import json
import time
from typing import Dict, Optional

import redis
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


class RateLimiter:
    """Redis-based rate limiter with sliding window algorithm."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_rate_limit: int = 100,
        default_window: int = 3600,  # 1 hour in seconds
    ):
        """
        Initialize rate limiter.

        Args:
            redis_url: Redis connection URL
            default_rate_limit: Default requests per window
            default_window: Time window in seconds
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.default_rate_limit = default_rate_limit
        self.default_window = default_window

        # Rate limit configurations for different endpoints/users
        self.rate_limits: Dict[str, Dict[str, int]] = {
            "default": {"limit": default_rate_limit, "window": default_window},
            "auth": {"limit": 10, "window": 900},  # 10 requests per 15 minutes
            "admin": {"limit": 1000, "window": 3600},  # Higher limit for admin
            "api_key": {"limit": 5000, "window": 3600},  # Higher limit for API keys
        }

    def get_client_identifier(self, request: Request) -> str:
        """Get unique identifier for the client."""
        # Try to get user ID from JWT token
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # Try to get API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key[:8]}"  # Use first 8 chars for privacy

        # Fall back to IP address
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        return f"ip:{client_ip}"

    def get_rate_limit_config(self, request: Request) -> Dict[str, int]:
        """Get rate limit configuration based on request context."""
        # Check if it's an admin endpoint
        if request.url.path.startswith("/admin"):
            return self.rate_limits["admin"]

        # Check if it's an auth endpoint
        if request.url.path.startswith("/auth"):
            return self.rate_limits["auth"]

        # Check if using API key
        if request.headers.get("X-API-Key"):
            return self.rate_limits["api_key"]

        return self.rate_limits["default"]

    def is_rate_limited(self, request: Request) -> tuple[bool, Dict[str, any]]:
        """
        Check if request should be rate limited.

        Returns:
            Tuple of (is_limited, rate_limit_info)
        """
        client_id = self.get_client_identifier(request)
        config = self.get_rate_limit_config(request)

        current_time = int(time.time())
        window_start = current_time - config["window"]

        # Redis key for this client's requests
        key = f"rate_limit:{client_id}"

        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()

            # Remove old entries outside the window
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current requests in window
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(current_time): current_time})

            # Set expiration for cleanup
            pipe.expire(key, config["window"])

            results = pipe.execute()
            current_count = results[1] + 1  # +1 for the request we just added

            rate_limit_info = {
                "limit": config["limit"],
                "remaining": max(0, config["limit"] - current_count),
                "reset_time": current_time + config["window"],
                "window": config["window"],
            }

            is_limited = current_count > config["limit"]

            return is_limited, rate_limit_info

        except redis.RedisError as e:
            # If Redis is down, allow the request but log the error
            print(f"Redis error in rate limiter: {e}")
            return False, {
                "limit": config["limit"],
                "remaining": config["limit"],
                "reset_time": current_time + config["window"],
                "window": config["window"],
            }

    def get_rate_limit_headers(self, rate_limit_info: Dict[str, any]) -> Dict[str, str]:
        """Get HTTP headers for rate limit information."""
        return {
            "X-RateLimit-Limit": str(rate_limit_info["limit"]),
            "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
            "X-RateLimit-Reset": str(rate_limit_info["reset_time"]),
            "X-RateLimit-Window": str(rate_limit_info["window"]),
        }


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, rate_limiter: RateLimiter):
        self.app = app
        self.rate_limiter = rate_limiter

    async def __call__(self, scope, receive, send):
        """Process request with rate limiting."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from starlette.requests import Request
        from starlette.responses import Response

        request = Request(scope, receive)

        # Skip rate limiting for health checks and static files
        if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
            await self.app(scope, receive, send)
            return

        # Check rate limit
        is_limited, rate_limit_info = self.rate_limiter.is_rate_limited(request)

        if is_limited:
            headers = self.rate_limiter.get_rate_limit_headers(rate_limit_info)
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {rate_limit_info['limit']} per {rate_limit_info['window']} seconds",
                    "retry_after": rate_limit_info["reset_time"] - int(time.time()),
                },
                headers=headers,
            )
            await response(scope, receive, send)
            return

        # Create a custom send function to add rate limit headers
        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = self.rate_limiter.get_rate_limit_headers(rate_limit_info)
                message["headers"] = message.get("headers", [])
                for key, value in headers.items():
                    message["headers"].append([key.encode(), value.encode()])
            await send(message)

        await self.app(scope, receive, send_with_headers)


# Dependency for manual rate limit checking
def check_rate_limit(request: Request, rate_limiter: RateLimiter = None):
    """Dependency function to check rate limits manually."""
    if rate_limiter is None:
        # Use default rate limiter if none provided
        rate_limiter = RateLimiter()

    is_limited, rate_limit_info = rate_limiter.is_rate_limited(request)

    if is_limited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Limit: {rate_limit_info['limit']} per {rate_limit_info['window']} seconds",
                "retry_after": rate_limit_info["reset_time"] - int(time.time()),
            },
            headers=rate_limiter.get_rate_limit_headers(rate_limit_info),
        )

    return rate_limit_info


# Circuit breaker for Redis connections
class CircuitBreaker:
    """Circuit breaker pattern for Redis connections."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise e


# Enhanced rate limiter with circuit breaker
class EnhancedRateLimiter(RateLimiter):
    """Rate limiter with circuit breaker for Redis failures."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circuit_breaker = CircuitBreaker()

    def is_rate_limited(self, request: Request) -> tuple[bool, Dict[str, any]]:
        """Check rate limit with circuit breaker protection."""
        try:
            return self.circuit_breaker.call(super().is_rate_limited, request)
        except Exception:
            # If circuit breaker is open or Redis fails, allow request
            config = self.get_rate_limit_config(request)
            current_time = int(time.time())

            return False, {
                "limit": config["limit"],
                "remaining": config["limit"],
                "reset_time": current_time + config["window"],
                "window": config["window"],
            }
