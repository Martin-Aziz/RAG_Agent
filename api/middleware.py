"""Rate limiting middleware for FastAPI.

Implements token bucket rate limiting to prevent abuse and ensure fair usage.
"""
import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import asyncio


class TokenBucket:
    """Token bucket for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        async with self._lock:
            # Refill tokens based on elapsed time
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now
            
            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get wait time until tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Wait time in seconds
        """
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        by_ip: bool = True,
        by_user: bool = False
    ):
        """Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Sustained rate limit
            burst_size: Maximum burst size
            by_ip: Enable per-IP rate limiting
            by_user: Enable per-user rate limiting
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.by_ip = by_ip
        self.by_user = by_user
        
        # Token buckets per identifier
        self.buckets: Dict[str, TokenBucket] = {}
        
        # Cleanup old buckets periodically
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
    
    def _get_identifier(self, request: Request) -> str:
        """Get unique identifier for rate limiting.
        
        Args:
            request: HTTP request
            
        Returns:
            Unique identifier string
        """
        identifiers = []
        
        if self.by_ip:
            # Get real IP (handle proxies)
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                ip = forwarded_for.split(",")[0].strip()
            else:
                ip = request.client.host if request.client else "unknown"
            identifiers.append(f"ip:{ip}")
        
        if self.by_user:
            # Try to extract user ID from request
            try:
                body = request.state.body  # Cached body
                if body and "user_id" in body:
                    identifiers.append(f"user:{body['user_id']}")
            except:
                pass
        
        return "|".join(identifiers) if identifiers else "global"
    
    def _get_bucket(self, identifier: str) -> TokenBucket:
        """Get or create token bucket for identifier.
        
        Args:
            identifier: Unique identifier
            
        Returns:
            Token bucket instance
        """
        if identifier not in self.buckets:
            refill_rate = self.requests_per_minute / 60.0  # tokens per second
            self.buckets[identifier] = TokenBucket(
                capacity=self.burst_size,
                refill_rate=refill_rate
            )
        
        return self.buckets[identifier]
    
    def _cleanup_old_buckets(self):
        """Remove inactive buckets to prevent memory leaks."""
        now = time.time()
        
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        # Remove buckets inactive for > 1 hour
        inactive_threshold = now - 3600
        to_remove = [
            key for key, bucket in self.buckets.items()
            if bucket.last_refill < inactive_threshold
        ]
        
        for key in to_remove:
            del self.buckets[key]
        
        self._last_cleanup = now
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Skip rate limiting for health checks and static files
        if request.url.path in ["/health", "/metrics"] or request.url.path.startswith("/static"):
            return await call_next(request)
        
        # Get identifier and bucket
        identifier = self._get_identifier(request)
        bucket = self._get_bucket(identifier)
        
        # Try to consume token
        allowed = await bucket.consume(tokens=1)
        
        if not allowed:
            wait_time = bucket.get_wait_time(tokens=1)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests",
                    "retry_after": int(wait_time) + 1
                },
                headers={"Retry-After": str(int(wait_time) + 1)}
            )
        
        # Cleanup old buckets periodically
        self._cleanup_old_buckets()
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
        
        return response
