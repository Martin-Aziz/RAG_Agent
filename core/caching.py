"""Caching utilities for the RAG Agent system.

Implements multiple caching strategies including in-memory LRU cache,
TTL-based cache, and async-compatible caching decorators.
"""
import time
import hashlib
import json
import asyncio
from typing import Any, Optional, Dict, Callable, TypeVar, cast
from functools import wraps
from collections import OrderedDict
from threading import RLock


T = TypeVar('T')


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.lock = RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing item and move to end
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                # Remove oldest item
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was found and removed
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }


class TTLCache:
    """Cache with Time-To-Live (TTL) expiration."""
    
    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000):
        """Initialize TTL cache.
        
        Args:
            default_ttl: Default TTL in seconds
            max_size: Maximum number of items
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.lock = RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                
                if time.time() < expiry:
                    self.hits += 1
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self.lock:
            # Clean up if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_expired()
                
                # If still at capacity, remove oldest
                if len(self.cache) >= self.max_size:
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                    del self.cache[oldest_key]
            
            ttl = ttl if ttl is not None else self.default_ttl
            expiry = time.time() + ttl
            self.cache[key] = (value, expiry)
    
    def _evict_expired(self) -> None:
        """Remove expired entries (must be called with lock held)."""
        now = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if expiry <= now
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            self._evict_expired()
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }


def make_cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    # Create a stable representation
    key_parts = []
    
    for arg in args:
        if isinstance(arg, (str, int, float, bool, type(None))):
            key_parts.append(str(arg))
        else:
            # For complex types, use type name
            key_parts.append(f"{type(arg).__name__}:{id(arg)}")
    
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool, type(None))):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(f"{k}={type(v).__name__}:{id(v)}")
    
    # Hash the key for consistent length
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(
    cache: Optional[Any] = None,
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for caching function results.
    
    Args:
        cache: Cache instance to use (creates default LRUCache if None)
        ttl: TTL for cached values (only for TTLCache)
        key_func: Custom function to generate cache key
        
    Returns:
        Decorated function
    """
    if cache is None:
        cache = LRUCache()
    
    if key_func is None:
        key_func = make_cache_key
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            # Generate cache key
            cache_key = f"{func.__module__}.{func.__name__}:{key_func(*args, **kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return cast(T, result)
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            if isinstance(cache, TTLCache):
                cache.put(cache_key, result, ttl=ttl)
            else:
                cache.put(cache_key, result)
            
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            # Generate cache key
            cache_key = f"{func.__module__}.{func.__name__}:{key_func(*args, **kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return cast(T, result)
            
            # Compute result
            result = await func(*args, **kwargs)
            
            # Cache result
            if isinstance(cache, TTLCache):
                cache.put(cache_key, result, ttl=ttl)
            else:
                cache.put(cache_key, result)
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(Callable[..., T], async_wrapper)
        else:
            return cast(Callable[..., T], sync_wrapper)
    
    return decorator


# Global caches for common use cases
query_cache = TTLCache(default_ttl=300.0, max_size=1000)  # 5 minute TTL
embedding_cache = LRUCache(max_size=10000)  # Large cache for embeddings
retrieval_cache = TTLCache(default_ttl=600.0, max_size=500)  # 10 minute TTL


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all global caches.
    
    Returns:
        Dictionary with statistics for each cache
    """
    return {
        "query_cache": query_cache.get_stats(),
        "embedding_cache": embedding_cache.get_stats(),
        "retrieval_cache": retrieval_cache.get_stats()
    }


def clear_all_caches() -> None:
    """Clear all global caches."""
    query_cache.clear()
    embedding_cache.clear()
    retrieval_cache.clear()
