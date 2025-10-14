"""Redis client configuration and utilities."""
import json
from typing import Any, Optional
import redis.asyncio as redis

from .config import settings


class RedisClient:
    """Async Redis client wrapper with utility methods."""
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Initialize Redis connection."""
        if self._client is None:
            self._client = await redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
    
    async def disconnect(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        await self.connect()
        return await self._client.get(key)
    
    async def set(
        self, 
        key: str, 
        value: str, 
        ex: Optional[int] = None
    ):
        """
        Set key-value pair with optional expiration.
        
        Args:
            key: Cache key
            value: Value to store
            ex: Expiration time in seconds
        """
        await self.connect()
        await self._client.set(key, value, ex=ex)
    
    async def delete(self, key: str):
        """Delete key."""
        await self.connect()
        await self._client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        await self.connect()
        return bool(await self._client.exists(key))
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON-serialized value."""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None
    
    async def set_json(
        self, 
        key: str, 
        value: Any, 
        ex: Optional[int] = None
    ):
        """Set JSON-serialized value."""
        json_value = json.dumps(value)
        await self.set(key, json_value, ex=ex)
    
    async def get_bytes(self, key: str) -> Optional[bytes]:
        """Get binary value."""
        await self.connect()
        # Create a new client that doesn't decode responses
        binary_client = await redis.from_url(
            settings.redis_url,
            decode_responses=False
        )
        value = await binary_client.get(key)
        await binary_client.close()
        return value
    
    async def set_bytes(
        self, 
        key: str, 
        value: bytes, 
        ex: Optional[int] = None
    ):
        """Set binary value."""
        await self.connect()
        # Create a new client that doesn't decode responses
        binary_client = await redis.from_url(
            settings.redis_url,
            decode_responses=False
        )
        await binary_client.set(key, value, ex=ex)
        await binary_client.close()
    
    async def clear_pattern(self, pattern: str):
        """Delete all keys matching pattern."""
        await self.connect()
        cursor = 0
        while True:
            cursor, keys = await self._client.scan(
                cursor=cursor, 
                match=pattern, 
                count=100
            )
            if keys:
                await self._client.delete(*keys)
            if cursor == 0:
                break
    
    async def ping(self) -> bool:
        """Check if Redis is accessible."""
        try:
            await self.connect()
            return await self._client.ping()
        except Exception:
            return False


# Singleton instance
redis_client = RedisClient()


async def get_redis() -> RedisClient:
    """Dependency for FastAPI."""
    await redis_client.connect()
    return redis_client


