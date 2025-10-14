"""Cache service wrapping Redis operations."""
from typing import Any, Optional
from uuid import UUID

from src.core.redis import redis_client
from src.core.config import settings
from src.core.exceptions import CacheException


class CacheService:
    """Service for caching operations using Redis."""
    
    def __init__(self):
        """Initialize cache service."""
        self.redis = redis_client
        self.metadata_ttl = settings.metadata_cache_ttl_seconds
    
    # Image caching
    
    async def get_image(self, image_id: UUID) -> Optional[bytes]:
        """
        Get cached image data.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Image bytes or None if not cached
        """
        try:
            key = f"image:{image_id}"
            return await self.redis.get_bytes(key)
        except Exception as e:
            # Don't raise on cache errors, just return None
            print(f"Cache get error: {e}")
            return None
    
    async def set_image(self, image_id: UUID, image_data: bytes, ttl: Optional[int] = None):
        """
        Cache image data.
        
        Args:
            image_id: Image UUID
            image_data: Image binary data
            ttl: Time to live in seconds (optional)
        """
        try:
            key = f"image:{image_id}"
            await self.redis.set_bytes(key, image_data, ex=ttl)
        except Exception as e:
            # Don't raise on cache errors
            print(f"Cache set error: {e}")
    
    async def delete_image(self, image_id: UUID):
        """
        Delete cached image.
        
        Args:
            image_id: Image UUID
        """
        try:
            key = f"image:{image_id}"
            await self.redis.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")
    
    # Thumbnail caching
    
    async def get_thumbnail(self, image_id: UUID) -> Optional[bytes]:
        """
        Get cached thumbnail data.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Thumbnail bytes or None if not cached
        """
        try:
            key = f"thumbnail:{image_id}"
            return await self.redis.get_bytes(key)
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set_thumbnail(self, image_id: UUID, thumbnail_data: bytes):
        """
        Cache thumbnail data.
        
        Args:
            image_id: Image UUID
            thumbnail_data: Thumbnail binary data
        """
        try:
            key = f"thumbnail:{image_id}"
            await self.redis.set_bytes(key, thumbnail_data)
        except Exception as e:
            print(f"Cache set error: {e}")
    
    async def delete_thumbnail(self, image_id: UUID):
        """
        Delete cached thumbnail data.
        
        Args:
            image_id: Image UUID
        """
        try:
            key = f"thumbnail:{image_id}"
            await self.redis.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")
    
    # Metadata caching
    
    async def get_metadata(self, key: str) -> Optional[Any]:
        """
        Get cached metadata (JSON).
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None
        """
        try:
            full_key = f"metadata:{key}"
            return await self.redis.get_json(full_key)
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set_metadata(self, key: str, data: Any):
        """
        Cache metadata (JSON) with TTL.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        try:
            full_key = f"metadata:{key}"
            await self.redis.set_json(full_key, data, ex=self.metadata_ttl)
        except Exception as e:
            print(f"Cache set error: {e}")
    
    async def delete_metadata(self, key: str):
        """
        Delete cached metadata.
        
        Args:
            key: Cache key
        """
        try:
            full_key = f"metadata:{key}"
            await self.redis.delete(full_key)
        except Exception as e:
            print(f"Cache delete error: {e}")
    
    # Pattern-based operations
    
    async def clear_pattern(self, pattern: str):
        """
        Clear all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "image:*")
        """
        try:
            await self.redis.clear_pattern(pattern)
        except Exception as e:
            print(f"Cache clear error: {e}")
    
    async def invalidate_image_cache(self, image_id: UUID):
        """
        Invalidate all caches related to an image.
        
        Args:
            image_id: Image UUID
        """
        await self.delete_image(image_id)
        await self.clear_pattern(f"thumbnail:{image_id}:*")
        await self.delete_metadata(f"image:{image_id}")
    
    # Health check
    
    async def ping(self) -> bool:
        """
        Check if cache is available.
        
        Returns:
            True if cache is responding
        """
        try:
            return await self.redis.ping()
        except Exception:
            return False

