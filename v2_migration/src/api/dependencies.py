"""Dependency injection for FastAPI routes."""
from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.services import (
    StorageService,
    CacheService,
    ImageService,
    ThumbnailService,
    PhotosetService,
    CaptionService,
    CropService,
    TagService,
)


# Singleton service instances
_storage_service: StorageService | None = None
_cache_service: CacheService | None = None


def get_storage_service() -> StorageService:
    """Get storage service (singleton)."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service


async def get_cache_service() -> CacheService:
    """Get cache service (singleton)."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


# Request-scoped services (get fresh instances with DB session)


async def get_image_service(
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
    cache: CacheService = Depends(get_cache_service),
) -> ImageService:
    """Get image service."""
    return ImageService(db, storage, cache)


async def get_thumbnail_service(
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
    cache: CacheService = Depends(get_cache_service),
) -> ThumbnailService:
    """Get thumbnail service."""
    return ThumbnailService(db, storage, cache)


async def get_photoset_service(
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> PhotosetService:
    """Get photoset service."""
    return PhotosetService(db, storage)


async def get_caption_service(
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> CaptionService:
    """Get caption service."""
    return CaptionService(db, storage)


async def get_crop_service(
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
    cache: CacheService = Depends(get_cache_service),
) -> CropService:
    """Get crop service."""
    return CropService(db, storage, cache)


async def get_tag_service(
    db: AsyncSession = Depends(get_db),
) -> TagService:
    """Get tag service."""
    return TagService(db)

