"""API routes."""
from fastapi import APIRouter

from . import health, images, photosets, captions, crops, tags

# Create API v2 router
api_router = APIRouter(prefix="/api/v2")

# Include all route modules
api_router.include_router(health.router, tags=["health"])
api_router.include_router(images.router, tags=["images"])
api_router.include_router(photosets.router, tags=["photosets"])
api_router.include_router(captions.router, tags=["captions"])
api_router.include_router(crops.router, tags=["crops"])
api_router.include_router(tags.router, tags=["tags"])

__all__ = ["api_router"]

