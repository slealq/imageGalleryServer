"""Health check endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_cache_service
from src.core.database import engine
from src.services import CacheService

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    database: str
    cache: str


@router.get("/health", response_model=HealthResponse)
async def health_check(
    cache: CacheService = Depends(get_cache_service)
):
    """
    Health check endpoint.
    
    Returns service health status.
    """
    # Check database
    try:
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    # Check cache
    try:
        cache_ok = await cache.ping()
        cache_status = "healthy" if cache_ok else "unavailable"
    except Exception:
        cache_status = "unavailable"
    
    # Overall status
    status = "healthy" if db_status == "healthy" else "unhealthy"
    
    return HealthResponse(
        status=status,
        database=db_status,
        cache=cache_status
    )

