"""Filters API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_filters_service
from src.models.schemas.filters import FiltersResponse
from src.services import FiltersService

router = APIRouter(prefix="/filters")


@router.get("", response_model=FiltersResponse)
async def get_available_filters(
    filters_service: FiltersService = Depends(get_filters_service)
):
    """
    Get all available filter options.
    
    Returns available actors, tags, and years that can be used to filter images.
    
    **Returns:**
    - **actors**: List of available actor names
    - **tags**: List of available tag names (excluding actors)
    - **years**: List of available years from photosets
    """
    filters_data = await filters_service.get_available_filters()
    
    return FiltersResponse(
        actors=filters_data["actors"],
        tags=filters_data["tags"],
        years=filters_data["years"]
    )

