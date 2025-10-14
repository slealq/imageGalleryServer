"""Photoset API routes."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_photoset_service
from src.core.exceptions import NotFoundException
from src.models.schemas.photoset import PhotosetResponse, PhotosetCreate
from src.models.schemas.common import PaginationParams
from src.services import PhotosetService

router = APIRouter(prefix="/photosets")


@router.get("/{photoset_id}", response_model=PhotosetResponse)
async def get_photoset(
    photoset_id: UUID,
    photoset_service: PhotosetService = Depends(get_photoset_service)
):
    """Get photoset by ID."""
    try:
        photoset = await photoset_service.get_photoset(photoset_id)
        image_count = await photoset_service.get_image_count(photoset_id)
        
        return PhotosetResponse(
            id=photoset.id,
            name=photoset.name,
            source_url=photoset.source_url,
            date=photoset.date,
            year=photoset.year,
            original_archive_filename=photoset.original_archive_filename,
            extra_metadata=photoset.extra_metadata,
            created_at=photoset.created_at,
            updated_at=photoset.updated_at,
            image_count=image_count,
            tags=[]  # TODO: Get tags
        )
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e.message))


@router.post("", response_model=PhotosetResponse, status_code=201)
async def create_photoset(
    photoset_data: PhotosetCreate,
    photoset_service: PhotosetService = Depends(get_photoset_service)
):
    """Create a new photoset."""
    photoset = await photoset_service.create_photoset(
        name=photoset_data.name,
        source_url=photoset_data.source_url,
        year=photoset_data.year,
        extra_metadata=photoset_data.extra_metadata
    )
    
    return PhotosetResponse(
        id=photoset.id,
        name=photoset.name,
        source_url=photoset.source_url,
        date=photoset.date,
        year=photoset.year,
        original_archive_filename=photoset.original_archive_filename,
        extra_metadata=photoset.extra_metadata,
        created_at=photoset.created_at,
        updated_at=photoset.updated_at,
        image_count=0,
        tags=[]
    )


# TODO: Add more endpoints:
# - GET /photosets (list)
# - PUT /photosets/{id}
# - DELETE /photosets/{id}
# - POST /photosets/{id}/extract

