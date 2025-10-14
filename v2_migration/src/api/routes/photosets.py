"""Photoset API routes."""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_photoset_service
from src.core.exceptions import NotFoundException
from src.models.schemas.photoset import PhotosetResponse, PhotosetCreate
from src.models.schemas.common import PaginationParams, PaginatedResponse
from src.services import PhotosetService

router = APIRouter(prefix="/photosets")


@router.get("", response_model=PaginatedResponse[PhotosetResponse])
async def list_photosets(
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=50, ge=1, le=1000, description="Max records to return"),
    year: Optional[int] = Query(default=None, description="Filter by year"),
    search: Optional[str] = Query(default=None, description="Search by name"),
    photoset_service: PhotosetService = Depends(get_photoset_service)
):
    """
    List photosets with pagination and optional filters.
    
    **Parameters:**
    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return (1-1000)
    - **year**: Filter photosets by year (optional)
    - **search**: Search photosets by name (optional)
    
    **Returns:** Paginated list of photosets with metadata
    """
    # Get photosets
    if year:
        photosets = await photoset_service.get_photosets_by_year(year, skip, limit)
    elif search:
        photosets = await photoset_service.search_photosets(search, skip, limit)
    else:
        photosets = await photoset_service.get_photosets(skip, limit)
    
    # Get total count
    total = await photoset_service.count_photosets()
    
    # Convert to response models
    items = []
    for ps in photosets:
        image_count = await photoset_service.get_image_count(ps.id)
        items.append(PhotosetResponse(
            id=ps.id,
            name=ps.name,
            source_url=ps.source_url,
            date=ps.date,
            year=ps.year,
            original_archive_filename=ps.original_archive_filename,
            extra_metadata=ps.extra_metadata,
            created_at=ps.created_at,
            updated_at=ps.updated_at,
            image_count=image_count,
            tags=[]  # TODO: Get tags
        ))
    
    # Calculate pagination fields
    page = (skip // limit) + 1 if limit > 0 else 1
    total_pages = (total + limit - 1) // limit if limit > 0 else 1
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=limit,
        total_pages=total_pages
    )


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


@router.get("/{photoset_id}/images", response_model=PaginatedResponse[dict])
async def list_photoset_images(
    photoset_id: UUID,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    photoset_service: PhotosetService = Depends(get_photoset_service)
):
    """
    List images in a specific photoset.
    
    **Parameters:**
    - **photoset_id**: UUID of the photoset
    - **skip**: Number of images to skip
    - **limit**: Maximum number of images to return
    
    **Returns:** Paginated list of images in the photoset
    """
    try:
        images = await photoset_service.get_photoset_images(photoset_id, skip, limit)
        total = await photoset_service.get_image_count(photoset_id)
        
        # Calculate pagination fields
        page = (skip // limit) + 1 if limit > 0 else 1
        total_pages = (total + limit - 1) // limit if limit > 0 else 1
        
        return PaginatedResponse(
            items=[{
                "id": str(img.id),
                "original_filename": img.original_filename,
                "file_path": img.file_path,
                "width": img.width,
                "height": img.height,
                "file_size": img.file_size,
                "mime_type": img.mime_type,
                "created_at": img.created_at.isoformat(),
                "updated_at": img.updated_at.isoformat(),
            } for img in images],
            total=total,
            page=page,
            page_size=limit,
            total_pages=total_pages
        )
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e.message))


# TODO: Add more endpoints:
# - PUT /photosets/{id}
# - DELETE /photosets/{id}
# - POST /photosets/{id}/extract

