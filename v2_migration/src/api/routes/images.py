"""Image API routes."""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, Query
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_image_service
from src.core.exceptions import NotFoundException
from src.models.schemas.image import ImageResponse, ImageMetadataResponse
from src.models.schemas.common import PaginationParams, FilterParams, PaginatedResponse
from src.services import ImageService

router = APIRouter(prefix="/images")


@router.get("", response_model=PaginatedResponse[ImageMetadataResponse])
async def list_images(
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max records to return"),
    photoset_id: Optional[UUID] = Query(default=None, description="Filter by photoset"),
    image_service: ImageService = Depends(get_image_service)
):
    """
    List images with pagination and optional filters.
    
    **Parameters:**
    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return (1-1000)
    - **photoset_id**: Filter images by photoset (optional)
    
    **Returns:** Paginated list of images with metadata
    """
    # Get images
    if photoset_id:
        images = await image_service.get_images_by_photoset(photoset_id, skip, limit)
        total = await image_service.count_images_by_photoset(photoset_id)
    else:
        images = await image_service.get_images(skip, limit)
        total = await image_service.count_images()
    
    # Convert to response models
    items = [
        ImageMetadataResponse(
            id=img.id,
            photoset_id=img.photoset_id,
            original_filename=img.original_filename,
            file_path=img.file_path,  # Missing field!
            width=img.width,
            height=img.height,
            file_size=img.file_size,
            mime_type=img.mime_type,
            created_at=img.created_at,
            updated_at=img.updated_at,
            has_caption=False,  # TODO: Check if caption exists
            has_crop=False,  # TODO: Check if crop exists
            tags=[]  # TODO: Get tags
        )
        for img in images
    ]
    
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


@router.get("/{image_id}")
async def get_image_file(
    image_id: UUID,
    image_service: ImageService = Depends(get_image_service)
):
    """Get image file with cache diagnostics."""
    try:
        image_data, diagnostics = await image_service.get_image_data(image_id)
        
        return Response(
            content=image_data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=31536000",
                "X-Cache-Status": "HIT" if diagnostics["cache_hit"] else "MISS",
                "X-Timing-Backend-Total": str(diagnostics["total_time_ms"]),
            }
        )
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e.message))


@router.get("/{image_id}/metadata", response_model=ImageMetadataResponse)
async def get_image_metadata(
    image_id: UUID,
    image_service: ImageService = Depends(get_image_service)
):
    """Get image metadata."""
    try:
        image = await image_service.get_image(image_id)
        # TODO: Build full metadata response with caption, tags, etc.
        return ImageMetadataResponse(
            id=image.id,
            photoset_id=image.photoset_id,
            original_filename=image.original_filename,
            file_path=image.file_path,  # Missing field!
            width=image.width,
            height=image.height,
            file_size=image.file_size,
            mime_type=image.mime_type,
            created_at=image.created_at,
            updated_at=image.updated_at,
            has_caption=False,  # TODO: Check if caption exists
            has_crop=False,  # TODO: Check if crop exists
            tags=[]  # TODO: Get tags
        )
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e.message))


@router.get("/{image_id}/thumbnail")
async def get_image_thumbnail(
    image_id: UUID,
    image_service: ImageService = Depends(get_image_service)
):
    """
    Get scaled-down thumbnail for fast browsing.
    Maintains original aspect ratio with reduced quality.
    """
    try:
        thumbnail_data = await image_service.get_thumbnail_data(image_id)
        if not thumbnail_data:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        
        return Response(
            content=thumbnail_data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=31536000",
            }
        )
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e.message))


# TODO: Add more endpoints:
# - POST /images (upload)
# - DELETE /images/{id}
# - POST /images/export

