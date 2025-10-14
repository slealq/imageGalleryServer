"""Image API routes."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_image_service
from src.core.exceptions import NotFoundException
from src.models.schemas.image import ImageResponse, ImageMetadataResponse
from src.models.schemas.common import PaginationParams, FilterParams
from src.services import ImageService

router = APIRouter(prefix="/images")


@router.get("/{image_id}")
async def get_image_file(
    image_id: UUID,
    image_service: ImageService = Depends(get_image_service)
):
    """Get image file."""
    try:
        image_data = await image_service.get_image_data(image_id)
        return Response(
            content=image_data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=31536000",
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


# TODO: Add more endpoints:
# - POST /images (upload)
# - DELETE /images/{id}
# - GET /images (list with pagination)
# - POST /images/export

