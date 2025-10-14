"""Crop API routes."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response

from src.api.dependencies import get_crop_service
from src.core.exceptions import NotFoundException, ImageProcessingException
from src.models.schemas.crop import CropResponse, CropCreate, CropWithImageResponse
from src.services import CropService

router = APIRouter(prefix="/images")


@router.get("/{image_id}/crop", response_model=CropWithImageResponse)
async def get_crop(
    image_id: UUID,
    crop_service: CropService = Depends(get_crop_service)
):
    """Get crop information and image URL."""
    crop = await crop_service.get_crop(image_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")
    
    return CropWithImageResponse(
        crop_info=CropResponse(
            id=crop.id,
            image_id=crop.image_id,
            target_size=crop.target_size,
            normalized_delta_x=crop.normalized_delta_x,
            normalized_delta_y=crop.normalized_delta_y,
            crop_file_path=crop.crop_file_path,
            created_at=crop.created_at,
            updated_at=crop.updated_at
        ),
        image_url=f"/api/v2/images/{image_id}/cropped"
    )


@router.get("/{image_id}/cropped")
async def get_cropped_image(
    image_id: UUID,
    crop_service: CropService = Depends(get_crop_service)
):
    """Get the cropped image file."""
    crop_data = await crop_service.get_crop_image(image_id)
    if not crop_data:
        raise HTTPException(status_code=404, detail="Cropped image not found")
    
    return Response(content=crop_data, media_type="image/png")


@router.post("/{image_id}/crop")
async def create_crop(
    image_id: UUID,
    crop_data: CropCreate,
    crop_service: CropService = Depends(get_crop_service)
):
    """Create or update a crop."""
    try:
        crop, crop_image = await crop_service.create_crop(
            image_id,
            crop_data.target_size,
            crop_data.normalized_deltas.x,
            crop_data.normalized_deltas.y
        )
        
        return Response(content=crop_image, media_type="image/png")
    except (NotFoundException, ImageProcessingException) as e:
        raise HTTPException(status_code=400 if isinstance(e, ImageProcessingException) else 404, detail=str(e.message))


# TODO: Add preview endpoint:
# - GET /images/{id}/preview/{size}

