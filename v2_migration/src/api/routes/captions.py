"""Caption API routes."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_caption_service
from src.core.exceptions import NotFoundException
from src.models.schemas.caption import CaptionResponse, CaptionCreate, CaptionGenerateRequest
from src.services import CaptionService

router = APIRouter(prefix="/images")


@router.get("/{image_id}/caption", response_model=CaptionResponse)
async def get_caption(
    image_id: UUID,
    caption_service: CaptionService = Depends(get_caption_service)
):
    """Get caption for an image."""
    caption = await caption_service.get_caption(image_id)
    if not caption:
        raise HTTPException(status_code=404, detail="Caption not found")
    
    return CaptionResponse(
        id=caption.id,
        image_id=caption.image_id,
        caption=caption.caption,
        generator_type=caption.generator_type,
        generator_metadata=caption.generator_metadata,
        created_at=caption.created_at,
        updated_at=caption.updated_at
    )


@router.post("/{image_id}/caption", response_model=CaptionResponse)
async def save_caption(
    image_id: UUID,
    caption_data: CaptionCreate,
    caption_service: CaptionService = Depends(get_caption_service)
):
    """Save or update a caption."""
    try:
        caption = await caption_service.save_caption(
            image_id,
            caption_data.caption,
            caption_data.generator_type
        )
        
        return CaptionResponse(
            id=caption.id,
            image_id=caption.image_id,
            caption=caption.caption,
            generator_type=caption.generator_type,
            generator_metadata=caption.generator_metadata,
            created_at=caption.created_at,
            updated_at=caption.updated_at
        )
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e.message))


@router.post("/{image_id}/caption/generate")
async def generate_caption(
    image_id: UUID,
    request: CaptionGenerateRequest,
    caption_service: CaptionService = Depends(get_caption_service)
):
    """Generate a caption for an image."""
    try:
        caption_text = await caption_service.generate_caption(
            image_id,
            request.prompt,
            save=True
        )
        return {"caption": caption_text}
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e.message))


# TODO: Add streaming endpoint:
# - POST /images/{id}/caption/stream

