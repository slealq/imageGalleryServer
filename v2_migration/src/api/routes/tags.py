"""Tag API routes."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_tag_service
from src.core.exceptions import NotFoundException
from src.models.schemas.tag import TagResponse, AddTagRequest, TagListResponse
from src.services import TagService

router = APIRouter(prefix="/tags")


@router.get("", response_model=TagListResponse)
async def get_all_tags(
    tag_service: TagService = Depends(get_tag_service)
):
    """Get all tags grouped by type."""
    tags_grouped = await tag_service.get_tags_grouped()
    
    all_tags = []
    for tags_list in tags_grouped.values():
        all_tags.extend(tags_list)
    
    return TagListResponse(
        tags=[
            TagResponse(
                id=tag.id,
                name=tag.name,
                tag_type=tag.tag_type,
                created_at=tag.created_at
            )
            for tag in all_tags
        ],
        tags_by_type={
            tag_type: [
                TagResponse(
                    id=tag.id,
                    name=tag.name,
                    tag_type=tag.tag_type,
                    created_at=tag.created_at
                )
                for tag in tags
            ]
            for tag_type, tags in tags_grouped.items()
        },
        total=len(all_tags)
    )


# Image tag endpoints (prefix: /images)
image_router = APIRouter(prefix="/images")


@image_router.post("/{image_id}/tags", response_model=TagResponse)
async def add_tag_to_image(
    image_id: UUID,
    tag_data: AddTagRequest,
    tag_service: TagService = Depends(get_tag_service)
):
    """Add a tag to an image."""
    try:
        tag = await tag_service.add_tag_to_image(
            image_id,
            tag_data.tag_name,
            tag_data.tag_type
        )
        
        return TagResponse(
            id=tag.id,
            name=tag.name,
            tag_type=tag.tag_type,
            created_at=tag.created_at
        )
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e.message))


# Register image routes to main router
router.include_router(image_router)


# TODO: Add more endpoints:
# - GET /images/{id}/tags
# - DELETE /images/{id}/tags/{tag_id}
# - POST /photosets/{id}/tags
# - GET /photosets/{id}/tags

