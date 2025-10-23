"""Legacy API routes for backward compatibility with old service.

These routes maintain the same paths and behavior as the original service
to ensure zero-downtime migration for existing clients.
"""
from __future__ import annotations

import io
import json
import zipfile
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.dependencies import (
    get_image_service,
    get_caption_service,
    get_crop_service,
    get_tag_service,
)
from src.core.exceptions import NotFoundException, ImageProcessingException
from src.models.schemas.caption import CaptionCreate, CaptionGenerateRequest
from src.models.schemas.crop import CropCreate, NormalizedDeltas
from src.models.schemas.tag import AddTagRequest
from src.services import ImageService, CaptionService, CropService, TagService

# Create legacy router (no prefix - routes at root level)
router = APIRouter()


# Legacy request/response models that match old service
class LegacyCaptionRequest(BaseModel):
    caption: str
    prompt: Optional[str] = None


class LegacyCaptionResponse(BaseModel):
    caption: str


class LegacyTagResponse(BaseModel):
    tags: List[str]


class LegacyAddTagRequest(BaseModel):
    tag: str


class LegacyCropRequest(BaseModel):
    imageId: str
    targetSize: int
    normalizedDeltas: NormalizedDeltas


class LegacyExportRequest(BaseModel):
    imageIds: List[str]


# Helper function to convert string image_id to UUID
async def get_image_uuid_from_string_id(
    image_id: str, 
    image_service: ImageService
) -> UUID:
    """
    Convert string image ID to UUID by looking up image.
    
    This function supports:
    1. Direct UUID strings
    2. Filename-based lookups (searches original_filename and file_path)
    
    Args:
        image_id: String identifier (UUID or filename pattern)
        image_service: Image service instance
        
    Returns:
        UUID of the found image
        
    Raises:
        HTTPException: If image not found
    """
    image = await image_service.find_image_by_string_id(image_id)
    
    if not image:
        raise HTTPException(
            status_code=404,
            detail=f"Image with ID '{image_id}' not found"
        )
    
    return image.id


# ============================================================================
# IMAGE ROUTES
# ============================================================================

@router.get("/images")
async def get_images_legacy(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Number of images per page"),
    actor: Optional[str] = Query(None, description="Filter by actor name"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    year: Optional[str] = Query(None, description="Filter by year"),
    has_caption: Optional[bool] = Query(None, description="Filter for images with captions"),
    has_crop: Optional[bool] = Query(None, description="Filter for images with crops"),
    image_service: ImageService = Depends(get_image_service)
):
    """
    Legacy endpoint: Get paginated images with filtering.
    
    Note: This endpoint is provided for backward compatibility.
    New clients should use /api/v2/images instead.
    
    TODO: Implement filtering by actor, tag, year, has_caption, has_crop
    """
    # Calculate skip/limit from page/page_size
    skip = (page - 1) * page_size
    limit = page_size
    
    # Get images
    images = await image_service.get_images(skip, limit)
    total = await image_service.count_images()
    
    # Calculate pagination
    total_pages = (total + page_size - 1) // page_size
    
    # Build response in old format
    # TODO: Add full metadata with tags, captions, crops
    return {
        "images": [
            {
                "id": str(img.id),
                "filename": img.original_filename,
                "size": img.file_size,
                "width": img.width,
                "height": img.height,
                "mime_type": img.mime_type,
                "created_at": img.created_at.isoformat(),
                "has_caption": False,  # TODO
                "has_crop": False,  # TODO
                "has_tags": False,  # TODO
                "tags": [],  # TODO
                "actors": [],  # TODO
                "year": None,  # TODO
            }
            for img in images
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }


@router.get("/images/batch")
async def get_images_batch_legacy(
    request: Request,
    start_page: int = Query(1, ge=1, description="Starting page number"),
    num_pages: int = Query(3, ge=1, le=5, description="Number of pages to fetch"),
    page_size: int = Query(50, ge=1, le=100, description="Number of images per page"),
    actor: Optional[str] = Query(None, description="Filter by actor name"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    year: Optional[str] = Query(None, description="Filter by year"),
    has_caption: Optional[bool] = Query(None, description="Filter for images with captions"),
    has_crop: Optional[bool] = Query(None, description="Filter for images with crops"),
    image_service: ImageService = Depends(get_image_service)
):
    """
    Legacy endpoint: Get multiple pages of images at once.
    
    Note: This endpoint is provided for backward compatibility.
    """
    responses = []
    for page in range(start_page, start_page + num_pages):
        response = await get_images_legacy(
            request=request,
            page=page,
            page_size=page_size,
            actor=actor,
            tag=tag,
            year=year,
            has_caption=has_caption,
            has_crop=has_crop,
            image_service=image_service
        )
        responses.append(response)
        if not response["images"]:  # Stop if we hit the end
            break
    return responses


@router.get("/images/{image_id}")
async def get_image_file_legacy(
    image_id: str,
    image_service: ImageService = Depends(get_image_service)
):
    """
    Legacy endpoint: Get image file by string ID.
    
    Note: This endpoint is provided for backward compatibility.
    Includes diagnostic headers for cache performance monitoring.
    """
    import time
    
    request_start = time.time()
    
    try:
        # Try to convert to UUID
        id_lookup_start = time.time()
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        id_lookup_time = round((time.time() - id_lookup_start) * 1000, 2)
        
        # Get image data with diagnostics
        image_data, diagnostics = await image_service.get_image_data(uuid_id)
        
        # Calculate total request time
        total_request_time = round((time.time() - request_start) * 1000, 2)
        
        # Build diagnostic headers
        headers = {
            "Cache-Control": "public, max-age=31536000",
            
            # Cache diagnostics
            "X-Cache-Status": "HIT" if diagnostics["cache_hit"] else "MISS",
            "X-Cache-Enabled": str(diagnostics["cache_enabled"]),
            
            # Timing breakdown (in milliseconds)
            "X-Timing-ID-Lookup": str(id_lookup_time),
            "X-Timing-Cache-Check": str(diagnostics["cache_check_time_ms"]),
            "X-Timing-DB-Lookup": str(diagnostics["db_lookup_time_ms"]),
            "X-Timing-File-Read": str(diagnostics["file_read_time_ms"]),
            "X-Timing-Image-Processing": str(diagnostics["image_processing_time_ms"]),
            "X-Timing-Cache-Write": str(diagnostics["cache_write_time_ms"]),
            "X-Timing-Backend-Total": str(diagnostics["total_time_ms"]),
            "X-Timing-Request-Total": str(total_request_time),
        }
        
        return Response(
            content=image_data,
            media_type="image/jpeg",
            headers=headers
        )
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


# ============================================================================
# CAPTION ROUTES
# ============================================================================

@router.post("/images/{image_id}/caption")
async def save_caption_legacy(
    image_id: str,
    caption_request: LegacyCaptionRequest,
    caption_service: CaptionService = Depends(get_caption_service),
    image_service: ImageService = Depends(get_image_service)
):
    """Legacy endpoint: Save a caption for an image."""
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        await caption_service.save_caption(
            uuid_id,
            caption_request.caption,
            generator_type="manual"
        )
        
        return {"message": "Caption saved successfully"}
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


@router.get("/images/{image_id}/caption")
async def get_caption_legacy(
    image_id: str,
    caption_service: CaptionService = Depends(get_caption_service),
    image_service: ImageService = Depends(get_image_service)
):
    """Legacy endpoint: Get caption for an image."""
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        caption = await caption_service.get_caption(uuid_id)
        if not caption:
            raise HTTPException(status_code=404, detail="Caption not found")
        
        return LegacyCaptionResponse(caption=caption.caption)
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


@router.post("/images/{image_id}/generate-caption")
async def generate_caption_legacy(
    image_id: str,
    request: LegacyCaptionRequest,
    caption_service: CaptionService = Depends(get_caption_service),
    image_service: ImageService = Depends(get_image_service)
):
    """Legacy endpoint: Generate a caption for an image."""
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        caption_text = await caption_service.generate_caption(
            uuid_id,
            request.prompt,
            save=False  # Don't auto-save in generate endpoint
        )
        
        return {"caption": caption_text}
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


@router.post("/images/{image_id}/stream-caption")
async def stream_caption_legacy(
    image_id: str,
    request: LegacyCaptionRequest,
    caption_service: CaptionService = Depends(get_caption_service),
    image_service: ImageService = Depends(get_image_service)
):
    """Legacy endpoint: Stream caption generation."""
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        async def generate():
            try:
                async for chunk in caption_service.stream_caption(uuid_id, request.prompt):
                    if chunk:
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


# ============================================================================
# TAG ROUTES
# ============================================================================

@router.get("/images/{image_id}/tags")
async def get_image_tags_legacy(
    image_id: str,
    tag_service: TagService = Depends(get_tag_service),
    image_service: ImageService = Depends(get_image_service)
):
    """Legacy endpoint: Get all tags for an image."""
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        tags = await tag_service.get_image_tags(uuid_id)
        
        return LegacyTagResponse(tags=[tag.name for tag in tags])
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


@router.get("/images/{image_id}/custom-tags")
async def get_image_custom_tags_legacy(
    image_id: str,
    tag_service: TagService = Depends(get_tag_service),
    image_service: ImageService = Depends(get_image_service)
):
    """Legacy endpoint: Get only custom tags for an image."""
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        # Get all tags and filter for custom type
        tags = await tag_service.get_image_tags(uuid_id)
        custom_tags = [tag.name for tag in tags if tag.tag_type == "custom"]
        
        return LegacyTagResponse(tags=custom_tags)
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


@router.post("/images/{image_id}/tags")
async def add_image_tag_legacy(
    image_id: str,
    request: LegacyAddTagRequest,
    tag_service: TagService = Depends(get_tag_service),
    image_service: ImageService = Depends(get_image_service)
):
    """Legacy endpoint: Add a tag to an image."""
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        await tag_service.add_tag_to_image(
            uuid_id,
            request.tag,
            tag_type="custom"
        )
        
        return {"message": f"Tag '{request.tag}' added successfully"}
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


# ============================================================================
# CROP ROUTES
# ============================================================================

@router.get("/images/{image_id}/crop")
async def get_crop_legacy(
    image_id: str,
    crop_service: CropService = Depends(get_crop_service),
    image_service: ImageService = Depends(get_image_service)
):
    """Legacy endpoint: Get crop information."""
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        crop = await crop_service.get_crop(uuid_id)
        if not crop:
            raise HTTPException(status_code=404, detail="Crop not found")
        
        return {
            "cropInfo": {
                "targetSize": crop.target_size,
                "normalizedDeltas": {
                    "x": crop.normalized_delta_x,
                    "y": crop.normalized_delta_y
                }
            },
            "imageUrl": f"/images/{image_id}/cropped"
        }
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


@router.get("/images/{image_id}/cropped")
async def get_cropped_image_legacy(
    image_id: str,
    crop_service: CropService = Depends(get_crop_service),
    image_service: ImageService = Depends(get_image_service)
):
    """Legacy endpoint: Get the cropped image file."""
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        crop_data = await crop_service.get_crop_image(uuid_id)
        if not crop_data:
            raise HTTPException(status_code=404, detail="Cropped image not found")
        
        return Response(content=crop_data, media_type="image/png")
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


@router.post("/images/{image_id}/crop")
async def create_crop_legacy(
    image_id: str,
    crop_request: LegacyCropRequest,
    crop_service: CropService = Depends(get_crop_service),
    image_service: ImageService = Depends(get_image_service)
):
    """Legacy endpoint: Create or update a crop."""
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        crop, crop_image = await crop_service.create_crop(
            uuid_id,
            crop_request.targetSize,
            crop_request.normalizedDeltas.x,
            crop_request.normalizedDeltas.y
        )
        
        return Response(content=crop_image, media_type="image/png")
    except (NotFoundException, ImageProcessingException) as e:
        status_code = 400 if isinstance(e, ImageProcessingException) else 404
        raise HTTPException(status_code=status_code, detail=str(e.message))


@router.get("/images/{image_id}/preview/{target_size}")
async def get_image_preview_legacy(
    image_id: str,
    target_size: int,
    image_service: ImageService = Depends(get_image_service)
):
    """
    Legacy endpoint: Get a scaled preview of the image.
    
    Returns a preview image scaled to fit within target_size while maintaining aspect ratio.
    """
    try:
        uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
        
        # TODO: Implement preview generation in ImageService
        # For now, return the original image
        # This would need to be added to the image service
        image_data = await image_service.get_image_data(uuid_id)
        
        return Response(content=image_data, media_type="image/png")
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Image not found")


# ============================================================================
# UTILITY ROUTES
# ============================================================================

@router.post("/api/export-images")
async def export_images_legacy(
    request: LegacyExportRequest,
    image_service: ImageService = Depends(get_image_service),
    crop_service: CropService = Depends(get_crop_service),
    caption_service: CaptionService = Depends(get_caption_service)
):
    """
    Legacy endpoint: Export images with crops and captions as a zip file.
    
    Creates a zip file containing:
    - Cropped images (if available)
    - Caption text files (if available)
    """
    try:
        # Create a BytesIO object to store the zip file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for image_id in request.imageIds:
                try:
                    # Convert string ID to UUID
                    uuid_id = await get_image_uuid_from_string_id(image_id, image_service)
                    
                    # Get image to determine base name
                    image = await image_service.get_image(uuid_id)
                    base_name = image.original_filename.rsplit('.', 1)[0] if '.' in image.original_filename else image.original_filename
                    
                    # Add cropped image if it exists
                    try:
                        crop_data = await crop_service.get_crop_image(uuid_id)
                        if crop_data:
                            zip_file.writestr(f"{base_name}.png", crop_data)
                    except:
                        pass  # No crop available
                    
                    # Add caption if it exists
                    try:
                        caption = await caption_service.get_caption(uuid_id)
                        if caption:
                            zip_file.writestr(f"{base_name}.txt", caption.caption)
                    except:
                        pass  # No caption available
                        
                except Exception as e:
                    # Skip images that can't be processed
                    print(f"Warning: Could not export image {image_id}: {str(e)}")
                    continue
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        # Return the zip file
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=exported_images.zip"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/warmup")
async def warmup_cache_legacy(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Number of images per page"),
):
    """
    Legacy endpoint: Pre-warm the image cache.
    
    TODO: Implement cache warming in v2
    """
    # V2 uses database-backed storage, so cache warming is less critical
    # Could implement thumbnail pre-generation here
    return {
        "status": "success",
        "message": "Cache warming not required in v2 (database-backed)",
        "page": page,
        "page_size": page_size
    }

