from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
import time
import json

from ..models.models import CaptionRequest, CaptionResponse
from ..services.cache_service import (
    caption_cache,
    PHOTOSET_METADATA_CACHE,
    save_caption_cache
)
from config import IMAGES_DIR

router = APIRouter()

@router.post("/caption", response_model=CaptionResponse)
async def add_caption(request: Request, caption_request: CaptionRequest):
    """Add or update a caption for an image."""
    start_time = time.time()
    try:
        # Validate image exists
        file_start = time.time()
        image_path = Path(IMAGES_DIR) / f"{caption_request.imageId}.jpg"
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        file_time = time.time() - file_start
        
        # Update caption cache
        cache_start = time.time()
        caption_cache[caption_request.imageId] = caption_request.caption
        cache_time = time.time() - cache_start
        
        # Save caption cache to disk
        save_start = time.time()
        save_caption_cache()
        save_time = time.time() - save_start
        
        # Update photoset metadata if it exists
        metadata_start = time.time()
        if caption_request.imageId in PHOTOSET_METADATA_CACHE:
            PHOTOSET_METADATA_CACHE[caption_request.imageId]['caption'] = caption_request.caption
        metadata_time = time.time() - metadata_start
        
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Caption update took {total_time:.3f}s (file check: {file_time:.3f}s, cache update: {cache_time:.3f}s, save: {save_time:.3f}s, metadata: {metadata_time:.3f}s)")
        
        return CaptionResponse(
            imageId=caption_request.imageId,
            caption=caption_request.caption
        )
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Error in caption update after {total_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/caption/{image_id}", response_model=CaptionResponse)
async def get_caption(request: Request, image_id: str):
    """Get the caption for an image."""
    start_time = time.time()
    try:
        # Check if image exists
        file_start = time.time()
        image_path = Path(IMAGES_DIR) / f"{image_id}.jpg"
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        file_time = time.time() - file_start
        
        # Get caption from cache
        cache_start = time.time()
        caption = caption_cache.get(image_id)
        cache_time = time.time() - cache_start
        
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Caption retrieval took {total_time:.3f}s (file check: {file_time:.3f}s, cache lookup: {cache_time:.3f}s)")
        
        return CaptionResponse(
            imageId=image_id,
            caption=caption
        )
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Error in caption retrieval after {total_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/caption/{image_id}")
async def delete_caption(request: Request, image_id: str):
    """Delete the caption for an image."""
    start_time = time.time()
    try:
        # Check if image exists
        file_start = time.time()
        image_path = Path(IMAGES_DIR) / f"{image_id}.jpg"
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        file_time = time.time() - file_start
        
        # Remove caption from cache
        cache_start = time.time()
        if image_id in caption_cache:
            del caption_cache[image_id]
        cache_time = time.time() - cache_start
        
        # Save caption cache to disk
        save_start = time.time()
        save_caption_cache()
        save_time = time.time() - save_start
        
        # Update photoset metadata if it exists
        metadata_start = time.time()
        if image_id in PHOTOSET_METADATA_CACHE:
            PHOTOSET_METADATA_CACHE[image_id].pop('caption', None)
        metadata_time = time.time() - metadata_start
        
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Caption deletion took {total_time:.3f}s (file check: {file_time:.3f}s, cache update: {cache_time:.3f}s, save: {save_time:.3f}s, metadata: {metadata_time:.3f}s)")
        
        return {"message": "Caption deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Error in caption deletion after {total_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
