from fastapi import APIRouter, HTTPException
from PIL import Image as PILImage
from models.models import CaptionRequest, CaptionResponse
from config import IMAGES_DIR
from caption_generator import get_caption_generator
from filter_manager import filter_manager
import json
from fastapi.responses import StreamingResponse
import os

router = APIRouter(prefix="/images", tags=["captions"])

# Initialize caption generator
caption_generator = get_caption_generator()

def get_image_path(image_id: str) -> str:
    """Get the full path of an image file by its ID."""
    image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
    return str(image_files[0]) if image_files else None

@router.post("/{image_id}/caption")
async def save_caption(image_id: str, caption_request: CaptionRequest):
    """
    Save a caption for an image
    """
    try:
        print(f"Attempting to save caption for image {image_id}")
        print(f"Caption request: {caption_request}")
        
        if not caption_request.caption:
            print("No caption provided in request")
            raise HTTPException(status_code=400, detail="No caption provided")
        
        # Verify image exists
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            print(f"Image not found: {image_id}")
            raise HTTPException(status_code=404, detail="Image not found")
        
        print(f"Found image file: {image_files[0]}")
        
        # Save caption to file
        caption_path = os.path.join(IMAGES_DIR, f"{image_id}_caption.txt")
        print(f"Saving caption to: {caption_path}")
        
        try:
            with open(caption_path, 'w') as f:
                f.write(caption_request.caption)
            print("Caption file written successfully")
        except Exception as write_error:
            print(f"Error writing caption file: {str(write_error)}")
            raise
        
        # Update in-memory cache
        try:
            filter_manager.image_captions[image_id] = caption_request.caption
            print("In-memory cache updated successfully")
        except Exception as cache_error:
            print(f"Error updating in-memory cache: {str(cache_error)}")
            raise
        
        return {"message": "Caption saved successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in save_caption: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{image_id}/caption", response_model=CaptionResponse)
async def get_caption(image_id: str):
    """
    Get the caption for an image
    """
    try:
        # Verify image exists
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Get caption
        caption = filter_manager.image_captions.get(image_id)
        if not caption:
            raise HTTPException(status_code=404, detail="No caption found for this image")
        
        return CaptionResponse(caption=caption)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{image_id}/generate-caption")
async def generate_caption(image_id: str, request: CaptionRequest):
    """
    Generate a caption for an image using the caption generator
    """
    try:
        # Get the image path from the image ID
        image_path = get_image_path(image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail="Image not found")

        # Load the image
        image = PILImage.open(image_path)
        
        # Generate caption using the caption generator
        caption = await caption_generator.generate_caption(image, request.prompt)
        
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{image_id}/stream-caption")
async def stream_caption(image_id: str, request: CaptionRequest):
    """
    Stream the caption generation process for an image
    """
    try:
        # Get the image path from the image ID
        image_path = get_image_path(image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail="Image not found")

        # Load the image
        image = PILImage.open(image_path)
        
        async def generate():
            try:
                async for chunk in caption_generator.stream_caption(image, request.prompt):
                    if chunk:  # Only send non-empty chunks
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            except Exception as e:
                # Send error as SSE
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                raise
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable proxy buffering
            }
        )
    except Exception as e:
        print(f"Error in stream_caption endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 