from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from PIL import Image as PILImage
import io
import os
import json
from pathlib import Path
from typing import Optional, Dict
from config import IMAGES_DIR
from caches.crop_cache import crop_cache

router = APIRouter(prefix="/images", tags=["crop"])

class NormalizedDeltas(BaseModel):
    x: float
    y: float

class CropRequest(BaseModel):
    imageId: str
    targetSize: int
    normalizedDeltas: NormalizedDeltas

def get_image_path(image_id: str) -> Optional[Path]:
    """Get the full path of an image file by its ID."""
    image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
    return image_files[0] if image_files else None

def generate_cropped_image(image: PILImage, target_size: int):
    # Calculate scaling factor to fit within target size
    width, height = image.size
    scale = max(target_size / width, target_size / height)
    new_size = (int(width * scale), int(height * scale))
    
    # Resize the image
    resized = image.resize(new_size, PILImage.Resampling.LANCZOS)

    return resized

@router.get("/{image_id}/crop")
async def get_crop(image_id: str):
    """
    Get crop information and cropped image for an image
    """
    try:
        # Verify image exists
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Get crop info
        crop_info = crop_cache.get_crop_metadata(image_id)
        if not crop_info:
            raise HTTPException(status_code=404, detail="No crop found for this image")
        
        # Get the cropped image
        cropped_image = crop_cache.get_crop_image(image_id)
        if not cropped_image:
            raise HTTPException(status_code=404, detail="Cropped image file not found")
        
        # Return both crop info and the cropped image
        return {
            "cropInfo": crop_info,
            "imageUrl": f"/images/{image_id}/cropped"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{image_id}/cropped")
async def get_cropped_image(image_id: str):
    """
    Get the cropped image file
    """
    try:
        # Get the cropped image data
        cropped_image = crop_cache.get_crop_image(image_id)
        if not cropped_image:
            raise HTTPException(status_code=404, detail="Cropped image not found")
        
        return Response(content=cropped_image, media_type="image/png")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{image_id}/crop")
async def crop_image(image_id: str, crop_request: CropRequest):
    """
    Crop an image according to the specified deltas and target size.
    """
    try:
        image_path = get_image_path(image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail="Image not found")

        with PILImage.open(image_path) as img:
            # First resize the image to fit within target size
            resized = generate_cropped_image(img, crop_request.targetSize)
            
            targetX = crop_request.normalizedDeltas.x * resized.width
            targetY = crop_request.normalizedDeltas.y * resized.height

            # Calculate slack (extra space) in both dimensions
            horizontal_slack = resized.width - crop_request.targetSize
            vertical_slack = resized.height - crop_request.targetSize
            
            # Add half of the slack to center the crop
            targetX += horizontal_slack / 2
            targetY += vertical_slack / 2

            # Generate the square sizes
            crop_width = crop_height = crop_request.targetSize
            
            print(f"Crop properties:")
            print(f"  x: {targetX}")
            print(f"  y: {targetY}")
            print(f"  width: {crop_width}")
            print(f"  height: {crop_height}")
            
            # Perform the crop on the resized image
            cropped = resized.crop((targetX, targetY, targetX + crop_width, targetY + crop_height))
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            cropped.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            image_data = img_byte_arr.getvalue()

            # Create crop metadata
            crop_metadata = {
                "targetSize": crop_request.targetSize,
                "normalizedDeltas": {
                    "x": crop_request.normalizedDeltas.x,
                    "y": crop_request.normalizedDeltas.y
                }
            }
            
            # Save both metadata and image using the cache
            if not crop_cache.save_crop(image_id, crop_metadata, image_data):
                raise HTTPException(status_code=500, detail="Failed to save crop data")
            
            return Response(content=image_data, media_type="image/png")
            
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{image_id}/preview/{target_size}")
async def get_image_preview(image_id: str, target_size: int):
    """
    Get a scaled preview of the image that fits within the target size while maintaining aspect ratio.
    The image will be scaled down so that the smaller size fits the target, but cropping will be required to make it a square
    """
    try:
        image_path = get_image_path(image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail="Image not found")

        with PILImage.open(image_path) as img:
            resized = generate_cropped_image(img, target_size)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            resized.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return Response(content=img_byte_arr.getvalue(), media_type="image/png")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 