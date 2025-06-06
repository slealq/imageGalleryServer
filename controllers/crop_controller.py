from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
import time
from PIL import Image as PILImage
import io

from ..models.models import CropRequest
from ..services.image_service import get_image_path, generate_cropped_image
from ..services.cache_service import image_cache
from config import IMAGES_DIR

router = APIRouter()

@router.post("/crop")
async def crop_image(request: Request, crop_request: CropRequest):
    """Crop an image based on normalized deltas."""
    start_time = time.time()
    try:
        # Get image path
        file_start = time.time()
        image_path = get_image_path(crop_request.imageId, IMAGES_DIR)
        if not image_path:
            raise HTTPException(status_code=404, detail="Image not found")
        file_time = time.time() - file_start
        
        # Open and process image
        process_start = time.time()
        with PILImage.open(image_path) as img:
            open_time = time.time() - process_start
            
            # Convert to RGB if needed
            convert_start = time.time()
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            convert_time = time.time() - convert_start
            
            # Generate cropped image
            crop_start = time.time()
            resized = generate_cropped_image(img, crop_request.targetSize)
            crop_time = time.time() - crop_start
            
            # Convert to bytes
            optimize_start = time.time()
            img_byte_arr = io.BytesIO()
            resized.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            img_byte_arr.seek(0)
            optimize_time = time.time() - optimize_start
            
            # Add to cache
            cache_start = time.time()
            image_data = img_byte_arr.getvalue()
            cache_key = f"{crop_request.imageId}_crop_{crop_request.targetSize}"
            image_cache.put(cache_key, image_data)
            cache_time = time.time() - cache_start
            
            # Add performance headers
            headers = {
                "X-File-Lookup-Time": f"{file_time:.3f}",
                "X-Image-Open-Time": f"{open_time:.3f}",
                "X-Image-Convert-Time": f"{convert_time:.3f}",
                "X-Crop-Time": f"{crop_time:.3f}",
                "X-Optimize-Time": f"{optimize_time:.3f}",
                "X-Cache-Store-Time": f"{cache_time:.3f}"
            }
            
            total_time = time.time() - start_time
            if total_time > 1.0:  # Log if total time exceeds 1 second
                print(f"Performance: Crop operation took {total_time:.3f}s (file lookup: {file_time:.3f}s, open: {open_time:.3f}s, convert: {convert_time:.3f}s, crop: {crop_time:.3f}s, optimize: {optimize_time:.3f}s, cache store: {cache_time:.3f}s)")
            
            return Response(
                content=image_data,
                media_type="image/jpeg",
                headers=headers
            )
            
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Error in crop operation after {total_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
