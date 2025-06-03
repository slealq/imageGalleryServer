from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from datetime import datetime
import json
from pathlib import Path
from fastapi.responses import FileResponse, Response
from PIL import Image as PILImage
import asyncio
import io

app = FastAPI(title="Image Service API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_DIR = Path("/Users/stuartleal/gallery-project/images")
IMAGES_PER_PAGE = 10

# In-memory storage for captions
image_captions: Dict[str, str] = {}
image_crops: Dict[str, dict] = {}  # Store crop info: {imageId: {"targetSize": int, "normalizedDeltas": {"x": float, "y": float}}}

def initialize_crop_cache():
    """
    Scan the images directory for existing cropped images and populate the cache.
    Cropped images follow the pattern: {imageId}_crop_{targetSize}.png
    Metadata is stored in {imageId}_crop_{targetSize}.json
    """
    for file in IMAGES_DIR.glob("*_crop_*.png"):
        try:
            # Extract imageId and targetSize from filename
            # Example: "abc123_crop_512.png" -> imageId="abc123", targetSize=512
            parts = file.stem.split('_crop_')
            if len(parts) != 2:
                continue
                
            image_id = parts[0]
            target_size = int(parts[1])
            
            # Try to load metadata from JSON file
            metadata_path = os.path.join(IMAGES_DIR, f"{image_id}_crop_{target_size}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    image_crops[image_id] = {
                        "targetSize": metadata["targetSize"],
                        "normalizedDeltas": metadata["normalizedDeltas"]
                    }
            else:
                # If no metadata file exists, use default values
                image_crops[image_id] = {
                    "targetSize": target_size,
                    "normalizedDeltas": {
                        "x": 0,
                        "y": 0
                    }
                }
        except (ValueError, IndexError, json.JSONDecodeError) as e:
            print(f"Error processing cropped image {file}: {e}")
            continue

# Initialize crop cache on startup
initialize_crop_cache()

# Models
class ImageMetadata(BaseModel):
    id: str
    filename: str
    size: int
    created_at: datetime
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    has_caption: bool = False
    collection_name: str = "Default Collection"
    has_tags: bool = False
    has_crop: bool = False

class ImageResponse(BaseModel):
    images: List[ImageMetadata]
    total: int
    page: int
    page_size: int
    total_pages: int

class CaptionRequest(BaseModel):
    caption: str

class CaptionResponse(BaseModel):
    caption: str

class NormalizedDeltas(BaseModel):
    x: float
    y: float

class CropRequest(BaseModel):
    imageId: str
    targetSize: int
    normalizedDeltas: NormalizedDeltas

# Configuration
# IMAGES_DIR = Path("/Users/stuartleal/Library/Mobile Documents/com~apple~CloudDocs/Downloads/4800watermarked")


# Ensure images directory exists
IMAGES_DIR.mkdir(exist_ok=True)

def get_image_dimensions(image_path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        with PILImage.open(image_path) as img:
            return img.size
    except Exception:
        return None, None

def get_image_path(image_id: str) -> Optional[Path]:
    """Get the full path of an image file by its ID."""
    image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
    return image_files[0] if image_files else None

@app.get("/images", response_model=ImageResponse)
async def get_images(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(IMAGES_PER_PAGE, ge=1, le=100, description="Number of images per page")
):
    """
    Get paginated list of images with their metadata
    """
    try:
        # Get all image files, excluding cropped images
        image_files = [f for f in IMAGES_DIR.glob("*") if f.is_file() and "_crop_" not in f.name]
        image_files.sort()
        total_images = len(image_files)
        
        # Calculate pagination
        total_pages = (total_images + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_images)
        
        # Get images for current page
        page_images = image_files[start_idx:end_idx]
        
        # Create metadata for each image
        images_metadata = []
        for img_file in page_images:
            stat = img_file.stat()
            width, height = get_image_dimensions(img_file)
            image_id = str(img_file.stem)
            metadata = ImageMetadata(
                id=image_id,
                filename=img_file.name,
                size=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                mime_type=f"image/{img_file.suffix[1:].lower()}" if img_file.suffix else "application/octet-stream",
                width=width,
                height=height,
                has_caption=image_id in image_captions,
                collection_name="Default Collection",  # Dummy data for now
                has_tags=False,  # Dummy data for now
                has_crop=image_id in image_crops
            )
            images_metadata.append(metadata)
        
        return ImageResponse(
            images=images_metadata,
            total=total_images,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{image_id}")
async def get_image(image_id: str):
    """
    Get a specific image by ID
    """
    try:
        # Find the image file
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_file = image_files[0]
        return FileResponse(image_file)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{image_id}/caption")
async def save_caption(image_id: str, caption_request: CaptionRequest):
    """
    Save a caption for an image
    """
    try:
        # Verify image exists
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Save caption
        image_captions[image_id] = caption_request.caption
        return {"message": "Caption saved successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{image_id}/caption", response_model=CaptionResponse)
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
        caption = image_captions.get(image_id)
        if not caption:
            raise HTTPException(status_code=404, detail="No caption found for this image")
        
        return CaptionResponse(caption=caption)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{image_id}/generate-caption", response_model=CaptionResponse)
async def generate_caption(image_id: str):
    """
    Generate a caption for an image (dummy implementation)
    """
    try:
        # Verify image exists
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Simulate processing time
        await asyncio.sleep(5)
        
        # Generate dummy caption
        dummy_caption = f"This is a generated caption for image {image_id}"
        return CaptionResponse(caption=dummy_caption)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def generate_cropped_image(image: PILImage, target_size: int):
    # Calculate scaling factor to fit within target size
    width, height = image.size
    scale = max(target_size / width, target_size / height)
    new_size = (int(width * scale), int(height * scale))
    
    # Resize the image
    resized = image.resize(new_size, PILImage.Resampling.LANCZOS)

    return resized

@app.get("/images/{image_id}/preview/{target_size}")
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

@app.get("/images/{image_id}/crop")
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
        crop_info = image_crops.get(image_id)
        if not crop_info:
            raise HTTPException(status_code=404, detail="No crop found for this image")
        
        # Get the cropped image
        cropped_image_path = os.path.join(IMAGES_DIR, f"{image_id}_{crop_info['targetSize']}.png")
        if not os.path.exists(cropped_image_path):
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

@app.get("/images/{image_id}/cropped")
async def get_cropped_image(image_id: str):
    """
    Get the cropped image file
    """
    try:
        # Get crop info to determine the file path
        crop_info = image_crops.get(image_id)
        if not crop_info:
            raise HTTPException(status_code=404, detail="No crop found for this image")
        
        # Get the cropped image path with _crop_ in the filename
        cropped_image_path = os.path.join(IMAGES_DIR, f"{image_id}_crop_{crop_info['targetSize']}.png")
        if not os.path.exists(cropped_image_path):
            raise HTTPException(status_code=404, detail="Cropped image file not found")
        
        return FileResponse(cropped_image_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{image_id}/crop")
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
            
            targetX = crop_request.normalizedDeltas.x * resized.width;
            targetY = crop_request.normalizedDeltas.y * resized.height;

            # Calculate slack (extra space) in both dimensions
            horizontal_slack = resized.width - crop_request.targetSize
            vertical_slack = resized.height - crop_request.targetSize
            
            # Add half of the slack to center the crop
            targetX += horizontal_slack / 2
            targetY += vertical_slack / 2

            # Generate the square sizes
            crop_width = crop_height = crop_request.targetSize;
            
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

            # Save the cropped image with _crop_ in the filename
            save_path = os.path.join(IMAGES_DIR, f"{image_id}_crop_{crop_request.targetSize}.png")
            cropped.save(save_path, format='PNG')
            
            # Create crop metadata
            crop_metadata = {
                "targetSize": crop_request.targetSize,
                "normalizedDeltas": {
                    "x": crop_request.normalizedDeltas.x,
                    "y": crop_request.normalizedDeltas.y
                }
            }
            
            # Save metadata to JSON file
            metadata_path = os.path.join(IMAGES_DIR, f"{image_id}_crop_{crop_request.targetSize}.json")
            with open(metadata_path, 'w') as f:
                json.dump(crop_metadata, f, indent=2)
            
            # Store crop information in memory
            image_crops[image_id] = crop_metadata
            
            return Response(content=img_byte_arr.getvalue(), media_type="image/png")
            
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.68.53", port=4322)
