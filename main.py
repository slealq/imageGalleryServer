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

# In-memory storage for captions
image_captions: Dict[str, str] = {}

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

class CropBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class CropRequest(BaseModel):
    imageId: str
    targetSize: int
    cropBox: CropBox

# Configuration
IMAGES_DIR = Path("/Users/stuartleal/Library/Mobile Documents/com~apple~CloudDocs/Downloads/4800watermarked")
#IMAGES_DIR = Path("/Users/stuartleal/gallery-project/images")
IMAGES_PER_PAGE = 10

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
        # Get all image files
        image_files = [f for f in IMAGES_DIR.glob("*") if f.is_file()]
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
                has_crop=False  # Dummy data for now
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
    
def get_cropped_image(image: PILImage, target_size: int):
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
            resized = get_cropped_image(img, target_size)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            resized.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return Response(content=img_byte_arr.getvalue(), media_type="image/png")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{image_id}/crop")
async def crop_image(image_id: str, crop_request: CropRequest):
    """
    Crop an image according to the specified crop box and target size.
    The crop box coordinates are given for the scaled down image.
    """
    try:
        image_path = get_image_path(image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail="Image not found")

        with PILImage.open(image_path) as img:
            # First resize the image to fit within target size
            resized = get_cropped_image(img, crop_request.targetSize)
            resized_width, resized_height = resized.size
            
            # Ensure crop box is within resized image bounds
            crop_x = max(0, min(crop_request.cropBox.x, resized_width - 1))
            crop_y = max(0, min(crop_request.cropBox.y, resized_height - 1))
            crop_width = min(crop_request.cropBox.width, resized_width - crop_x)
            crop_height = min(crop_request.cropBox.height, resized_height - crop_y)

            print(f"Crop properties:")
            print(f"  x: {crop_x}")
            print(f"  y: {crop_y}")
            print(f"  width: {crop_width}")
            print(f"  height: {crop_height}")
            print(f"  resized dimensions: {resized_width}x{resized_height}")
            
            # Perform the crop on the resized image
            cropped = resized.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            cropped.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return Response(content=img_byte_arr.getvalue(), media_type="image/png")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.68.59", port=4322)
