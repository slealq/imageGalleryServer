from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
import json
from pathlib import Path
from fastapi.responses import FileResponse
from PIL import Image as PILImage

app = FastAPI(title="Image Service API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ImageMetadata(BaseModel):
    id: str
    filename: str
    size: int
    created_at: datetime
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None

class ImageResponse(BaseModel):
    images: List[ImageMetadata]
    total: int
    page: int
    page_size: int
    total_pages: int

# Configuration
IMAGES_DIR = Path("/Users/stuartleal/Library/Mobile Documents/com~apple~CloudDocs/Downloads/4800watermarked")
IMAGES_PER_PAGE = 10

# Ensure images directory exists
IMAGES_DIR.mkdir(exist_ok=True)

def get_image_dimensions(image_path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        with PILImage.open(image_path) as img:
            return img.size
    except Exception:
        return None, None

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
            metadata = ImageMetadata(
                id=str(img_file.stem),
                filename=img_file.name,
                size=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                mime_type=f"image/{img_file.suffix[1:].lower()}" if img_file.suffix else "application/octet-stream",
                width=width,
                height=height
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
