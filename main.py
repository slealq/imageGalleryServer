from caches.crop_cache import crop_cache
from caption_generator import get_caption_generator
from collections import OrderedDict
from config import IMAGES_DIR, IMAGES_PER_PAGE, SERVER_HOST, SERVER_PORT, PROFILING_ENABLED, PROFILING_DIR, PHOTOSET_METADATA_DIRECTORY # Import profiling config
from controllers.CropController import router as crop_router
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from filter_manager import filter_manager
from functools import partial
from io import BytesIO
from pathlib import Path
from PIL import Image as PILImage
from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.types import ASGIApp
from starlette.types import Scope, Receive, Send
from starlette.websockets import WebSocket
from typing import List, Optional, Dict, OrderedDict
import asyncio
import concurrent.futures
import io
import json
import os
import time # Import time module
import uuid
import uvicorn
import zipfile


class RequestTimingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate timing
        process_time = time.time() - start_time
        
        # Add timing and request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        
        # Log timing
        print(f"Request {request_id}: {request.method} {request.url.path} completed in {process_time:.3f}s")
        
        return response

app = FastAPI(title="Image Service API")

# Add request timing middleware
app.add_middleware(RequestTimingMiddleware)

# Include routers
app.include_router(crop_router)

# Initialize caption generator
caption_generator = get_caption_generator()

# Image cache implementation
class ImageCache:
    def __init__(self, max_size_bytes: int = 10 * 1024 * 1024 * 1024):  # 10GB default
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.cache: OrderedDict[str, bytes] = OrderedDict()  # LRU cache
        self.locks: Dict[str, asyncio.Lock] = {}  # Per-image locks
    
    def _get_lock(self, image_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific image ID."""
        if image_id not in self.locks:
            self.locks[image_id] = asyncio.Lock()
        return self.locks[image_id]
    
    async def get(self, image_id: str) -> Optional[bytes]:
        print(f"Getting image from cache {image_id}")
        print(f"Cache size: {len(self.cache)} ")

        async with self._get_lock(image_id):
            if image_id in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(image_id)
                self.cache[image_id] = value
                return value
            return None
    
    async def put(self, image_id: str, image_data: bytes):
        print(f"Putting image in cache {image_id}")
        print(f"Cache size: {len(self.cache)} ")

        async with self._get_lock(image_id):
            # If key exists, remove it first to update size
            if image_id in self.cache:
                self.current_size_bytes -= len(self.cache[image_id])
                self.cache.pop(image_id)
            
            # Evict items if needed
            while self.current_size_bytes + len(image_data) > self.max_size_bytes and self.cache:
                # Remove least recently used item
                _, removed_data = self.cache.popitem(last=False)
                self.current_size_bytes -= len(removed_data)
            
            # Add new item
            self.cache[image_id] = image_data
            self.current_size_bytes += len(image_data)
    
    def clear(self):
        self.cache.clear()
        self.current_size_bytes = 0
        self.locks.clear()  # Clear all locks when cache is cleared

# Initialize image cache
image_cache = ImageCache()

# Create a thread pool for CPU-bound operations
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def process_image_in_thread(image_path: Path) -> Optional[bytes]:
    """Process an image in a separate thread."""
    try:
        with PILImage.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Create a BytesIO object to store the optimized image
            output = io.BytesIO()
            # Use faster optimization settings
            img.save(output, format='JPEG', quality=85, optimize=False)
            output.seek(0)
            return output.getvalue()
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

async def process_image_for_cache(image_id: str) -> tuple[bool, bool, bool]:
    """
    Process an image for caching.
    
    Args:
        image_id: ID of the image to process
        
    Returns:
        Tuple of (success, was_skipped, was_error)
    """
    # Skip if already in cache
    cached_image = await image_cache.get(image_id)
    if cached_image:
        print(f"Image {image_id} already in cache")
        return False, True, False
    
    try:
        # Get image path
        image_path = get_image_path(image_id)
        if not image_path:
            return False, False, True
        
        # Process image in thread pool
        loop = asyncio.get_event_loop()
        image_data = await loop.run_in_executor(
            thread_pool,
            partial(process_image_in_thread, image_path)
        )
        
        if not image_data:
            return False, False, True
            
        # Add to cache
        await image_cache.put(image_id, image_data)
        return True, False, False
            
    except Exception as e:
        print(f"Error processing image {image_id}: {str(e)}")
        return False, False, True

def get_image_path(image_id: str) -> Optional[Path]:
    """Get the full path of an image file by its ID."""
    # This function is correct as is
    image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
    return image_files[0] if image_files else None

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
    year: Optional[str] = None
    tags: List[str] = []
    actors: List[str] = []

class ImageResponse(BaseModel):
    images: List[ImageMetadata]
    total: int
    page: int
    page_size: int
    total_pages: int

class CaptionRequest(BaseModel):
    prompt: Optional[str] = None
    caption: Optional[str] = None

class CaptionResponse(BaseModel):
    caption: str

class TagResponse(BaseModel):
    tags: List[str]

class AddTagRequest(BaseModel):
    tag: str

class ExportRequest(BaseModel):
    imageIds: List[str]

@app.get("/filters", response_model=Dict[str, List[str]])
async def get_available_filters():
    """
    Get all available filters (actors, tags, years) that can be used to filter images.
    """

    return filter_manager.get_available_filters()

@app.on_event("startup")
async def startup_event():
    """Initialize caches and warm up image cache on startup"""
    filter_manager.initialize()

@app.on_event("shutdown")
async def save_all_caches():
    """Saves all caches to files on shutdown."""
    filter_manager.save_all_caches()

def create_image_metadata(img_file: Path) -> ImageMetadata:
    """
    Create metadata for a single image.
    
    Args:
        img_file: Path to the image file
        
    Returns:
        ImageMetadata object containing the image's metadata
    """
    stat = img_file.stat()
    image_id = str(img_file.stem)
    width, height = filter_manager.image_dimensions.get(image_id, (None, None))

    image_filter_metadata = filter_manager.get_image_filter_metadata(image_id)
    
    return ImageMetadata(
        id=image_id,
        filename=img_file.name,
        size=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_ctime),
        mime_type=f"image/{img_file.suffix[1:].lower()}" if img_file.suffix else "application/octet-stream",
        width=width,
        height=height,
        has_caption=image_id in filter_manager.image_captions,
        collection_name="Default Collection",
        has_tags=len(image_filter_metadata['tags']) > 0,
        has_crop=crop_cache.has_crop_metadata(image_id),
        year=image_filter_metadata['year'],
        tags=image_filter_metadata['tags'],
        actors=image_filter_metadata['actors']
    )

def calculate_pagination(
    total_items: int,
    page: int,
    page_size: int
) -> tuple[int, int, int, int]:
    """
    Calculate pagination parameters.
    
    Args:
        total_items: Total number of items
        page: Current page number
        page_size: Number of items per page
        
    Returns:
        Tuple of (total_pages, start_idx, end_idx, total_items)
    """
    total_pages = (total_items + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_items)
    return total_pages, start_idx, end_idx, total_items

def apply_image_filters(
    images: List[Path],
    actor: Optional[str] = None,
    tag: Optional[str] = None,
    year: Optional[str] = None,
    has_caption: Optional[bool] = None,
    has_crop: Optional[bool] = None
) -> List[Path]:
    """Apply filters to a list of images based on the provided criteria."""
    return filter_manager.apply_filters(
        images,
        actor=actor,
        tag=tag,
        year=year,
        has_caption=has_caption,
        has_crop=has_crop
    )

@app.get("/images", response_model=ImageResponse)
async def get_images(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(IMAGES_PER_PAGE, ge=1, le=100, description="Number of images per page"),
    actor: Optional[str] = Query(None, description="Filter by actor name"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    year: Optional[str] = Query(None, description="Filter by year"),
    has_caption: Optional[bool] = Query(None, description="Filter for images with captions"),
    has_crop: Optional[bool] = Query(None, description="Filter for images with crops")
):
    """
    Get a paginated list of images with optional filtering.
    Also triggers cache warming for next pages.
    """
    start_time = time.time()
    
    # Apply filters if provided
    filter_start = time.time()
    filtered_images = apply_image_filters(
        filter_manager.cached_image_files,
        actor=actor,
        tag=tag,
        year=year,
        has_caption=has_caption,
        has_crop=has_crop
    )
    filter_time = time.time() - filter_start
    
    if filter_time > 1.0:
        print(f"Performance: Filtering took {filter_time:.3f}s")
    
    # Calculate pagination
    total_pages, start_idx, end_idx, total_images = calculate_pagination(
        len(filtered_images),
        page,
        page_size
    )
    
    # Get images for current page
    page_images = filtered_images[start_idx:end_idx]
    
    # Create metadata for each image
    metadata_start = time.time()
    images_metadata = [create_image_metadata(img_file) for img_file in page_images]
    metadata_time = time.time() - metadata_start
    
    total_time = time.time() - start_time
    if total_time > 1.0:
        print(f"Performance: Page {page} processed in {total_time:.3f}s (metadata: {metadata_time:.3f}s)")
    
    # Trigger predictive cache warming for unfiltered requests
    # if not (actor or tag or year or has_caption is not None or has_crop is not None):
        # asyncio.create_task(predict_and_warm_cache(page))
    
    response = ImageResponse(
        images=images_metadata,
        total=total_images,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )
    
    return response

@app.get("/images/batch", response_model=List[ImageResponse])
async def get_images_batch(
    request: Request,
    start_page: int = Query(1, ge=1, description="Starting page number"),
    num_pages: int = Query(3, ge=1, le=5, description="Number of pages to fetch"),
    page_size: int = Query(IMAGES_PER_PAGE, ge=1, le=100, description="Number of images per page"),
    actor: Optional[str] = Query(None, description="Filter by actor name"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    year: Optional[str] = Query(None, description="Filter by year"),
    has_caption: Optional[bool] = Query(None, description="Filter for images with captions"),
    has_crop: Optional[bool] = Query(None, description="Filter for images with crops")
):
    """
    Get multiple pages of images at once for efficient preloading.
    """
    responses = []
    for page in range(start_page, start_page + num_pages):
        response = await get_images(
            request=request,
            page=page,
            page_size=page_size,
            actor=actor,
            tag=tag,
            year=year,
            has_caption=has_caption,
            has_crop=has_crop
        )
        responses.append(response)
        if not response.images:  # Stop if we hit the end
            break
    return responses

@app.get("/images/{image_id}")
async def get_image(image_id: str, request: Request):
    """
    Get a specific image by ID from cache or file
    """
    start_time = time.time()
    incoming_latency = None
    
    # Get request start timestamp from header
    request_start = request.headers.get("X-Request-Start-Timestamp")
    if request_start:
        try:
            request_start = float(request_start)
            incoming_latency = start_time - request_start
            print(f"Request {request.state.request_id}: Incoming network latency: {incoming_latency:.3f}s")
        except (ValueError, TypeError):
            print(f"Request {request.state.request_id}: Invalid request-start-timestamp header")
    
    try:
        # Try to get from cache first
        cache_start = time.time()
        cached_image = await image_cache.get(image_id)
        cache_time = time.time() - cache_start
        
        if cached_image:
            # Add caching headers
            headers = {
                "Cache-Control": "public, max-age=31536000",  # Cache for 1 year
                "ETag": f'"{hash(cached_image)}"',  # Use hash of image data as ETag
                "X-Cache-Status": "HIT",  # Add cache status header
                "X-Cache-Lookup-Time": f"{cache_time:.3f}",  # Add cache lookup time
                "X-Response-Start-Timestamp": f"{time.time()}",  # Add response start timestamp
                "X-Incoming-Latency": f"{incoming_latency:.3f}" if incoming_latency is not None else "N/A"  # Add incoming latency
            }
            
            # Check if client has cached version
            if_none_match = request.headers.get("if-none-match")
            if if_none_match and if_none_match == headers["ETag"]:
                print(f"Request {request.state.request_id}: Image {image_id} - Client cache hit (304)")
                return Response(status_code=304, headers={"X-Cache-Status": "CLIENT_HIT"})  # Not Modified
            
            total_time = time.time() - start_time
            if total_time > 1.0:  # Log if total time exceeds 1 second
                print(f"Performance: Image {image_id} served from cache in {total_time:.3f}s (cache lookup: {cache_time:.3f}s)")
            
            print(f"Request {request.state.request_id}: Image {image_id} - Server cache hit")
            return Response(
                content=cached_image,
                media_type="image/jpeg",
                headers=headers
            )
        
        print(f"Request {request.state.request_id}: Image {image_id} - Cache miss")
        
        # If not in cache, get from file
        file_start = time.time()
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_file = image_files[0]
        file_time = time.time() - file_start
        
        # Open and optimize the image
        process_start = time.time()
        with PILImage.open(image_file) as img:
            open_time = time.time() - process_start
            
            convert_start = time.time()
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            convert_time = time.time() - convert_start
            
            # Create a BytesIO object to store the optimized image
            optimize_start = time.time()
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            output.seek(0)
            optimize_time = time.time() - optimize_start
            
            # Add to cache
            cache_start = time.time()
            image_data = output.getvalue()
            await image_cache.put(image_id, image_data)
            cache_time = time.time() - cache_start
            
            # Add caching headers
            headers = {
                "Cache-Control": "public, max-age=31536000",
                "ETag": f'"{hash(image_data)}"',
                "X-Cache-Status": "MISS",  # Add cache status header
                "X-File-Lookup-Time": f"{file_time:.3f}",
                "X-Image-Open-Time": f"{open_time:.3f}",
                "X-Image-Convert-Time": f"{convert_time:.3f}",
                "X-Image-Optimize-Time": f"{optimize_time:.3f}",
                "X-Cache-Store-Time": f"{cache_time:.3f}",
                "response-start-timestamp": f"{time.time()}",  # Add response start timestamp
                "X-Incoming-Latency": f"{incoming_latency:.3f}" if incoming_latency is not None else "N/A"  # Add incoming latency
            }
            
            process_time = time.time() - process_start
            total_time = time.time() - start_time
            
            if total_time > 1.0:  # Log if total time exceeds 1 second
                print(f"Performance: Image {image_id} processed in {total_time:.3f}s (file lookup: {file_time:.3f}s, open: {open_time:.3f}s, convert: {convert_time:.3f}s, optimize: {optimize_time:.3f}s, cache store: {cache_time:.3f}s)")
            
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
            print(f"Performance: Error processing image {image_id} after {total_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{image_id}/caption")
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
        caption = filter_manager.image_captions.get(image_id)
        if not caption:
            raise HTTPException(status_code=404, detail="No caption found for this image")
        
        return CaptionResponse(caption=caption)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{image_id}/tags", response_model=TagResponse)
async def get_image_tags(image_id: str):
    """
    Get all tags for an image (both scene and image-specific tags)
    """
    try:
        # Verify image exists
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Get all tags for the image
        tags = filter_manager.get_tags_for_image(image_id)
        
        return TagResponse(tags=tags)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{image_id}/tags")
async def add_image_tag(image_id: str, request: AddTagRequest):
    """
    Add a tag to an image
    """
    try:
        # Verify image exists
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Add the tag
        filter_manager.set_tags_in_file_for_image(image_id, [request.tag])
        
        return {"message": f"Tag '{request.tag}' added successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{image_id}/generate-caption")
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

@app.post("/api/stream-caption/{image_id}")
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

@app.post("/api/export-images")
async def export_images(request: ExportRequest):
    try:
        print(f"Starting export for {len(request.imageIds)} images: {request.imageIds}")
        
        # Create a BytesIO object to store the zip file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for image_id in request.imageIds:
                print(f"\nProcessing image: {image_id}")
                
                # Get image filename without extension
                image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
                if not image_files:
                    print(f"  Warning: No image file found for {image_id}")
                    continue
                
                image_path = image_files[0]
                base_name = image_path.stem
                print(f"  Base name: {base_name}")
                
                # Add cropped image if it exists
                if crop_cache.has_crop_metadata(image_id):
                    crop_path = crop_cache.get_crop_image_path(image_id)
                    print(f"  Checking for crop at: {crop_path}")

                    cropped_image_bytes = crop_cache.get_crop_image(image_id)
                    
                    if cropped_image_bytes:
                        print(f"  Adding cropped image to zip")

                        zip_file.writestr(f"{base_name}.png", cropped_image_bytes)
                    else:
                        print(f"  Warning: Crop file not found at {crop_path}")
                else:
                    print(f"  No crop info found for {image_id}")
                
                # Add caption if it exists
                caption_path = os.path.join(IMAGES_DIR, f"{image_id}_caption.txt")
                print(f"  Checking for caption at: {caption_path}")
                
                if os.path.exists(caption_path):
                    print(f"  Adding caption to zip")
                    with open(caption_path, 'r') as f:
                        # Use the same base name for both crop and caption
                        zip_file.writestr(f"{base_name}.txt", f.read())
                else:
                    print(f"  No caption file found for {image_id}")
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        # Get the size of the zip file
        zip_size = zip_buffer.getbuffer().nbytes
        print(f"\nZip file created. Size: {zip_size} bytes")
        
        if zip_size == 0:
            print("Warning: Created zip file is empty!")
        
        # Return the zip file
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=exported_images.zip"
            }
        )
    except Exception as e:
        print(f"Error in export_images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/warmup")
async def warmup_cache(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(IMAGES_PER_PAGE, ge=1, le=100, description="Number of images per page"),
    actor: Optional[str] = Query(None, description="Filter by actor name"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    year: Optional[str] = Query(None, description="Filter by year"),
    has_caption: Optional[bool] = Query(None, description="Filter for images with captions"),
    has_crop: Optional[bool] = Query(None, description="Filter for images with crops")
):
    """
    Pre-warm the image cache for a specific page of images.
    Uses the same filtering logic as the images API but performs cache warming instead of returning results.
    """

    print(f"Warmup cache for page {page}, page_size {page_size}, actor {actor}, tag {tag}, year {year}, has_caption {has_caption}, has_crop {has_crop}")

    start_time = time.time()
    
    # Apply filters if provided
    filter_start = time.time()
    filtered_images = apply_image_filters(
        filter_manager.cached_image_files,
        actor=actor,
        tag=tag,
        year=year,
        has_caption=has_caption,
        has_crop=has_crop
    )
    filter_time = time.time() - filter_start
    
    if filter_time > 1.0:
        print(f"Performance: Cache warmup filtering took {filter_time:.3f}s")
    
    # Calculate pagination
    total_pages, start_idx, end_idx, total_images = calculate_pagination(
        len(filtered_images),
        page,
        page_size
    )
    
    # Get images for current page
    page_images = filtered_images[start_idx:end_idx]
    
    # Warm up cache for each image in parallel
    warmup_start = time.time()
    warmed_up = 0
    skipped = 0
    errors = 0
    
    # Create tasks for all images
    tasks = []
    for img_file in page_images:
        image_id = str(img_file.stem)
        tasks.append(process_image_for_cache(image_id))
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Process results
    for success, was_skipped, was_error in results:
        if success:
            warmed_up += 1
        elif was_skipped:
            skipped += 1
        elif was_error:
            errors += 1
    
    warmup_time = time.time() - warmup_start
    total_time = time.time() - start_time
    
    if total_time > 1.0:
        print(f"Performance: Cache warmup for page {page} completed in {total_time:.3f}s (warmup: {warmup_time:.3f}s)")
    
    response = {
        "status": "success",
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "total_images": total_images,
        "warmed_up": warmed_up,
        "skipped": skipped,
        "errors": errors,
        "processing_time": total_time
    }
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
