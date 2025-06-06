from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, FileResponse
from typing import List, Optional
from pathlib import Path
import time
from PIL import Image as PILImage
import io
import asyncio

from ..models.models import ImageResponse
from ..services.image_service import (
    get_image_path,
    generate_cropped_image,
    apply_image_filters,
    calculate_pagination,
    process_image_for_cache,
    create_image_metadata
)
from ..services.cache_service import (
    image_cache,
    cached_image_files,
    PHOTOSET_METADATA_CACHE
)
from config import IMAGES_DIR, IMAGES_PER_PAGE

router = APIRouter()

@router.get("/images", response_model=ImageResponse)
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
    """Get a paginated list of images with optional filtering."""
    start_time = time.time()
    
    # Apply filters if provided
    filter_start = time.time()
    filtered_images = apply_image_filters(
        cached_image_files,
        actor=actor,
        tag=tag,
        year=year,
        has_caption=has_caption,
        has_crop=has_crop,
        photoset_metadata=PHOTOSET_METADATA_CACHE
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
    images_metadata = [create_image_metadata(img_file, PHOTOSET_METADATA_CACHE) for img_file in page_images]
    metadata_time = time.time() - metadata_start
    
    total_time = time.time() - start_time
    if total_time > 1.0:
        print(f"Performance: Page {page} processed in {total_time:.3f}s (metadata: {metadata_time:.3f}s)")
    
    # Trigger predictive cache warming for unfiltered requests
    if not (actor or tag or year or has_caption is not None or has_crop is not None):
        asyncio.create_task(predict_and_warm_cache(page))
    
    return ImageResponse(
        images=images_metadata,
        total=total_images,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )

@router.get("/images/batch", response_model=List[ImageResponse])
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
    """Get multiple pages of images at once for efficient preloading."""
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

@router.get("/images/{image_id}")
async def get_image(image_id: str, request: Request):
    """Get a specific image by ID from cache or file."""
    start_time = time.time()
    try:
        # Try to get from cache first
        cache_start = time.time()
        cached_image = image_cache.get(image_id)
        cache_time = time.time() - cache_start
        
        if cached_image:
            # Add caching headers
            headers = {
                "Cache-Control": "public, max-age=31536000",  # Cache for 1 year
                "ETag": f'"{hash(cached_image)}"',  # Use hash of image data as ETag
                "X-Cache-Status": "HIT",  # Add cache status header
                "X-Cache-Lookup-Time": f"{cache_time:.3f}"  # Add cache lookup time
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
        image_path = get_image_path(image_id, IMAGES_DIR)
        if not image_path:
            raise HTTPException(status_code=404, detail="Image not found")
        
        file_time = time.time() - file_start
        
        # Open and optimize the image
        process_start = time.time()
        with PILImage.open(image_path) as img:
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
            image_cache.put(image_id, image_data)
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
                "X-Cache-Store-Time": f"{cache_time:.3f}"
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

@router.get("/images/{image_id}/preview/{target_size}")
async def get_image_preview(image_id: str, target_size: int):
    """Get a scaled preview of the image that fits within the target size."""
    try:
        image_path = get_image_path(image_id, IMAGES_DIR)
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

async def predict_and_warm_cache(current_page: int, page_size: int = IMAGES_PER_PAGE):
    """Predict and warm up cache for likely upcoming requests."""
    try:
        # Create a mock request for background task
        mock_request = Request(scope={
            'type': 'http',
            'method': 'GET',
            'path': '/images',
            'headers': []
        })
        
        # Get current page images
        response = await get_images(
            request=mock_request,
            page=current_page,
            page_size=page_size
        )
        
        # Warm up current page images
        current_page_tasks = []
        for image_metadata in response.images:
            image_id = image_metadata.id
            if image_id not in image_cache.cache:
                current_page_tasks.append(process_image_for_cache(image_id, IMAGES_DIR))
        
        # Start warming up current page images
        if current_page_tasks:
            asyncio.create_task(asyncio.gather(*current_page_tasks))
        
        # Warm up next 3 pages (typical batch size)
        next_pages_task = asyncio.create_task(warm_up_next_pages(current_page, num_pages=3))
        
        # Wait for current page to complete before starting next pages
        if current_page_tasks:
            await asyncio.gather(*current_page_tasks)
        await next_pages_task
        
    except Exception as e:
        print(f"Error in predict_and_warm_cache: {str(e)}")

async def warm_up_next_pages(current_page: int, num_pages: int = 10):
    """Warm up cache for next N pages."""
    for page in range(current_page + 1, current_page + num_pages + 1):
        await predict_and_warm_cache(page)
