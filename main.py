from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, OrderedDict
import os
from datetime import datetime
import json
from pathlib import Path
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image as PILImage
import io
import zipfile
from io import BytesIO
import time # Import time module
from config import IMAGES_DIR, IMAGES_PER_PAGE, SERVER_HOST, SERVER_PORT, PROFILING_ENABLED, PROFILING_DIR, PHOTOSET_METADATA_DIRECTORY # Import profiling config
from caption_generator import get_caption_generator
import uvicorn
import threading
from collections import OrderedDict
import sys
import asyncio
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.datastructures import Headers
from starlette.requests import Request as StarletteRequest
from starlette.websockets import WebSocket
from starlette.types import Scope, Receive, Send

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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define cache file paths
DIMENSIONS_CACHE_FILE = IMAGES_DIR / "dimensions_cache.json"
CAPTIONS_CACHE_FILE = IMAGES_DIR / "captions_cache.json"
CROPS_CACHE_FILE = IMAGES_DIR / "crops_cache.json"

# In-memory storage for captions, crops, and dimensions
image_captions: Dict[str, str] = {}
image_crops: Dict[str, dict] = {}  # Store crop info: {imageId: {"targetSize": int, "normalizedDeltas": {"x": float, "y": float}}}
image_dimensions: Dict[str, tuple[int, int]] = {} # Store dimensions: {imageId: (width, height)}
cached_image_files: List[Path] = [] # Cache for the list of image file paths

# Initialize photoset metadata cache
PHOTOSET_METADATA_CACHE = {
    'actors': {},  # actor_name -> set of scene_ids
    'tags': {},    # tag_name -> set of scene_ids
    'year': {},    # year -> set of scene_ids
    'scenes': set(),  # set of all scene_ids
    'scene_metadata': {}  # scene_id -> {actors: [], tags: [], year: str}
}

# Initialize caption generator
caption_generator = get_caption_generator()

# Image cache implementation
class ImageCache:
    def __init__(self, max_size_bytes: int = 10 * 1024 * 1024 * 1024):  # 10GB default
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.cache: OrderedDict[str, bytes] = OrderedDict()  # LRU cache
        self.lock = threading.Lock()
    
    def get(self, image_id: str) -> Optional[bytes]:
        print(f"Getting image from cache {image_id}")
        print(f"Cache size: {len(self.cache)} ")

        with self.lock:
            if image_id in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(image_id)
                self.cache[image_id] = value
                return value
            return None
    
    def put(self, image_id: str, image_data: bytes):
        print(f"Putting image in cache {image_id}")
        print(f"Cache size: {len(self.cache)} ")

        with self.lock:
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
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0

# Initialize image cache
image_cache = ImageCache()

def read_photoset_metadata():
    global PHOTOSET_METADATA_CACHE
    
    # Read all JSON files in the metadata directory
    json_files = [f for f in os.listdir(PHOTOSET_METADATA_DIRECTORY) if f.endswith('.json')]
    
    for filename in json_files:
        filename_base = os.path.splitext(filename)[0]
        file_path = os.path.join(PHOTOSET_METADATA_DIRECTORY, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Initialize scene metadata
                scene_metadata = {
                    'actors': [],
                    'tags': [],
                    'year': None
                }
                
                # Process actors
                for each_actor in data['actors']:
                    scene_set = PHOTOSET_METADATA_CACHE['actors'].get(each_actor, set())
                    scene_set.add(filename_base)
                    PHOTOSET_METADATA_CACHE['actors'][each_actor] = scene_set
                    scene_metadata['actors'].append(each_actor)

                # Process tags
                for each_tag in data['tags']:
                    scene_set = PHOTOSET_METADATA_CACHE['tags'].get(each_tag, set())
                    scene_set.add(filename_base)
                    PHOTOSET_METADATA_CACHE['tags'][each_tag] = scene_set
                    scene_metadata['tags'].append(each_tag)

                # Process year
                year = data['date'].split(', ')[1]
                scene_set = PHOTOSET_METADATA_CACHE['year'].get(year, set())
                scene_set.add(filename_base)
                PHOTOSET_METADATA_CACHE['year'][year] = scene_set
                scene_metadata['year'] = year

                # Add scene to scenes set and store its metadata
                PHOTOSET_METADATA_CACHE['scenes'].add(filename_base)
                PHOTOSET_METADATA_CACHE['scene_metadata'][filename_base] = scene_metadata
                                        
        except (json.JSONDecodeError, IOError) as e:
            continue

def load_cache(cache_file: Path) -> dict:
    """Loads cache from a JSON file."""
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {cache_file}. Starting with empty cache.")
            return {}
        except Exception as e:
            print(f"Error loading cache from {cache_file}: {e}. Starting with empty cache.")
            return {}
    return {}

def save_cache(cache_data: dict, cache_file: Path):
    """Saves cache to a JSON file."""
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cache saved to {cache_file}")
    except Exception as e:
        print(f"Error saving cache to {cache_file}: {e}")


def initialize_caption_cache():
    """
    Load captions from cache file and scan directory for new captions.
    """
    print("Initializing caption cache...")
    global image_captions # Declare intent to modify the global variable
    image_captions = load_cache(CAPTIONS_CACHE_FILE)
    
    # Get current caption files in the directory
    current_caption_files = {file.stem.replace('_caption', '') for file in IMAGES_DIR.glob("*_caption.txt")} # Use a set for faster lookup

    # Find new caption files not in cache
    new_caption_ids = current_caption_files - set(image_captions.keys())

    for image_id in new_caption_ids:
        try:
            # Read caption from file for new files
            caption_file = IMAGES_DIR / f"{image_id}_caption.txt"
            if caption_file.exists():
                 with open(caption_file, 'r') as f:
                     caption = f.read().strip()
                     image_captions[image_id] = caption
        except Exception as e:
            print(f"Error processing new caption file {caption_file}: {e}")
            continue
            
    print(f"Caption cache initialized with {len(image_captions)} entries.")

def initialize_crop_cache():
    """
    Load crop info from cache file and scan directory for new crop info.
    """
    print("Initializing crop cache...")
    global image_crops # Declare intent to modify the global variable
    image_crops = load_cache(CROPS_CACHE_FILE)
    
    # Get current crop metadata files in the directory
    current_crop_files = {file.stem.replace('_crop_', '') for file in IMAGES_DIR.glob("*_crop_*.json")} # Use a set for faster lookup

    # Find new crop files not in cache
    new_crop_ids = current_crop_files - set(image_crops.keys())

    for image_id in new_crop_ids:
         try:
            # Try to load metadata from JSON file for new files
            # Assuming there might be multiple crop sizes per image, we'll just take the first one found for caching purposes
            # A more robust approach might handle multiple crops per image ID
            metadata_files = list(IMAGES_DIR.glob(f"{image_id}_crop_*.json"))
            if metadata_files:
                metadata_path = metadata_files[0]
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    image_crops[image_id] = {
                        "targetSize": metadata.get("targetSize"), # Use .get() for safety
                        "normalizedDeltas": metadata.get("normalizedDeltas") # Use .get() for safety
                    }
         except (ValueError, IndexError, json.JSONDecodeError) as e:
             print(f"Error processing new crop file {metadata_path}: {e}") # Use metadata_path here
             continue
             
    print(f"Crop cache initialized with {len(image_crops)} entries.")

def initialize_dimension_cache():
    """
    Load dimensions from cache file and scan directory for new images.
    """
    print("Initializing dimension cache...")
    global image_dimensions # Declare intent to modify the global variable
    
    # Load existing cache
    # Note: JSON keys are strings, so tuple keys (width, height) need handling if you were saving that way.
    # Our current dimension cache uses imageId as key, which is already a string, so direct load works.
    loaded_cache = load_cache(DIMENSIONS_CACHE_FILE)
    # Convert list from JSON back to tuple if necessary (json saves tuples as lists)
    image_dimensions = {k: tuple(v) for k, v in loaded_cache.items()} if loaded_cache else {}
    
    # Get all image files, excluding cropped images and non-image files
    all_image_files = [
        f for f in IMAGES_DIR.glob("*") 
        if f.is_file() 
        and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        and "_crop_" not in f.name  # Exclude cropped images
        and not f.name.startswith('.')  # Exclude hidden files like .DS_Store
    ]
    
    # Find new image files not in cache
    current_image_ids = {str(f.stem) for f in all_image_files}
    cached_image_ids = set(image_dimensions.keys())
    new_image_ids = current_image_ids - cached_image_ids

    print(f"Found {len(new_image_ids)} new images not in dimension cache.")

    for img_file in all_image_files:
        image_id = str(img_file.stem)
        # Only process if it's a new image or its dimension is missing from cache
        if image_id in new_image_ids or image_id not in image_dimensions:
             width, height = get_image_dimensions_from_file(img_file)
             if width is not None and height is not None:
                 image_dimensions[image_id] = (width, height) # Cache the result

    print(f"Dimension cache initialized/updated with {len(image_dimensions)} entries.")

def get_image_dimensions_from_file(image_path: Path) -> tuple[Optional[int], Optional[int]]:
    """Helper function to get image dimensions by opening the file."""
    try:
        with PILImage.open(image_path) as img:
            return img.size
    except Exception:
        return None, None

def get_image_dimensions(image_id: str) -> tuple[Optional[int], Optional[int]]:
    """Get image dimensions from cache or file, and cache the result if read from file."""
    # This function now primarily serves as a getter from the in-memory cache
    # The caching from file is handled during initialization and when new images are added/processed via other endpoints if necessary.
    # However, the current logic in initialize_dimension_cache covers the main case.
    
    if image_id in image_dimensions:
        return image_dimensions[image_id]
    
    # Fallback: If somehow not in cache (shouldn't happen after proper initialization), read from file and add to cache
    print(f"Warning: Image dimensions for {image_id} not in cache. Reading from file.")
    image_path = get_image_path(image_id)
    if not image_path:
        return None, None
    
    width, height = get_image_dimensions_from_file(image_path)
    if width is not None and height is not None:
        image_dimensions[image_id] = (width, height) # Cache the result
        # Consider saving cache here too if you expect dimensions to be fetched outside of startup for new files

    return width, height

def get_image_path(image_id: str) -> Optional[Path]:
    """Get the full path of an image file by its ID."""
    # This function is correct as is
    image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
    return image_files[0] if image_files else None

class MockRequest:
    """A mock request object for background tasks"""
    def __init__(self):
        self.state = type('State', (), {'request_id': str(uuid.uuid4())})()
        self.method = "BACKGROUND"
        self.url = type('URL', (), {'path': "/background/warmup"})()
        self.headers = Headers({})

async def warm_up_cache(page: int = 1, page_size: int = IMAGES_PER_PAGE):
    """Warm up the cache with images from a specific page"""
    try:
        # Create a mock request for background task
        mock_request = MockRequest()
        
        # Get images for the page
        response = await get_images(
            request=mock_request,
            page=page,
            page_size=page_size
        )
        
        # Process each image in the page
        for image_metadata in response.images:
            image_id = image_metadata.id
            if image_id not in image_cache.cache:
                try:
                    # Get image path
                    image_path = get_image_path(image_id)
                    if not image_path:
                        continue
                    
                    # Open and optimize the image
                    with PILImage.open(image_path) as img:
                        if img.mode in ('RGBA', 'P'):
                            img = img.convert('RGB')
                        
                        # Create a BytesIO object to store the optimized image
                        output = io.BytesIO()
                        img.save(output, format='JPEG', quality=85, optimize=True)
                        output.seek(0)
                        
                        # Add to cache
                        image_cache.put(image_id, output.getvalue())
                except Exception as e:
                    print(f"Error warming up cache for image {image_id}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error in warm_up_cache: {str(e)}")

async def warm_up_next_pages(current_page: int, num_pages: int = 10):
    """Warm up cache for next N pages"""
    for page in range(current_page + 1, current_page + num_pages + 1):
        await warm_up_cache(page)

async def predict_and_warm_cache(current_page: int, page_size: int = IMAGES_PER_PAGE):
    """
    Predict and warm up cache for likely upcoming requests based on current page.
    This includes:
    1. Current page images (if not already cached)
    2. Next 3 pages (typical batch request size)
    3. Preview sizes for current page images
    """
    try:
        # Create a mock request for background task
        mock_request = MockRequest()
        
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
                current_page_tasks.append(warm_up_cache(current_page))
        
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

@app.on_event("startup")
async def startup_event():
    """Initialize caches and warm up image cache on startup"""
    print("Loading caches on startup...")
    
    # Ensure profiling directory exists if profiling is enabled
    if PROFILING_ENABLED:
        PROFILING_DIR.mkdir(exist_ok=True)
    
    # Initialize photoset metadata cache
    read_photoset_metadata()
    print(f"Photoset metadata cache initialized with {len(PHOTOSET_METADATA_CACHE['scenes'])} scenes")
        
    initialize_caption_cache()
    initialize_crop_cache()
    initialize_dimension_cache()
    
    # Build cached image file list on startup
    print("Building cached image file list...")
    global cached_image_files
    cached_image_files = [
        f for f in IMAGES_DIR.glob("*") 
        if f.is_file() 
        and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        and "_crop_" not in f.name
        and not f.name.startswith('.')
    ]
    cached_image_files.sort()
    print(f"Cached image file list built with {len(cached_image_files)} files.")
    
    # Warm up image cache with first page
    print("Warming up image cache...")
    await warm_up_cache(page=1)
    print("Image cache warmed up.")
    
    print("Caches loaded and image list built.")

@app.on_event("shutdown")
async def save_all_caches():
    """Saves all caches to files on shutdown."""
    print("Saving caches on shutdown...")
    save_cache(image_captions, CAPTIONS_CACHE_FILE)
    save_cache(image_crops, CROPS_CACHE_FILE)
    # Convert tuples in dimension cache to lists for JSON serialization
    dimensions_to_save = {k: list(v) for k, v in image_dimensions.items()}
    save_cache(dimensions_to_save, DIMENSIONS_CACHE_FILE)
    print("Caches saved.")

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

class NormalizedDeltas(BaseModel):
    x: float
    y: float

class CropRequest(BaseModel):
    imageId: str
    targetSize: int
    normalizedDeltas: NormalizedDeltas

class ExportRequest(BaseModel):
    imageIds: List[str]

# Configuration
# IMAGES_DIR = Path("/Users/stuartleal/Library/Mobile Documents/com~apple~CloudDocs/Downloads/4800watermarked")


# Ensure images directory exists
IMAGES_DIR.mkdir(exist_ok=True)

@app.get("/filters", response_model=Dict[str, List[str]])
async def get_available_filters():
    """
    Get all available filters (actors, tags, years) that can be used to filter images.
    """
    return {
        "actors": sorted(PHOTOSET_METADATA_CACHE['actors'].keys()),
        "tags": sorted(PHOTOSET_METADATA_CACHE['tags'].keys()),
        "years": sorted(PHOTOSET_METADATA_CACHE['year'].keys())
    }

def find_base_name(image_id: str) -> str:
    """
    Find the base name for an image by matching against scene metadata keys.
    
    The scene metadata keys contain the original base names for images. This function
    looks for a scene metadata key that is contained within the image ID.
    
    Args:
        image_id (str): The full image ID to find the base name for
        
    Returns:
        str: The matching base name from scene metadata, or None if no match is found
    """
    scene_keys = PHOTOSET_METADATA_CACHE['scene_metadata'].keys()
    for scene_key in scene_keys:
        if scene_key in image_id:
            return scene_key
    return None

def apply_image_filters(
    images: List[Path],
    actor: Optional[str] = None,
    tag: Optional[str] = None,
    year: Optional[str] = None,
    has_caption: Optional[bool] = None,
    has_crop: Optional[bool] = None
) -> List[Path]:
    """
    Apply filters to a list of images based on the provided criteria.
    
    Args:
        images: List of image paths to filter
        actor: Filter by actor name
        tag: Filter by tag
        year: Filter by year
        has_caption: Filter for images with captions
        has_crop: Filter for images with crops
        
    Returns:
        List of filtered image paths
    """
    if not any([actor, tag, year, has_caption is not None, has_crop is not None]):
        return images
        
    filtered_images = []
    for img_file in images:
        image_id = str(img_file.stem)
        base_name = find_base_name(image_id)
        include = True
        
        if actor:
            actor_scenes = PHOTOSET_METADATA_CACHE['actors'].get(actor, set())
            if base_name not in actor_scenes:
                include = False
        
        if tag and include:
            tag_scenes = PHOTOSET_METADATA_CACHE['tags'].get(tag, set())
            if base_name not in tag_scenes:
                include = False
        
        if year and include:
            year_scenes = PHOTOSET_METADATA_CACHE['year'].get(year, set())
            if base_name not in year_scenes:
                include = False
        
        if has_caption is not None and include:
            has_caption_value = image_id in image_captions
            if has_caption != has_caption_value:
                include = False
        
        if has_crop is not None and include:
            has_crop_value = image_id in image_crops
            if has_crop != has_crop_value:
                include = False
            
        if include:
            filtered_images.append(img_file)
    
    return filtered_images

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
    base_name = find_base_name(image_id)
    width, height = image_dimensions.get(image_id, (None, None))
    
    scene_metadata = PHOTOSET_METADATA_CACHE['scene_metadata'].get(base_name, {
        'actors': [],
        'tags': [],
        'year': None
    })
    
    return ImageMetadata(
        id=image_id,
        filename=img_file.name,
        size=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_ctime),
        mime_type=f"image/{img_file.suffix[1:].lower()}" if img_file.suffix else "application/octet-stream",
        width=width,
        height=height,
        has_caption=image_id in image_captions,
        collection_name="Default Collection",
        has_tags=len(scene_metadata['tags']) > 0,
        has_crop=image_id in image_crops,
        year=scene_metadata['year'],
        tags=scene_metadata['tags'],
        actors=scene_metadata['actors']
    )

async def process_image_for_cache(image_id: str) -> tuple[bool, bool, bool]:
    """
    Process an image for caching.
    
    Args:
        image_id: ID of the image to process
        
    Returns:
        Tuple of (success, was_skipped, was_error)
    """
    # Skip if already in cache
    if image_id in image_cache.cache:
        print(f"Image {image_id} already in cache")
        return False, True, False
    
    try:
        # Get image path
        print(f"Getting image path for {image_id}")
        image_path = get_image_path(image_id)
        if not image_path:
            return False, False, True
        
        # Open and optimize the image
        print(f"Opening image {image_path}")
        with PILImage.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Create a BytesIO object to store the optimized image
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            output.seek(0)
            
            # Add to cache
            print(f"Adding image to cache {image_id}")
            image_cache.put(image_id, output.getvalue())
            return True, False, False
            
    except Exception as e:
        print(f"Error processing image {image_id}: {str(e)}")
        return False, False, True

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
        cached_image_files,
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
    if not (actor or tag or year or has_caption is not None or has_crop is not None):
        asyncio.create_task(predict_and_warm_cache(page))
    
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
            image_captions[image_id] = caption_request.caption
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
        caption = image_captions.get(image_id)
        if not caption:
            raise HTTPException(status_code=404, detail="No caption found for this image")
        
        return CaptionResponse(caption=caption)
    
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
            print("It's the first")
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Get crop info
        crop_info = image_crops.get(image_id)
        if not crop_info:
            print("It's the second")
            raise HTTPException(status_code=404, detail="No crop found for this image")
        
        # Get the cropped image
        cropped_image_path = os.path.join(IMAGES_DIR, f"{image_id}_crop_{crop_info['targetSize']}.png")
        if not os.path.exists(cropped_image_path):
            print("It's the third")
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
                if image_id in image_crops:
                    crop_info = image_crops[image_id]
                    crop_path = os.path.join(IMAGES_DIR, f"{image_id}_crop_{crop_info['targetSize']}.png")
                    print(f"  Checking for crop at: {crop_path}")
                    
                    if os.path.exists(crop_path):
                        print(f"  Adding cropped image to zip")
                        with open(crop_path, 'rb') as f:
                            # Use the same base name for both crop and caption
                            zip_file.writestr(f"{base_name}.png", f.read())
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
        cached_image_files,
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
    
    # Warm up cache for each image
    warmup_start = time.time()
    warmed_up = 0
    skipped = 0
    errors = 0
    
    for img_file in page_images:
        image_id = str(img_file.stem)
        success, was_skipped, was_error = await process_image_for_cache(image_id)
        
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
