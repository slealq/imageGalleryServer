from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
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

app = FastAPI(title="Image Service API")

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

def read_photoset_metadata():
    global PHOTOSET_METADATA_CACHE
    print(f"Reading photoset metadata from directory: {PHOTOSET_METADATA_DIRECTORY}")
    
    # Read all JSON files in the metadata directory
    json_files = [f for f in os.listdir(PHOTOSET_METADATA_DIRECTORY) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in metadata directory")
    
    for filename in json_files:
        filename_base = os.path.splitext(filename)[0]
        print(f"\nProcessing metadata file: {filename}")
        print(f"Base name: {filename_base}")

        file_path = os.path.join(PHOTOSET_METADATA_DIRECTORY, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                print(f"Loaded JSON data: {data}")
                
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
                    print(f"Added actor {each_actor} for scene {filename_base}")

                # Process tags
                for each_tag in data['tags']:
                    scene_set = PHOTOSET_METADATA_CACHE['tags'].get(each_tag, set())
                    scene_set.add(filename_base)
                    PHOTOSET_METADATA_CACHE['tags'][each_tag] = scene_set
                    scene_metadata['tags'].append(each_tag)
                    print(f"Added tag {each_tag} for scene {filename_base}")

                # Process year
                year = data['date'].split(', ')[1]
                scene_set = PHOTOSET_METADATA_CACHE['year'].get(year, set())
                scene_set.add(filename_base)
                PHOTOSET_METADATA_CACHE['year'][year] = scene_set
                scene_metadata['year'] = year
                print(f"Added year {year} for scene {filename_base}")

                # Add scene to scenes set and store its metadata
                PHOTOSET_METADATA_CACHE['scenes'].add(filename_base)
                PHOTOSET_METADATA_CACHE['scene_metadata'][filename_base] = scene_metadata
                print(f"Added scene {filename_base} to scenes set with metadata")
                                        
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {filename}: {str(e)}")
            continue
    
    print("\nFinal cache state:")
    print(f"Number of actors: {len(PHOTOSET_METADATA_CACHE['actors'])}")
    print(f"Number of tags: {len(PHOTOSET_METADATA_CACHE['tags'])}")
    print(f"Number of years: {len(PHOTOSET_METADATA_CACHE['year'])}")
    print(f"Number of scenes: {len(PHOTOSET_METADATA_CACHE['scenes'])}")
    print(f"Number of scene metadata entries: {len(PHOTOSET_METADATA_CACHE['scene_metadata'])}")

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

@app.on_event("startup")
async def load_all_caches():
    """Loads all caches from files and builds the cached image file list on startup."""
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
        and f.suffix.lower() in ['.jpg', '.jpeg', '.png']  # Only include image files
        and "_crop_" not in f.name  # Exclude cropped images
        and not f.name.startswith('.')  # Exclude hidden files like .DS_Store
    ]
    cached_image_files.sort()
    print(f"Cached image file list built with {len(cached_image_files)} files.")
    
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

@app.get("/images", response_model=ImageResponse)
async def get_images(
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
    """
    # Start timing if profiling is enabled
    step_start_time = time.time() if PROFILING_ENABLED else None
    
    # Apply filters if provided
    filtered_images = cached_image_files
    if actor or tag or year or has_caption is not None or has_crop is not None:
        print(f"\nApplying filters:")
        # Strip quotes from actor name if present
        if actor:
            actor = actor.strip('"\'')
        print(f"Actor filter: {actor}")
        print(f"Tag filter: {tag}")
        print(f"Year filter: {year}")
        print(f"Has caption filter: {has_caption}")
        print(f"Has crop filter: {has_crop}")
        print(f"Available actors in cache: {list(PHOTOSET_METADATA_CACHE['actors'].keys())}")
        
        filtered_images = []
        for img_file in cached_image_files:
            image_id = str(img_file.stem)
            # Extract base name by removing the additional numbers at the end
            # The format is typically: Name_Name__Date_Resolution_Number_Number
            # We want to keep everything up to the resolution number
            base_name = '_'.join(image_id.split('_')[:-2])  # Remove last two number segments
            include = True
            
            print(f"\nProcessing image: {image_id}")
            print(f"Base name: {base_name}")

            print(f"Available actors in cache: {list(PHOTOSET_METADATA_CACHE['actors'].keys())}")

            if actor in PHOTOSET_METADATA_CACHE['actors']:
                print(f"Actor {actor} found in cache")
            else:
                print(f"Actor {actor} not found in cache")

            actor_scenes = PHOTOSET_METADATA_CACHE['actors'].get(actor, set())
            print(f"Scenes for actor '{actor}': {actor_scenes}")
            
            if actor:
                if base_name not in actor_scenes:
                    print(f"Image {base_name} not found in scenes for actor {actor}")
                    include = False
                else:
                    print(f"Image {base_name} found in scenes for actor {actor}")
            
            if tag and include:
                tag_scenes = PHOTOSET_METADATA_CACHE['tags'].get(tag, set())
                print(f"Scenes for tag '{tag}': {tag_scenes}")
                if base_name not in tag_scenes:
                    print(f"Image {base_name} not found in scenes for tag {tag}")
                    include = False
                else:
                    print(f"Image {base_name} found in scenes for tag {tag}")
            
            if year and include:
                year_scenes = PHOTOSET_METADATA_CACHE['year'].get(year, set())
                print(f"Scenes for year '{year}': {year_scenes}")
                if base_name not in year_scenes:
                    print(f"Image {base_name} not found in scenes for year {year}")
                    include = False
                else:
                    print(f"Image {base_name} found in scenes for year {year}")
            
            # Check caption filter
            if has_caption is not None and include:
                has_caption_value = image_id in image_captions
                if has_caption != has_caption_value:
                    print(f"Image {image_id} caption status ({has_caption_value}) doesn't match filter ({has_caption})")
                    include = False
                else:
                    print(f"Image {image_id} caption status matches filter")
            
            # Check crop filter
            if has_crop is not None and include:
                has_crop_value = image_id in image_crops
                if has_crop != has_crop_value:
                    print(f"Image {image_id} crop status ({has_crop_value}) doesn't match filter ({has_crop})")
                    include = False
                else:
                    print(f"Image {image_id} crop status matches filter")
                
            if include:
                print(f"Including image {image_id} in filtered results")
                filtered_images.append(img_file)
            else:
                print(f"Excluding image {image_id} from filtered results")
        
        print(f"\nFiltering complete. Found {len(filtered_images)} matching images")
    
    # Calculate pagination
    total_images = len(filtered_images)
    total_pages = (total_images + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_images)
    
    if PROFILING_ENABLED and step_start_time is not None:
        print(f"Step 1: Apply filters and calculate pagination: {time.time() - step_start_time:.4f} seconds")
    
    # Get images for current page
    step_start_time = time.time() if PROFILING_ENABLED else None
    page_images = filtered_images[start_idx:end_idx]
    
    if PROFILING_ENABLED and step_start_time is not None:
        print(f"Step 2: Get images for current page: {time.time() - step_start_time:.4f} seconds")
    
    # Create metadata for each image
    step_start_time = time.time() if PROFILING_ENABLED else None

    images_metadata = []
    for img_file in page_images:
        stat = img_file.stat()
        image_id = str(img_file.stem)
        # Extract base name by removing the additional numbers at the end
        # The format is typically: Name_Name__Date_Resolution_Number_Number
        # We want to keep everything up to the resolution number
        base_name = '_'.join(image_id.split('_')[:-2])  # Remove last two number segments
        width, height = image_dimensions.get(image_id, (None, None))
        
        # Get metadata from scene_metadata index
        scene_metadata = PHOTOSET_METADATA_CACHE['scene_metadata'].get(base_name, {
            'actors': [],
            'tags': [],
            'year': None
        })
        
        print(f"Processing image_id: {image_id}")
        print(f"Processing base_name: {base_name}")
        print(f"Scene metadata keys: {list(PHOTOSET_METADATA_CACHE['scene_metadata'].keys())}")
        print(f"Scene metadata: {scene_metadata}")
        
        metadata = ImageMetadata(
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
        images_metadata.append(metadata)
    
    if PROFILING_ENABLED and step_start_time is not None:
        print(f"Step 3: Create metadata for {len(images_metadata)} images: {time.time() - step_start_time:.4f} seconds")
    
    return ImageResponse(
        images=images_metadata,
        total=total_images,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )

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

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
