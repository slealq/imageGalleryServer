from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image as PILImage
import io
import time
from .cache_service import image_cache, image_dimensions, image_captions, image_crops, cached_image_files

def get_image_path(image_id: str, images_dir: Path) -> Optional[Path]:
    """Get the full path of an image file by its ID."""
    image_files = list(images_dir.glob(f"{image_id}.*"))
    return image_files[0] if image_files else None

def generate_cropped_image(image: PILImage, target_size: int):
    """Generate a cropped image that fits within the target size."""
    width, height = image.size
    scale = max(target_size / width, target_size / height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, PILImage.Resampling.LANCZOS)

def apply_image_filters(
    images: List[Path],
    actor: Optional[str] = None,
    tag: Optional[str] = None,
    year: Optional[str] = None,
    has_caption: Optional[bool] = None,
    has_crop: Optional[bool] = None,
    photoset_metadata: dict = None
) -> List[Path]:
    """Apply filters to a list of images based on the provided criteria."""
    if not any([actor, tag, year, has_caption is not None, has_crop is not None]):
        return images
        
    filtered_images = []
    for img_file in images:
        image_id = str(img_file.stem)
        base_name = '_'.join(image_id.split('_')[:-2])
        include = True
        
        if actor:
            actor_scenes = photoset_metadata['actors'].get(actor, set())
            if base_name not in actor_scenes:
                include = False
        
        if tag and include:
            tag_scenes = photoset_metadata['tags'].get(tag, set())
            if base_name not in tag_scenes:
                include = False
        
        if year and include:
            year_scenes = photoset_metadata['year'].get(year, set())
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
) -> Tuple[int, int, int, int]:
    """Calculate pagination parameters."""
    total_pages = (total_items + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_items)
    return total_pages, start_idx, end_idx, total_items

async def process_image_for_cache(image_id: str, images_dir: Path) -> Tuple[bool, bool, bool]:
    """Process an image for caching."""
    if image_id in image_cache.cache:
        return False, True, False
    
    try:
        image_path = get_image_path(image_id, images_dir)
        if not image_path:
            return False, False, True
        
        with PILImage.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            output.seek(0)
            
            image_cache.put(image_id, output.getvalue())
            return True, False, False
            
    except Exception as e:
        print(f"Error processing image {image_id}: {str(e)}")
        return False, False, True

def create_image_metadata(img_file: Path, photoset_metadata: dict) -> dict:
    """Create metadata for a single image."""
    stat = img_file.stat()
    image_id = str(img_file.stem)
    base_name = '_'.join(image_id.split('_')[:-2])
    width, height = image_dimensions.get(image_id, (None, None))
    
    scene_metadata = photoset_metadata['scene_metadata'].get(base_name, {
        'actors': [],
        'tags': [],
        'year': None
    })
    
    return {
        "id": image_id,
        "filename": img_file.name,
        "size": stat.st_size,
        "created_at": stat.st_ctime,
        "mime_type": f"image/{img_file.suffix[1:].lower()}" if img_file.suffix else "application/octet-stream",
        "width": width,
        "height": height,
        "has_caption": image_id in image_captions,
        "collection_name": "Default Collection",
        "has_tags": len(scene_metadata['tags']) > 0,
        "has_crop": image_id in image_crops,
        "year": scene_metadata['year'],
        "tags": scene_metadata['tags'],
        "actors": scene_metadata['actors']
    }
