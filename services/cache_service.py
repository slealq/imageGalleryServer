import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image as PILImage
import io

class ImageCache:
    def __init__(self, max_size_bytes: int = 10 * 1024 * 1024 * 1024):  # 10GB default
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.cache: OrderedDict[str, bytes] = OrderedDict()  # LRU cache
        self.lock = threading.Lock()
    
    def get(self, image_id: str) -> Optional[bytes]:
        with self.lock:
            if image_id in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(image_id)
                self.cache[image_id] = value
                return value
            return None
    
    def put(self, image_id: str, image_data: bytes):
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

# Global caches
image_captions: Dict[str, str] = {}
image_crops: Dict[str, dict] = {}
image_dimensions: Dict[str, Tuple[int, int]] = {}
cached_image_files: List[Path] = []

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

def initialize_caption_cache(cache_file: Path, images_dir: Path):
    """Initialize caption cache from file and directory."""
    global image_captions
    image_captions = load_cache(cache_file)
    
    current_caption_files = {file.stem.replace('_caption', '') for file in images_dir.glob("*_caption.txt")}
    new_caption_ids = current_caption_files - set(image_captions.keys())

    for image_id in new_caption_ids:
        try:
            caption_file = images_dir / f"{image_id}_caption.txt"
            if caption_file.exists():
                with open(caption_file, 'r') as f:
                    caption = f.read().strip()
                    image_captions[image_id] = caption
        except Exception as e:
            print(f"Error processing new caption file {caption_file}: {e}")
            continue
            
    print(f"Caption cache initialized with {len(image_captions)} entries.")

def initialize_crop_cache(cache_file: Path, images_dir: Path):
    """Initialize crop cache from file and directory."""
    global image_crops
    image_crops = load_cache(cache_file)
    
    current_crop_files = {file.stem.replace('_crop_', '') for file in images_dir.glob("*_crop_*.json")}
    new_crop_ids = current_crop_files - set(image_crops.keys())

    for image_id in new_crop_ids:
        try:
            metadata_files = list(images_dir.glob(f"{image_id}_crop_*.json"))
            if metadata_files:
                metadata_path = metadata_files[0]
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    image_crops[image_id] = {
                        "targetSize": metadata.get("targetSize"),
                        "normalizedDeltas": metadata.get("normalizedDeltas")
                    }
        except Exception as e:
            print(f"Error processing new crop file {metadata_path}: {e}")
            continue
            
    print(f"Crop cache initialized with {len(image_crops)} entries.")

def initialize_dimension_cache(cache_file: Path, images_dir: Path):
    """Initialize dimension cache from file and directory."""
    global image_dimensions
    
    loaded_cache = load_cache(cache_file)
    image_dimensions = {k: tuple(v) for k, v in loaded_cache.items()} if loaded_cache else {}
    
    all_image_files = [
        f for f in images_dir.glob("*") 
        if f.is_file() 
        and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        and "_crop_" not in f.name
        and not f.name.startswith('.')
    ]
    
    current_image_ids = {str(f.stem) for f in all_image_files}
    cached_image_ids = set(image_dimensions.keys())
    new_image_ids = current_image_ids - cached_image_ids

    print(f"Found {len(new_image_ids)} new images not in dimension cache.")

    for img_file in all_image_files:
        image_id = str(img_file.stem)
        if image_id in new_image_ids or image_id not in image_dimensions:
            width, height = get_image_dimensions_from_file(img_file)
            if width is not None and height is not None:
                image_dimensions[image_id] = (width, height)

    print(f"Dimension cache initialized/updated with {len(image_dimensions)} entries.")

def get_image_dimensions_from_file(image_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Get image dimensions by opening the file."""
    try:
        with PILImage.open(image_path) as img:
            return img.size
    except Exception:
        return None, None
