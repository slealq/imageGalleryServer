import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image as PILImage
from config import METADATA_DIRECTORY, PHOTOSET_METADATA_DIRECTORY, CAPTIONS_DIRECTORY, IMAGES_DIR, IMAGE_METADATA_DIRECTORY
from caches.crop_cache import crop_cache

class FilterManager:
    """Manages image filtering operations."""
    
    def __init__(self):
        self.metadata_cache = {
            'actors': {},  # actor_name -> set of scene_ids
            'tags': {},    # tag_name -> set of scene_ids
            'year': {},    # year -> set of scene_ids
            'scenes': set(),  # set of all scene_ids
            'scene_metadata': {}  # scene_id -> {actors: [], tags: [], year: str}
        }
        
        # Define cache file paths
        self.DIMENSIONS_CACHE_FILE = METADATA_DIRECTORY / "dimensions_cache.json"
        self.CAPTIONS_CACHE_FILE = METADATA_DIRECTORY / "captions_cache.json"
        # the image metadata contains information that's only relevant for each image, and unique for it
        self.IMAGE_METADATA_FILE = IMAGE_METADATA_DIRECTORY / "metadata.json"
        
        # In-memory storage for captions, and dimensions
        self.image_captions: Dict[str, str] = {}
        self.image_dimensions: Dict[str, tuple[int, int]] = {} # Store dimensions: {imageId: (width, height)}
        self.cached_image_files: List[Path] = [] # Cache for the list of image file paths
        self.image_metadata: Dict[str, Dict] = {}

    def get_image_path(self, image_id: str) -> Optional[Path]:
        """Get the full path of an image file by its ID."""
        # This function is correct as is
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        return image_files[0] if image_files else None

    def get_available_filters(self):
        # Get all image tags from image metadata
        all_per_image_tags = set()

        for image_id, metadata in self.image_metadata.items():
            if 'tags' in metadata:
                all_per_image_tags.update(metadata['tags'])

        return {
            "actors": sorted(self.metadata_cache['actors'].keys()),
            "tags": sorted(set(self.metadata_cache['tags'].keys()) | all_per_image_tags),
            "years": sorted(self.metadata_cache['year'].keys())
        }
    
    def get_image_filter_metadata(self, image_id: str):
        base_name = self.find_base_name(image_id)
        print(f"Debug: Getting metadata for image {image_id}")
        print(f"Debug: Base name found: {base_name}")

        # Get image-specific tags
        image_tags = set(self.get_tags_from_file_for_image(image_id))
        print(f"Debug: Image-specific tags: {image_tags}")

        # Get scene metadata
        image_metadata = self.metadata_cache['scene_metadata'].get(base_name, {
            'actors': [],
            'tags': [],
            'year': None
        })
        print(f"Debug: Scene metadata: {image_metadata}")

        # Create a new copy of the metadata to avoid modifying the cache
        new_metadata = {
            'actors': image_metadata['actors'].copy(),
            'tags': image_metadata['tags'].copy(),
            'year': image_metadata['year']
        }

        # Combine and deduplicate tags
        scene_tags = set(new_metadata["tags"])
        print(f"Debug: Scene tags: {scene_tags}")
        all_tags = scene_tags.union(image_tags)
        new_metadata["tags"] = list(all_tags)
        print(f"Debug: Combined tags: {new_metadata['tags']}")
        
        return new_metadata
    
    def read_photoset_metadata(self):
        """Read and cache photoset metadata from JSON files."""
        # Read all JSON files in the metadata directory
        json_files = [f for f in os.listdir(PHOTOSET_METADATA_DIRECTORY) if f.endswith('.json')]
        print(f"Debug: Found {len(json_files)} photoset metadata files")
        
        for filename in json_files:
            filename_base = os.path.splitext(filename)[0]
            file_path = os.path.join(PHOTOSET_METADATA_DIRECTORY, filename)
            print(f"Debug: Processing photoset file: {filename}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    print(f"Debug: Loaded data from {filename}: {data}")
                    
                    # Initialize scene metadata
                    scene_metadata = {
                        'actors': [],
                        'tags': [],
                        'year': None
                    }
                    
                    # Process actors
                    for each_actor in data['actors']:
                        scene_set = self.metadata_cache['actors'].get(each_actor, set())
                        scene_set.add(filename_base)
                        self.metadata_cache['actors'][each_actor] = scene_set
                        scene_metadata['actors'].append(each_actor)

                    # Process tags
                    for each_tag in data['tags']:
                        scene_set = self.metadata_cache['tags'].get(each_tag, set())
                        scene_set.add(filename_base)
                        self.metadata_cache['tags'][each_tag] = scene_set
                        scene_metadata['tags'].append(each_tag)

                    # Process year
                    year = data['date'].split(', ')[1]
                    scene_set = self.metadata_cache['year'].get(year, set())
                    scene_set.add(filename_base)
                    self.metadata_cache['year'][year] = scene_set
                    scene_metadata['year'] = year

                    # Add scene to scenes set and store its metadata
                    self.metadata_cache['scenes'].add(filename_base)
                    self.metadata_cache['scene_metadata'][filename_base] = scene_metadata
                    print(f"Debug: Added scene metadata for {filename_base}: {scene_metadata}")
                                            
            except (json.JSONDecodeError, IOError) as e:
                print(f"Debug: Error processing {filename}: {e}")
                continue
    
    def find_base_name(self, image_id: str) -> str:
        """Find the base name for an image by matching against scene metadata keys."""
        scene_keys = self.metadata_cache['scene_metadata'].keys()
        for scene_key in scene_keys:
            if scene_key in image_id:
                return scene_key
        return None
    
    def apply_filters(
        self,
        images: List[Path],
        actor: Optional[str] = None,
        tag: Optional[str] = None,
        year: Optional[str] = None,
        has_caption: Optional[bool] = None,
        has_crop: Optional[bool] = None
    ) -> List[Path]:
        """Apply filters to a list of images based on the provided criteria."""
        if not any([actor, tag, year, has_caption is not None, has_crop is not None]):
            return images
            
        filtered_images = []
        for img_file in images:
            image_id = str(img_file.stem)
            base_name = self.find_base_name(image_id)
            include = True
            
            if actor:
                actor_scenes = self.metadata_cache['actors'].get(actor, set())
                if base_name not in actor_scenes:
                    include = False
            
            if tag and include:
                tag_scenes = self.metadata_cache['tags'].get(tag, set())
                tags_for_image = self.get_tags_from_file_for_image(image_id)
                if base_name not in tag_scenes and tag not in tags_for_image:
                    include = False
            
            if year and include:
                year_scenes = self.metadata_cache['year'].get(year, set())
                if base_name not in year_scenes:
                    include = False
            
            if has_caption is not None and include:
                has_caption_value = image_id in self.image_captions
                if has_caption != has_caption_value:
                    include = False
            
            if has_crop is not None and include:
                has_crop_value = crop_cache.has_crop_metadata(image_id)
                if has_crop != has_crop_value:
                    include = False
                
            if include:
                filtered_images.append(img_file)
        
        return filtered_images

    def load_cache(self, cache_file: Path) -> dict:
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

    def save_cache(self, cache_data: dict, cache_file: Path):
        """Saves cache to a JSON file."""
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"Cache saved to {cache_file}")
        except Exception as e:
            print(f"Error saving cache to {cache_file}: {e}")

    def initialize_caption_cache(self):
        """
        Load captions from cache file and scan directory for new captions.
        """
        print("Initializing caption cache...")
        self.image_captions = self.load_cache(self.CAPTIONS_CACHE_FILE)
        
        # Get current caption files in the directory
        current_caption_files = {file.stem.replace('_caption', '') for file in CAPTIONS_DIRECTORY.glob("*_caption.txt")} # Use a set for faster lookup

        # Find new caption files not in cache
        new_caption_ids = current_caption_files - set(self.image_captions.keys())

        for image_id in new_caption_ids:
            try:
                # Read caption from file for new files
                caption_file = CAPTIONS_DIRECTORY / f"{image_id}_caption.txt"
                if caption_file.exists():
                     with open(caption_file, 'r') as f:
                         caption = f.read().strip()
                         self.image_captions[image_id] = caption
            except Exception as e:
                print(f"Error processing new caption file {caption_file}: {e}")
                continue
                
        print(f"Caption cache initialized with {len(self.image_captions)} entries.")

    def initialize_dimension_cache(self):
        """
        Load dimensions from cache file and scan directory for new images.
        """
        print("Initializing dimension cache...")
        
        # Load existing cache
        # Note: JSON keys are strings, so tuple keys (width, height) need handling if you were saving that way.
        # Our current dimension cache uses imageId as key, which is already a string, so direct load works.
        loaded_cache = self.load_cache(self.DIMENSIONS_CACHE_FILE)
        # Convert list from JSON back to tuple if necessary (json saves tuples as lists)
        self.image_dimensions = {k: tuple(v) for k, v in loaded_cache.items()} if loaded_cache else {}
        
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
        cached_image_ids = set(self.image_dimensions.keys())
        new_image_ids = current_image_ids - cached_image_ids

        print(f"Found {len(new_image_ids)} new images not in dimension cache.")

        for img_file in all_image_files:
            image_id = str(img_file.stem)
            # Only process if it's a new image or its dimension is missing from cache
            if image_id in new_image_ids or image_id not in self.image_dimensions:
                 width, height = self.get_image_dimensions_from_file(img_file)
                 if width is not None and height is not None:
                     self.image_dimensions[image_id] = (width, height) # Cache the result

        print(f"Dimension cache initialized/updated with {len(self.image_dimensions)} entries.")

    def get_image_dimensions_from_file(self, image_path: Path) -> tuple[Optional[int], Optional[int]]:
        """Helper function to get image dimensions by opening the file."""
        try:
            with PILImage.open(image_path) as img:
                return img.size
        except Exception:
            return None, None

    def get_image_dimensions(self, image_id: str) -> tuple[Optional[int], Optional[int]]:
        """Get image dimensions from cache or file, and cache the result if read from file."""
        # This function now primarily serves as a getter from the in-memory cache
        # The caching from file is handled during initialization and when new images are added/processed via other endpoints if necessary.
        # However, the current logic in initialize_dimension_cache covers the main case.
        
        if image_id in self.image_dimensions:
            return self.image_dimensions[image_id]
        
        # Fallback: If somehow not in cache (shouldn't happen after proper initialization), read from file and add to cache
        print(f"Warning: Image dimensions for {image_id} not in cache. Reading from file.")
        image_path = self.get_image_path(image_id)
        if not image_path:
            return None, None
        
        width, height = self.get_image_dimensions_from_file(image_path)
        if width is not None and height is not None:
            self.image_dimensions[image_id] = (width, height) # Cache the result
            # Consider saving cache here too if you expect dimensions to be fetched outside of startup for new files

        return width, height

    def get_tags_from_file_for_image(self, image_id: str) -> List[str]:
        """Get image-specific metadata from the metadata file.
        
        Args:
            image_id: The ID of the image to get metadata for
            
        Returns:
            List[str]: A list of tags for the image, or an empty list if none exists
        """
        print(f"Debug: Reading tags from file for image {image_id}")
        print(f"Debug: Current image_metadata: {self.image_metadata}")
        tags = self.image_metadata.get(image_id, {}).get("tags", [])
        print(f"Debug: Found tags: {tags}")
        return tags if isinstance(tags, list) else list(tags)

    def set_tags_in_file_for_image(self, image_id: str, tags: List[str]) -> None:
        """Set image-specific metadata in the metadata file.
        
        Args:
            image_id: The ID of the image to set metadata for
            tags: A list of tags to add
        """
        # Initialize metadata structure if it doesn't exist
        if image_id not in self.image_metadata:
            self.image_metadata[image_id] = {}
        
        # Get current tags and ensure they're in a list
        current_tags = self.image_metadata[image_id].get("tags", [])
        if isinstance(current_tags, set):
            current_tags = list(current_tags)
        
        # Add new tags and deduplicate
        all_tags = set(current_tags + tags)
        
        # Update metadata with deduplicated list
        self.image_metadata[image_id]["tags"] = list(all_tags)
        
        # Save back to file
        self.save_cache(self.image_metadata, self.IMAGE_METADATA_FILE)

    def get_tags_for_image(self, image_id: str) -> List[str]:
        """Get combined metadata for an image, including both file-specific and scene metadata.
        
        Args:
            image_id: The ID of the image to get combined metadata for
            
        Returns:
            List[str]: A list of deduplicated tags for the image
        """
        print(f"\nDebug: get_tags_for_image called for image_id: {image_id}")
        
        # Get image-specific tags
        print("Debug: Getting image-specific tags...")
        image_tags = set(self.get_tags_from_file_for_image(image_id))
        print(f"Debug: Image-specific tags: {image_tags}")
        
        # Get scene metadata
        print("Debug: Finding base name for scene metadata...")
        base_name = self.find_base_name(image_id)
        print(f"Debug: Found base_name: {base_name}")
        
        scene_metadata = self.metadata_cache['scene_metadata'].get(base_name, {
            'actors': [],
            'tags': [],
            'year': None
        })
        print(f"Debug: Scene metadata: {scene_metadata}")
        
        # Combine and deduplicate tags
        print("Debug: Combining and deduplicating tags...")
        scene_tags = set(scene_metadata["tags"])
        print(f"Debug: Scene tags: {scene_tags}")
        
        all_tags = scene_tags.union(image_tags)
        combined_tags = list(all_tags)
        print(f"Debug: Combined tags: {combined_tags}")
        
        return combined_tags

    def save_all_caches(self):
        """Saves all caches to files."""
        print("Saving caches...")
        self.save_cache(self.image_captions, self.CAPTIONS_CACHE_FILE)
        # Convert tuples in dimension cache to lists for JSON serialization
        dimensions_to_save = {k: list(v) for k, v in self.image_dimensions.items()}
        self.save_cache(dimensions_to_save, self.DIMENSIONS_CACHE_FILE)
        print("Caches saved.")

    def initialize(self):
        """Initialize all caches and build image file list."""
        print("Loading caches on startup...")
        
        self.read_photoset_metadata()
        print(f"Photoset metadata cache initialized with {len(self.metadata_cache['scenes'])} scenes")
        
        self.initialize_caption_cache()
        crop_cache.initialize()
        self.initialize_dimension_cache()
        
        print("Building cached image file list...")
        self.cached_image_files = [
            f for f in IMAGES_DIR.glob("*") 
            if f.is_file() 
            and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            and "_crop_" not in f.name
            and not f.name.startswith('.')
        ]
        self.cached_image_files.sort()
        print(f"Cached image file list built with {len(self.cached_image_files)} files.")

        print(f"Initializing image metadata cache")
        self.image_metadata = self.load_cache(self.IMAGE_METADATA_FILE)

        print(f"Image metadata cache initialized with {len(self.image_metadata)} entries")
        
        print("Caches loaded and image list built.") 

# Singleton
filter_manager = FilterManager()