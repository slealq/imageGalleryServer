import json
from pathlib import Path
from typing import Dict, Optional, Tuple, BinaryIO
from PIL import Image as PILImage
import io
from config import CROPS_DIR

class CropCache:
    """A class to manage the crop cache for images."""
    
    def __init__(self):
        self._crops_info: Dict[str, dict] = {}  # Store crop info: {imageId: {"targetSize": int, "normalizedDeltas": {"x": float, "y": float}}}
        self._crop_images_data: Dict[str, bytes] = {}
    
    @property
    def crops(self) -> Dict[str, dict]:
        """Get the current crop cache."""
        return self._crops_info
    
    def get_crop_metadata(self, image_id: str) -> Optional[dict]:
        """Get crop metadata for a specific image."""
        return self._crops_info.get(image_id)
    
    def set_crop_metadata(self, image_id: str, crop_info: dict):
        """Set crop metadata for a specific image."""
        self._crops_info[image_id] = crop_info
    
    def has_crop_metadata(self, image_id: str) -> bool:
        """Check if an image has crop information."""
        return image_id in self._crops_info
    
    def remove_crop(self, image_id: str):
        """Remove crop information and image for a specific image."""
        if image_id in self._crops_info:
            # Remove metadata
            del self._crops_info[image_id]
            
            # Remove image file if it exists
            crop_info = self._crops_info.get(image_id)
            if crop_info:
                image_path = self._get_crop_image_path(image_id, crop_info['targetSize'])
                if image_path.exists():
                    image_path.unlink()
    
    def get_crop_image_path(self, image_id: str) -> Optional[Path]:
        """Get the path to the cropped image file."""
        crop_info = self.get_crop_metadata(image_id)
        if not crop_info:
            return None
        return self._get_crop_image_path(image_id, crop_info['targetSize'])
    
    def _get_crop_image_path(self, image_id: str, target_size: int) -> Path:
        """Get the path to the cropped image file for a specific target size."""
        return CROPS_DIR / f"{image_id}_crop_{target_size}.png"
    
    def get_crop_image(self, image_id: str) -> Optional[bytes]:
        """Get the cropped image data."""

        if image_id in self._crop_images_data:
            return self._crop_images_data[image_id]

        image_path = self.get_crop_image_path(image_id)
        if not image_path or not image_path.exists():
            return None
            
        try:
            with open(image_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading crop image for {image_id}: {e}")
            return None
    
    def write_crop_image(self, image_id: str, image_data: bytes) -> bool:
        """Set the cropped image data."""
        crop_info = self.get_crop_metadata(image_id)
        if not crop_info:
            return False
        
        self._crop_images_data[image_id] = image_data
            
        try:
            image_path = self._get_crop_image_path(image_id, crop_info['targetSize'])
            with open(image_path, 'wb') as f:
                f.write(image_data)
            return True
        except Exception as e:
            print(f"Error saving crop image for {image_id}: {e}")
            return False
    
    def save_crop(self, image_id: str, crop_info: dict, image_data: bytes) -> bool:
        """Save both crop metadata and image data."""
        try:
            # Save metadata
            self.set_crop_metadata(image_id, crop_info)
            
            # Save image
            if not self.write_crop_image(image_id, image_data):
                return False
                
            # Save metadata to JSON file
            metadata_path = CROPS_DIR / f"{image_id}_crop_{crop_info['targetSize']}.json"
            with open(metadata_path, 'w') as f:
                json.dump(crop_info, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving crop for {image_id}: {e}")
            return False
    
    def initialize(self):
        """
        Initialize the crop cache by loading from file and scanning directory for new crop info.
        """
        print("Initializing crop cache...")

        # Try to load metadata from JSON file for new files
        print(f"Crops dir is: {CROPS_DIR}")
        metadata_files = list(CROPS_DIR.glob("*_crop_*.json"))

        print(f"I have {len(metadata_files)} crop metadata files to load")

        if metadata_files:
            for metadata_path in metadata_files:
                try:
                    # Extract image_id from filename by removing everything after _crop_
                    image_id = metadata_path.stem.split('_crop_')[0]
                    
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self._crops_info[image_id] = {
                            "targetSize": metadata.get("targetSize"),
                            "normalizedDeltas": metadata.get("normalizedDeltas")
                        }
                except Exception as e:
                    print(f"Error processing crop file {metadata_path}: {e}")
                    continue
                
        print(f"Crop cache initialized with {len(self._crops_info)} entries.")

# Create a singleton instance
crop_cache = CropCache() 