"""Factory Boy factories for creating test data."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from src.models.database import Photoset, Image, Caption, Crop, Tag


class PhotosetFactory:
    """Factory for creating Photoset instances."""
    
    @staticmethod
    def create(**kwargs: Any) -> Photoset:
        """Create a Photoset with default values."""
        defaults = {
            "id": uuid.uuid4(),
            "name": "Test Photoset",
            "year": 2024,
            "source_url": "https://example.com/test",
            "extra_metadata": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        defaults.update(kwargs)
        return Photoset(**defaults)


class ImageFactory:
    """Factory for creating Image instances."""
    
    @staticmethod
    def create(**kwargs: Any) -> Image:
        """Create an Image with default values."""
        # Generate unique file path if not provided
        image_id = kwargs.get("id", uuid.uuid4())
        defaults = {
            "id": image_id,
            "original_filename": f"test_image_{image_id}.jpg",
            "file_path": f"test/test_image_{image_id}.jpg",  # Unique path
            "width": 1920,
            "height": 1080,
            "file_size": 2458624,
            "mime_type": "image/jpeg",
            "extra_metadata": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        defaults.update(kwargs)
        return Image(**defaults)


class CaptionFactory:
    """Factory for creating Caption instances."""
    
    @staticmethod
    def create(**kwargs: Any) -> Caption:
        """Create a Caption with default values."""
        defaults = {
            "id": uuid.uuid4(),
            "caption": "A test caption",
            "generator_type": "test",
            "generator_metadata": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        defaults.update(kwargs)
        return Caption(**defaults)


class CropFactory:
    """Factory for creating Crop instances."""
    
    @staticmethod
    def create(**kwargs: Any) -> Crop:
        """Create a Crop with default values."""
        defaults = {
            "id": uuid.uuid4(),
            "target_size": 512,
            "normalized_delta_x": 0.0,
            "normalized_delta_y": 0.0,
            "crop_file_path": "test_crop.png",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        defaults.update(kwargs)
        return Crop(**defaults)


class TagFactory:
    """Factory for creating Tag instances."""
    
    @staticmethod
    def create(**kwargs: Any) -> Tag:
        """Create a Tag with default values."""
        defaults = {
            "id": uuid.uuid4(),
            "name": "test-tag",
            "tag_type": "custom",
            "created_at": datetime.utcnow(),
        }
        defaults.update(kwargs)
        return Tag(**defaults)

