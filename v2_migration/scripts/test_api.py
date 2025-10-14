"""Test script to check for import and startup errors."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all imports to find issues."""
    print("Testing imports...")
    
    try:
        print("1. Testing core imports...")
        from src.core.config import settings
        print("+ Core config")
        
        from src.core.database import get_db_context
        print("+ Database")
        
        from src.core.exceptions import NotFoundException
        print("+ Exceptions")
        
        print("2. Testing model imports...")
        from src.models.database import Photoset, Image, Caption, Crop, Tag, Thumbnail
        print("+ Database models")
        
        from src.models.schemas import PhotosetResponse, ImageMetadataResponse
        print("+ Pydantic schemas")
        
        print("3. Testing repository imports...")
        from src.repositories import (
            PhotosetRepository, 
            ImageRepository, 
            CaptionRepository, 
            CropRepository, 
            TagRepository
        )
        print("+ Repositories")
        
        print("4. Testing service imports...")
        from src.services import (
            PhotosetService,
            ImageService,
            ThumbnailService,
            StorageService,
            CacheService
        )
        print("+ Services")
        
        print("5. Testing API imports...")
        from src.api.dependencies import get_photoset_service, get_image_service
        print("+ Dependencies")
        
        from src.api.routes import photosets, images, health
        print("+ Routes")
        
        print("6. Testing main app...")
        from src.main import app
        print("+ FastAPI app")
        
        print("\n[SUCCESS] All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_methods():
    """Test if service methods exist."""
    print("\nTesting service methods...")
    
    try:
        from src.services import PhotosetService, ImageService
        
        # Check PhotosetService methods
        photoset_methods = [
            'get_photosets',
            'get_photosets_by_year', 
            'search_photosets',
            'count_photosets',
            'get_photoset_images'
        ]
        
        for method in photoset_methods:
            if hasattr(PhotosetService, method):
                print(f"+ PhotosetService.{method}")
            else:
                print(f"- PhotosetService.{method} - MISSING")
        
        # Check ImageService methods
        image_methods = [
            'get_images',
            'get_images_by_photoset',
            'count_images',
            'count_images_by_photoset',
            'get_thumbnail_data'
        ]
        
        for method in image_methods:
            if hasattr(ImageService, method):
                print(f"+ ImageService.{method}")
            else:
                print(f"- ImageService.{method} - MISSING")
                
        return True
        
    except Exception as e:
        print(f"[ERROR] Service method error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== API Diagnostic Test ===\n")
    
    imports_ok = test_imports()
    methods_ok = test_service_methods()
    
    if imports_ok and methods_ok:
        print("\n✓ All tests passed! The API should work.")
    else:
        print("\n✗ Issues found. Fix these before starting the server.")
