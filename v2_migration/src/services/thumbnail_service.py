"""Thumbnail service for generating and managing image thumbnails."""
import io
from typing import Optional, Dict, Tuple
from uuid import UUID
from PIL import Image as PILImage

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.exceptions import ImageProcessingException
from src.models.database import Thumbnail
from src.repositories import ImageRepository
from src.services.storage_service import StorageService
from src.services.cache_service import CacheService


class ThumbnailService:
    """Service for thumbnail operations."""
    
    def __init__(
        self,
        db: AsyncSession,
        storage: StorageService,
        cache: CacheService
    ):
        """
        Initialize thumbnail service.
        
        Args:
            db: Database session
            storage: Storage service
            cache: Cache service
        """
        self.db = db
        self.storage = storage
        self.cache = cache
        self.image_repo = ImageRepository(db)
        
        # Get thumbnail sizes from config
        self.sizes = {
            "small": settings.thumbnail_small_size,
            "medium": settings.thumbnail_medium_size,
            "large": settings.thumbnail_large_size,
        }
    
    async def generate_thumbnail(
        self,
        image_id: UUID,
        size_name: str,
        force: bool = False
    ) -> Thumbnail:
        """
        Generate a thumbnail for an image.
        
        Args:
            image_id: Image UUID
            size_name: Size name (small, medium, large)
            force: Force regeneration even if exists
            
        Returns:
            Thumbnail model
            
        Raises:
            ImageProcessingException: If generation fails
        """
        # Validate size
        if size_name not in self.sizes:
            raise ImageProcessingException(
                f"Invalid thumbnail size: {size_name}. "
                f"Valid sizes: {', '.join(self.sizes.keys())}"
            )
        
        target_size = self.sizes[size_name]
        
        # Check if thumbnail already exists
        if not force:
            existing = await self._get_existing_thumbnail(image_id, size_name)
            if existing:
                return existing
        
        # Get original image
        image = await self.image_repo.get_by_id_or_fail(image_id)
        image_path = self.storage.images_dir / image.file_path
        
        if not image_path.exists():
            raise ImageProcessingException(f"Original image not found: {image_path}")
        
        # Generate thumbnail
        try:
            with PILImage.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')
                
                # Calculate dimensions maintaining aspect ratio
                thumb_width, thumb_height = self._calculate_thumbnail_size(
                    img.size,
                    target_size
                )
                
                # Resize
                img.thumbnail((thumb_width, thumb_height), PILImage.Resampling.LANCZOS)
                
                # Save to storage
                thumb_path = self.storage.get_thumbnail_path(image_id, size_name)
                img.save(thumb_path, 'JPEG', quality=85, optimize=True)
                
                # Save thumbnail bytes to cache
                with open(thumb_path, 'rb') as f:
                    thumb_data = f.read()
                await self.cache.set_thumbnail(image_id, size_name, thumb_data)
        
        except Exception as e:
            raise ImageProcessingException(f"Failed to generate thumbnail: {e}")
        
        # Create or update thumbnail record
        thumbnail = Thumbnail(
            image_id=image_id,
            size_name=size_name,
            width=thumb_width,
            height=thumb_height,
            file_path=str(thumb_path.relative_to(self.storage.thumbnails_dir))
        )
        
        self.db.add(thumbnail)
        await self.db.flush()
        await self.db.refresh(thumbnail)
        
        return thumbnail
    
    async def generate_all_thumbnails(self, image_id: UUID) -> Dict[str, Thumbnail]:
        """
        Generate all thumbnail sizes for an image.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Dictionary mapping size names to thumbnails
        """
        thumbnails = {}
        for size_name in self.sizes.keys():
            try:
                thumbnail = await self.generate_thumbnail(image_id, size_name)
                thumbnails[size_name] = thumbnail
            except Exception as e:
                print(f"Failed to generate {size_name} thumbnail for {image_id}: {e}")
        
        return thumbnails
    
    async def get_thumbnail(
        self,
        image_id: UUID,
        size_name: str,
        generate_if_missing: bool = True
    ) -> Optional[bytes]:
        """
        Get thumbnail data.
        
        Args:
            image_id: Image UUID
            size_name: Size name
            generate_if_missing: Generate if doesn't exist
            
        Returns:
            Thumbnail bytes or None
        """
        # Try cache first
        cached = await self.cache.get_thumbnail(image_id, size_name)
        if cached:
            return cached
        
        # Try storage
        thumb_path = self.storage.get_thumbnail_path(image_id, size_name)
        if thumb_path.exists():
            thumb_data = self.storage.read_file(thumb_path)
            # Cache for next time
            await self.cache.set_thumbnail(image_id, size_name, thumb_data)
            return thumb_data
        
        # Generate if requested
        if generate_if_missing:
            await self.generate_thumbnail(image_id, size_name)
            # Read the newly generated thumbnail
            if thumb_path.exists():
                return self.storage.read_file(thumb_path)
        
        return None
    
    async def delete_thumbnails(self, image_id: UUID):
        """
        Delete all thumbnails for an image.
        
        Args:
            image_id: Image UUID
        """
        # Delete from cache
        for size_name in self.sizes.keys():
            await self.cache.delete_metadata(f"thumbnail:{image_id}:{size_name}")
        
        # Delete from storage
        thumb_dir = self.storage.thumbnails_dir / str(image_id)
        self.storage.delete_directory(thumb_dir)
        
        # Delete from database (cascade should handle this, but explicit is better)
        from src.models.database import Thumbnail
        from sqlalchemy import delete
        
        await self.db.execute(
            delete(Thumbnail).where(Thumbnail.image_id == image_id)
        )
        await self.db.flush()
    
    def _calculate_thumbnail_size(
        self,
        original_size: Tuple[int, int],
        target_size: int
    ) -> Tuple[int, int]:
        """
        Calculate thumbnail dimensions maintaining aspect ratio.
        
        Args:
            original_size: (width, height) of original
            target_size: Target max dimension
            
        Returns:
            (width, height) for thumbnail
        """
        width, height = original_size
        
        if width > height:
            new_width = target_size
            new_height = int((target_size / width) * height)
        else:
            new_height = target_size
            new_width = int((target_size / height) * width)
        
        return new_width, new_height
    
    async def _get_existing_thumbnail(
        self,
        image_id: UUID,
        size_name: str
    ) -> Optional[Thumbnail]:
        """Get existing thumbnail from database."""
        from sqlalchemy import select
        
        result = await self.db.execute(
            select(Thumbnail).where(
                Thumbnail.image_id == image_id,
                Thumbnail.size_name == size_name
            )
        )
        return result.scalar_one_or_none()

