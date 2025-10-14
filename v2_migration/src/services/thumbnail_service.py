"""Thumbnail service for generating and managing image thumbnails."""
import io
from typing import Optional, Tuple
from uuid import UUID
from PIL import Image as PILImage
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from src.core.config import settings
from src.core.exceptions import ImageProcessingException
from src.models.database import Thumbnail
from src.repositories import ImageRepository
from src.services.storage_service import StorageService
from src.services.cache_service import CacheService


class ThumbnailService:
    """Service for thumbnail operations - generates scaled-down versions maintaining aspect ratio."""
    
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
        
        # Thumbnail configuration
        self.max_dimension = settings.thumbnail_max_dimension
        self.quality = settings.thumbnail_quality
    
    async def generate_thumbnail(
        self,
        image_id: UUID,
        force: bool = False
    ) -> Thumbnail:
        """
        Generate a scaled-down thumbnail for an image.
        Maintains original aspect ratio, reduces quality for faster loading.
        
        Args:
            image_id: Image UUID
            force: Force regeneration even if exists
            
        Returns:
            Thumbnail model
            
        Raises:
            ImageProcessingException: If generation fails
        """
        # Check if thumbnail already exists
        if not force:
            existing = await self._get_existing_thumbnail(image_id)
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
                
                original_width, original_height = img.size
                
                # Calculate scaled dimensions maintaining aspect ratio
                thumb_width, thumb_height = self._calculate_scaled_size(
                    original_width,
                    original_height
                )
                
                # Resize with high-quality downsampling
                img = img.resize(
                    (thumb_width, thumb_height),
                    PILImage.Resampling.LANCZOS
                )
                
                # Save to storage
                thumb_path = self.storage.get_thumbnail_path(image_id)
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save with reduced quality for faster loading
                img.save(
                    thumb_path,
                    'JPEG',
                    quality=self.quality,
                    optimize=True,
                    progressive=True  # Progressive JPEGs load faster
                )
                
                # Cache thumbnail bytes
                with open(thumb_path, 'rb') as f:
                    thumb_data = f.read()
                await self.cache.set_thumbnail(image_id, thumb_data)
        
        except Exception as e:
            raise ImageProcessingException(f"Failed to generate thumbnail: {e}")
        
        # Create or update thumbnail record
        # Delete existing thumbnail record if forcing regeneration
        if force:
            await self.db.execute(
                delete(Thumbnail).where(Thumbnail.image_id == image_id)
            )
        
        thumbnail = Thumbnail(
            image_id=image_id,
            size_name="preview",  # Single size for browsing
            width=thumb_width,
            height=thumb_height,
            file_path=str(thumb_path.relative_to(self.storage.thumbnails_dir))
        )
        
        self.db.add(thumbnail)
        await self.db.flush()
        await self.db.refresh(thumbnail)
        
        return thumbnail
    
    async def get_thumbnail(
        self,
        image_id: UUID,
        generate_if_missing: bool = True
    ) -> Optional[Thumbnail]:
        """
        Get thumbnail for an image, optionally generating if missing.
        
        Args:
            image_id: Image UUID
            generate_if_missing: Generate thumbnail if it doesn't exist
            
        Returns:
            Thumbnail model or None
        """
        # Check database first
        thumbnail = await self._get_existing_thumbnail(image_id)
        
        if thumbnail:
            return thumbnail
        
        if generate_if_missing:
            return await self.generate_thumbnail(image_id)
        
        return None
    
    async def get_thumbnail_data(
        self,
        image_id: UUID
    ) -> Optional[bytes]:
        """
        Get thumbnail image data (bytes).
        
        Args:
            image_id: Image UUID
            
        Returns:
            Thumbnail image bytes or None
        """
        # Try cache first
        cached_data = await self.cache.get_thumbnail(image_id)
        if cached_data:
            return cached_data
        
        # Get from database/storage
        thumbnail = await self.get_thumbnail(image_id)
        if not thumbnail:
            return None
        
        # Read from disk
        thumb_path = self.storage.thumbnails_dir / thumbnail.file_path
        if not thumb_path.exists():
            return None
        
        with open(thumb_path, 'rb') as f:
            data = f.read()
        
        # Update cache
        await self.cache.set_thumbnail(image_id, data)
        
        return data
    
    async def delete_thumbnail(self, image_id: UUID) -> bool:
        """
        Delete thumbnail for an image.
        
        Args:
            image_id: Image UUID
            
        Returns:
            True if deleted, False if not found
        """
        thumbnail = await self._get_existing_thumbnail(image_id)
        
        if not thumbnail:
            return False
        
        # Delete file
        thumb_path = self.storage.thumbnails_dir / thumbnail.file_path
        if thumb_path.exists():
            thumb_path.unlink()
        
        # Delete from database
        await self.db.delete(thumbnail)
        await self.db.flush()
        
        # Clear cache
        await self.cache.delete_thumbnail(image_id)
        
        return True
    
    def _calculate_scaled_size(
        self,
        original_width: int,
        original_height: int
    ) -> Tuple[int, int]:
        """
        Calculate thumbnail dimensions maintaining aspect ratio.
        Scales down so the longest dimension equals max_dimension.
        
        Args:
            original_width: Original image width
            original_height: Original image height
            
        Returns:
            (width, height) for thumbnail
        """
        # If image is already smaller than max dimension, keep original size
        if original_width <= self.max_dimension and original_height <= self.max_dimension:
            return (original_width, original_height)
        
        # Calculate scale factor based on longest dimension
        if original_width > original_height:
            scale_factor = self.max_dimension / original_width
        else:
            scale_factor = self.max_dimension / original_height
        
        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        return (new_width, new_height)
    
    async def _get_existing_thumbnail(self, image_id: UUID) -> Optional[Thumbnail]:
        """Get existing thumbnail from database."""
        result = await self.db.execute(
            select(Thumbnail).where(Thumbnail.image_id == image_id)
        )
        return result.scalar_one_or_none()
