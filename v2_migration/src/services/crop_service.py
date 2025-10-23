"""Crop service for managing image crops."""
import io
from typing import Optional
from uuid import UUID
from PIL import Image as PILImage

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import NotFoundException, ImageProcessingException
from src.models.database import Crop
from src.repositories import CropRepository, ImageRepository
from src.services.storage_service import StorageService
from src.services.cache_service import CacheService


class CropService:
    """Service for crop operations."""
    
    def __init__(
        self,
        db: AsyncSession,
        storage: StorageService,
        cache: CacheService
    ):
        """
        Initialize crop service.
        
        Args:
            db: Database session
            storage: Storage service
            cache: Cache service
        """
        self.db = db
        self.storage = storage
        self.cache = cache
        self.crop_repo = CropRepository(db)
        self.image_repo = ImageRepository(db)
    
    async def get_crop(self, image_id: UUID) -> Optional[Crop]:
        """
        Get crop for an image.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Crop model or None
        """
        return await self.crop_repo.get_by_image_id(image_id)
    
    async def get_crop_image(self, image_id: UUID) -> Optional[bytes]:
        """
        Get cropped image data.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Cropped image bytes or None
        """
        crop = await self.crop_repo.get_by_image_id(image_id)
        if not crop:
            return None
        
        crop_path = self.storage.crops_dir / crop.crop_file_path
        if not crop_path.exists():
            return None
        
        return self.storage.read_file(crop_path)
    
    async def create_crop(
        self,
        image_id: UUID,
        target_size: int,
        normalized_delta_x: float,
        normalized_delta_y: float
    ) -> tuple[Crop, bytes]:
        """
        Create or update a crop for an image.
        
        Args:
            image_id: Image UUID
            target_size: Target size in pixels
            normalized_delta_x: Normalized X delta (-1 to 1)
            normalized_delta_y: Normalized Y delta (-1 to 1)
            
        Returns:
            Tuple of (Crop model, cropped image bytes)
            
        Raises:
            NotFoundException: If image not found
            ImageProcessingException: If crop generation fails
        """
        # Verify image exists
        image = await self.image_repo.get_by_id_or_fail(image_id)
        image_path = self.storage.images_dir / image.file_path
        
        if not image_path.exists():
            raise NotFoundException("Image file", str(image_id))
        
        # Generate cropped image
        try:
            with PILImage.open(image_path) as img:
                # Resize image to fit within target size
                resized = self._resize_for_crop(img, target_size)
                
                # Calculate crop coordinates
                target_x = normalized_delta_x * resized.width
                target_y = normalized_delta_y * resized.height
                
                # Add slack to center the crop
                horizontal_slack = resized.width - target_size
                vertical_slack = resized.height - target_size
                
                target_x += horizontal_slack / 2
                target_y += vertical_slack / 2
                
                # Perform the crop
                cropped = resized.crop((
                    target_x,
                    target_y,
                    target_x + target_size,
                    target_y + target_size
                ))
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                cropped.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                image_data = img_byte_arr.getvalue()
        
        except Exception as e:
            raise ImageProcessingException(f"Failed to generate crop: {e}")
        
        # Save crop file
        crop_path = self.storage.get_crop_path(image_id)
        self.storage.save_file(image_data, crop_path)
        
        # Save crop metadata in database
        relative_path = crop_path.name  # Just the filename
        crop = await self.crop_repo.upsert(
            image_id=image_id,
            target_size=target_size,
            normalized_delta_x=normalized_delta_x,
            normalized_delta_y=normalized_delta_y,
            crop_file_path=relative_path
        )
        
        return crop, image_data
    
    async def delete_crop(self, image_id: UUID) -> bool:
        """
        Delete a crop.
        
        Args:
            image_id: Image UUID
            
        Returns:
            True if deleted
        """
        crop = await self.crop_repo.get_by_image_id(image_id)
        if not crop:
            return False
        
        # Delete crop file
        crop_path = self.storage.crops_dir / crop.crop_file_path
        self.storage.delete_file(crop_path)
        
        # Delete from database
        return await self.crop_repo.delete_by_image_id(image_id)
    
    async def get_preview(
        self,
        image_id: UUID,
        target_size: int
    ) -> bytes:
        """
        Get a preview of the image resized for cropping.
        
        Args:
            image_id: Image UUID
            target_size: Target size in pixels
            
        Returns:
            Resized image bytes
            
        Raises:
            NotFoundException: If image not found
        """
        image = await self.image_repo.get_by_id_or_fail(image_id)
        image_path = self.storage.images_dir / image.file_path
        
        if not image_path.exists():
            raise NotFoundException("Image file", str(image_id))
        
        try:
            with PILImage.open(image_path) as img:
                resized = self._resize_for_crop(img, target_size)
                
                # Convert to bytes
                output = io.BytesIO()
                resized.save(output, format='PNG')
                output.seek(0)
                return output.getvalue()
        
        except Exception as e:
            raise ImageProcessingException(f"Failed to generate preview: {e}")
    
    def _resize_for_crop(self, img: PILImage.Image, target_size: int) -> PILImage.Image:
        """
        Resize image so the smaller dimension fits the target size.
        
        Args:
            img: PIL Image
            target_size: Target size in pixels
            
        Returns:
            Resized PIL Image
        """
        width, height = img.size
        scale = max(target_size / width, target_size / height)
        new_size = (int(width * scale), int(height * scale))
        
        return img.resize(new_size, PILImage.Resampling.LANCZOS)


