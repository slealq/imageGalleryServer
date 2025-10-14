"""Image service for managing images."""
import io
from pathlib import Path
from typing import List, Optional, BinaryIO, Tuple
from uuid import UUID
from PIL import Image as PILImage

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import NotFoundException, ImageProcessingException
from src.models.database import Image
from src.models.schemas.common import FilterParams, PaginationParams
from src.repositories import ImageRepository
from src.services.storage_service import StorageService
from src.services.cache_service import CacheService
from src.services.thumbnail_service import ThumbnailService


class ImageService:
    """Service for image operations."""
    
    def __init__(
        self,
        db: AsyncSession,
        storage: StorageService,
        cache: CacheService
    ):
        """
        Initialize image service.
        
        Args:
            db: Database session
            storage: Storage service
            cache: Cache service
        """
        self.db = db
        self.storage = storage
        self.cache = cache
        self.repo = ImageRepository(db)
        self.thumbnail_service = ThumbnailService(db, storage, cache)
    
    async def get_image(self, image_id: UUID) -> Image:
        """
        Get image by ID.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Image model
            
        Raises:
            NotFoundException: If image not found
        """
        return await self.repo.get_by_id_or_fail(image_id)
    
    async def get_image_data(self, image_id: UUID, use_cache: bool = True) -> bytes:
        """
        Get image file data.
        
        Args:
            image_id: Image UUID
            use_cache: Whether to use cache
            
        Returns:
            Image bytes
            
        Raises:
            NotFoundException: If image not found
        """
        # Try cache first
        if use_cache:
            cached = await self.cache.get_image(image_id)
            if cached:
                return cached
        
        # Get from storage
        image = await self.repo.get_by_id_or_fail(image_id)
        image_path = self.storage.images_dir / image.file_path
        
        if not image_path.exists():
            raise NotFoundException("Image file", str(image_id))
        
        # Read and optionally optimize
        with PILImage.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Save as optimized JPEG
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            output.seek(0)
            image_data = output.getvalue()
        
        # Cache for next time
        if use_cache:
            await self.cache.set_image(image_id, image_data)
        
        return image_data
    
    async def list_images(
        self,
        pagination: PaginationParams,
        filters: Optional[FilterParams] = None
    ) -> tuple[List[Image], int]:
        """
        List images with pagination and filters.
        
        Args:
            pagination: Pagination parameters
            filters: Filter parameters
            
        Returns:
            Tuple of (images list, total count)
        """
        # Build query based on filters
        skip = (pagination.page - 1) * pagination.page_size
        limit = pagination.page_size
        
        # Apply filters
        if filters:
            if filters.photoset_id:
                images = await self.repo.get_by_photoset(
                    filters.photoset_id,
                    skip=skip,
                    limit=limit
                )
                total = await self.repo.count_by_photoset(filters.photoset_id)
            elif filters.has_caption is not None:
                images = await self.repo.get_with_caption_filter(
                    filters.has_caption,
                    skip=skip,
                    limit=limit
                )
                total = len(images)  # TODO: Implement count method
            elif filters.has_crop is not None:
                images = await self.repo.get_with_crop_filter(
                    filters.has_crop,
                    skip=skip,
                    limit=limit
                )
                total = len(images)  # TODO: Implement count method
            elif filters.tag:
                images = await self.repo.get_by_tag(
                    filters.tag,
                    skip=skip,
                    limit=limit
                )
                total = len(images)  # TODO: Implement count method
            else:
                images = await self.repo.get_all(skip=skip, limit=limit)
                total = await self.repo.count()
        else:
            images = await self.repo.get_all(skip=skip, limit=limit)
            total = await self.repo.count()
        
        return images, total
    
    async def create_image(
        self,
        photoset_id: Optional[UUID],
        filename: str,
        file_data: bytes,
        mime_type: Optional[str] = None
    ) -> Image:
        """
        Create a new image.
        
        Args:
            photoset_id: Photoset UUID (optional)
            filename: Original filename
            file_data: Image binary data
            mime_type: MIME type
            
        Returns:
            Created image model
            
        Raises:
            ImageProcessingException: If image processing fails
        """
        try:
            # Get image dimensions
            img = PILImage.open(io.BytesIO(file_data))
            width, height = img.size
            img.close()
        except Exception as e:
            raise ImageProcessingException(f"Invalid image data: {e}")
        
        # Determine storage path
        if photoset_id:
            file_path = self.storage.get_image_path(photoset_id, filename)
            relative_path = str(file_path.relative_to(self.storage.images_dir))
        else:
            # Store in root for orphaned images
            file_path = self.storage.images_dir / filename
            relative_path = filename
        
        # Save file
        self.storage.save_file(file_data, file_path)
        
        # Create database record
        image = Image(
            photoset_id=photoset_id,
            original_filename=filename,
            file_path=relative_path,
            width=width,
            height=height,
            file_size=len(file_data),
            mime_type=mime_type or "image/jpeg"
        )
        
        return await self.repo.create(image)
    
    async def upload_image(
        self,
        photoset_id: Optional[UUID],
        file: BinaryIO,
        filename: str
    ) -> Image:
        """
        Upload an image file.
        
        Args:
            photoset_id: Photoset UUID (optional)
            file: Uploaded file
            filename: Original filename
            
        Returns:
            Created image model
        """
        # Read file data
        file_data = file.read()
        
        return await self.create_image(photoset_id, filename, file_data)
    
    async def delete_image(self, image_id: UUID) -> bool:
        """
        Delete an image and all associated data.
        
        Args:
            image_id: Image UUID
            
        Returns:
            True if deleted
        """
        image = await self.repo.get_by_id(image_id)
        if not image:
            return False
        
        # Delete file from storage
        image_path = self.storage.images_dir / image.file_path
        self.storage.delete_file(image_path)
        
        # Invalidate cache
        await self.cache.invalidate_image_cache(image_id)
        
        # Delete from database (cascade will handle related records)
        return await self.repo.delete(image_id)
    
    async def update_image_metadata(
        self,
        image_id: UUID,
        updates: dict
    ) -> Image:
        """
        Update image metadata.
        
        Args:
            image_id: Image UUID
            updates: Dictionary of fields to update
            
        Returns:
            Updated image model
            
        Raises:
            NotFoundException: If image not found
        """
        updated = await self.repo.update(image_id, updates)
        if not updated:
            raise NotFoundException("Image", str(image_id))
        
        # Invalidate cache
        await self.cache.delete_metadata(f"image:{image_id}")
        
        return updated
    
    async def get_images(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[Image]:
        """
        Get paginated list of images.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of images
        """
        return await self.repo.get_all(skip=skip, limit=limit)
    
    async def get_images_by_photoset(
        self,
        photoset_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[Image]:
        """
        Get images in a photoset.
        
        Args:
            photoset_id: Photoset UUID
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of images
        """
        return await self.repo.get_by_photoset(photoset_id, skip=skip, limit=limit)
    
    async def count_images(self) -> int:
        """Get total number of images."""
        return await self.repo.count()
    
    async def count_images_by_photoset(self, photoset_id: UUID) -> int:
        """Get number of images in a photoset."""
        return await self.repo.count_by_photoset(photoset_id)
    
    async def get_thumbnail_data(
        self,
        image_id: UUID,
        generate_if_missing: bool = True
    ) -> Optional[bytes]:
        """
        Get thumbnail data for an image.
        
        Args:
            image_id: Image UUID
            generate_if_missing: Generate thumbnail if it doesn't exist
            
        Returns:
            Thumbnail bytes or None
        """
        return await self.thumbnail_service.get_thumbnail_data(image_id)

