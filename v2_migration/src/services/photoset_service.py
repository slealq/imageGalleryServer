"""Photoset service for managing photosets."""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import NotFoundException
from src.models.database import Photoset, Image
from src.models.schemas.common import PaginationParams
from src.repositories import PhotosetRepository, ImageRepository
from src.services.storage_service import StorageService


class PhotosetService:
    """Service for photoset operations."""
    
    def __init__(
        self,
        db: AsyncSession,
        storage: StorageService
    ):
        """
        Initialize photoset service.
        
        Args:
            db: Database session
            storage: Storage service
        """
        self.db = db
        self.storage = storage
        self.repo = PhotosetRepository(db)
        self.image_repo = ImageRepository(db)
    
    async def get_photoset(self, photoset_id: UUID) -> Photoset:
        """
        Get photoset by ID.
        
        Args:
            photoset_id: Photoset UUID
            
        Returns:
            Photoset model
            
        Raises:
            NotFoundException: If photoset not found
        """
        return await self.repo.get_by_id_or_fail(photoset_id)
    
    async def get_photoset_with_images(self, photoset_id: UUID) -> Photoset:
        """
        Get photoset with all images loaded.
        
        Args:
            photoset_id: Photoset UUID
            
        Returns:
            Photoset with images
            
        Raises:
            NotFoundException: If photoset not found
        """
        photoset = await self.repo.get_with_images(photoset_id)
        if not photoset:
            raise NotFoundException("Photoset", str(photoset_id))
        return photoset
    
    async def list_photosets(
        self,
        pagination: PaginationParams,
        year: Optional[int] = None
    ) -> tuple[List[Photoset], int]:
        """
        List photosets with pagination.
        
        Args:
            pagination: Pagination parameters
            year: Filter by year (optional)
            
        Returns:
            Tuple of (photosets list, total count)
        """
        skip = (pagination.page - 1) * pagination.page_size
        limit = pagination.page_size
        
        if year:
            photosets = await self.repo.get_by_year(year, skip=skip, limit=limit)
            # TODO: Add count method for filtered queries
            total = len(photosets)
        else:
            photosets = await self.repo.get_all(skip=skip, limit=limit)
            total = await self.repo.count()
        
        return photosets, total
    
    async def create_photoset(
        self,
        name: str,
        source_url: Optional[str] = None,
        year: Optional[int] = None,
        **kwargs
    ) -> Photoset:
        """
        Create a new photoset.
        
        Args:
            name: Photoset name
            source_url: Source URL
            year: Year
            **kwargs: Additional fields
            
        Returns:
            Created photoset model
        """
        photoset = Photoset(
            name=name,
            source_url=source_url,
            year=year,
            **kwargs
        )
        
        return await self.repo.create(photoset)
    
    async def update_photoset(
        self,
        photoset_id: UUID,
        updates: dict
    ) -> Photoset:
        """
        Update photoset metadata.
        
        Args:
            photoset_id: Photoset UUID
            updates: Dictionary of fields to update
            
        Returns:
            Updated photoset model
            
        Raises:
            NotFoundException: If photoset not found
        """
        updated = await self.repo.update(photoset_id, updates)
        if not updated:
            raise NotFoundException("Photoset", str(photoset_id))
        return updated
    
    async def delete_photoset(self, photoset_id: UUID) -> bool:
        """
        Delete a photoset and all associated images.
        
        Args:
            photoset_id: Photoset UUID
            
        Returns:
            True if deleted
        """
        photoset = await self.repo.get_by_id(photoset_id)
        if not photoset:
            return False
        
        # Delete photoset directory from storage
        photoset_dir = self.storage.images_dir / str(photoset_id)
        self.storage.delete_directory(photoset_dir)
        
        # Delete from database (cascade will handle images)
        return await self.repo.delete(photoset_id)
    
    async def get_image_count(self, photoset_id: UUID) -> int:
        """
        Get number of images in photoset.
        
        Args:
            photoset_id: Photoset UUID
            
        Returns:
            Image count
        """
        return await self.repo.get_image_count(photoset_id)
    
    async def search_photosets(
        self,
        name_pattern: str,
        pagination: PaginationParams
    ) -> tuple[List[Photoset], int]:
        """
        Search photosets by name.
        
        Args:
            name_pattern: Name pattern to search for
            pagination: Pagination parameters
            
        Returns:
            Tuple of (photosets list, total count)
        """
        skip = (pagination.page - 1) * pagination.page_size
        limit = pagination.page_size
        
        photosets = await self.repo.search_by_name(
            name_pattern,
            skip=skip,
            limit=limit
        )
        
        return photosets, len(photosets)
    
    async def get_photosets(
        self,
        skip: int = 0,
        limit: int = 50
    ) -> List[Photoset]:
        """
        Get paginated list of photosets.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of photosets
        """
        return await self.repo.get_all(skip=skip, limit=limit)
    
    async def get_photosets_by_year(
        self,
        year: int,
        skip: int = 0,
        limit: int = 50
    ) -> List[Photoset]:
        """
        Get photosets by year.
        
        Args:
            year: Year to filter by
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of photosets
        """
        return await self.repo.get_by_year(year, skip=skip, limit=limit)
    
    async def search_photosets(
        self,
        name_pattern: str,
        skip: int = 0,
        limit: int = 50
    ) -> List[Photoset]:
        """
        Search photosets by name pattern.
        
        Args:
            name_pattern: Name pattern to search for
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of matching photosets
        """
        return await self.repo.search_by_name(name_pattern, skip=skip, limit=limit)
    
    async def count_photosets(self) -> int:
        """Get total number of photosets."""
        return await self.repo.count()
    
    async def get_photoset_images(
        self,
        photoset_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[Image]:
        """
        Get paginated list of images in a photoset.
        
        Args:
            photoset_id: Photoset UUID
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of images
            
        Raises:
            NotFoundException: If photoset not found
        """
        # Verify photoset exists
        if not await self.repo.get_by_id(photoset_id):
            raise NotFoundException("Photoset", str(photoset_id))
        
        return await self.image_repo.get_by_photoset(photoset_id, skip=skip, limit=limit)

