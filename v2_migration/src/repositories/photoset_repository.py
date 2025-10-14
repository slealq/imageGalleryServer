"""Photoset repository."""
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.database import Photoset, PhotosetTag
from src.repositories.base import BaseRepository


class PhotosetRepository(BaseRepository[Photoset]):
    """Repository for photoset operations."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(Photoset, db)
    
    async def get_with_images(self, id: UUID) -> Optional[Photoset]:
        """
        Get photoset with all associated images loaded.
        
        Args:
            id: Photoset UUID
            
        Returns:
            Photoset with images or None
        """
        result = await self.db.execute(
            select(Photoset)
            .options(selectinload(Photoset.images))
            .where(Photoset.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_with_tags(self, id: UUID) -> Optional[Photoset]:
        """
        Get photoset with all tags loaded.
        
        Args:
            id: Photoset UUID
            
        Returns:
            Photoset with tags or None
        """
        result = await self.db.execute(
            select(Photoset)
            .options(selectinload(Photoset.photoset_tags))
            .where(Photoset.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_year(
        self,
        year: int,
        skip: int = 0,
        limit: int = 100
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
        result = await self.db.execute(
            select(Photoset)
            .where(Photoset.year == year)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def search_by_name(
        self,
        name_pattern: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Photoset]:
        """
        Search photosets by name pattern.
        
        Args:
            name_pattern: SQL pattern to match (use % for wildcards)
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of matching photosets
        """
        result = await self.db.execute(
            select(Photoset)
            .where(Photoset.name.ilike(f"%{name_pattern}%"))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_image_count(self, id: UUID) -> int:
        """
        Get number of images in photoset.
        
        Args:
            id: Photoset UUID
            
        Returns:
            Image count
        """
        from src.models.database import Image
        
        result = await self.db.execute(
            select(func.count())
            .select_from(Image)
            .where(Image.photoset_id == id)
        )
        return result.scalar_one()
    
    async def get_by_archive_filename(self, filename: str) -> Optional[Photoset]:
        """
        Get photoset by original archive filename.
        
        Args:
            filename: Original archive filename
            
        Returns:
            Photoset or None
        """
        result = await self.db.execute(
            select(Photoset)
            .where(Photoset.original_archive_filename == filename)
        )
        return result.scalar_one_or_none()
    
    async def get_by_name(self, name: str) -> Optional[Photoset]:
        """
        Get photoset by name.
        
        Args:
            name: Photoset name
            
        Returns:
            Photoset or None
        """
        result = await self.db.execute(
            select(Photoset)
            .where(Photoset.name == name)
        )
        return result.scalar_one_or_none()


