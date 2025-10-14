"""Image repository."""
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.database import Image, ImageTag, Tag
from src.repositories.base import BaseRepository


class ImageRepository(BaseRepository[Image]):
    """Repository for image operations."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(Image, db)
    
    async def get_by_photoset(
        self,
        photoset_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[Image]:
        """
        Get images by photoset.
        
        Args:
            photoset_id: Photoset UUID
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of images
        """
        result = await self.db.execute(
            select(Image)
            .where(Image.photoset_id == photoset_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_with_caption(self, id: UUID) -> Optional[Image]:
        """
        Get image with caption loaded.
        
        Args:
            id: Image UUID
            
        Returns:
            Image with caption or None
        """
        result = await self.db.execute(
            select(Image)
            .options(selectinload(Image.caption))
            .where(Image.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_with_crop(self, id: UUID) -> Optional[Image]:
        """
        Get image with crop loaded.
        
        Args:
            id: Image UUID
            
        Returns:
            Image with crop or None
        """
        result = await self.db.execute(
            select(Image)
            .options(selectinload(Image.crop))
            .where(Image.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_with_tags(self, id: UUID) -> Optional[Image]:
        """
        Get image with tags loaded.
        
        Args:
            id: Image UUID
            
        Returns:
            Image with tags or None
        """
        result = await self.db.execute(
            select(Image)
            .options(selectinload(Image.image_tags))
            .where(Image.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_file_path(self, file_path: str) -> Optional[Image]:
        """
        Get image by file path.
        
        Args:
            file_path: File path
            
        Returns:
            Image or None
        """
        result = await self.db.execute(
            select(Image).where(Image.file_path == file_path)
        )
        return result.scalar_one_or_none()
    
    async def get_with_caption_filter(
        self,
        has_caption: bool,
        skip: int = 0,
        limit: int = 100
    ) -> List[Image]:
        """
        Get images filtered by caption presence.
        
        Args:
            has_caption: Whether to include only images with captions
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of images
        """
        from src.models.database import Caption
        
        query = select(Image)
        
        if has_caption:
            query = query.join(Caption).where(Caption.image_id == Image.id)
        else:
            query = query.outerjoin(Caption).where(Caption.id == None)
        
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_with_crop_filter(
        self,
        has_crop: bool,
        skip: int = 0,
        limit: int = 100
    ) -> List[Image]:
        """
        Get images filtered by crop presence.
        
        Args:
            has_crop: Whether to include only images with crops
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of images
        """
        from src.models.database import Crop
        
        query = select(Image)
        
        if has_crop:
            query = query.join(Crop).where(Crop.image_id == Image.id)
        else:
            query = query.outerjoin(Crop).where(Crop.id == None)
        
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_by_tag(
        self,
        tag_name: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Image]:
        """
        Get images by tag name.
        
        Args:
            tag_name: Tag name
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of images
        """
        result = await self.db.execute(
            select(Image)
            .join(ImageTag)
            .join(Tag)
            .where(Tag.name == tag_name)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def count_by_photoset(self, photoset_id: UUID) -> int:
        """
        Count images in photoset.
        
        Args:
            photoset_id: Photoset UUID
            
        Returns:
            Image count
        """
        result = await self.db.execute(
            select(func.count())
            .select_from(Image)
            .where(Image.photoset_id == photoset_id)
        )
        return result.scalar_one()


