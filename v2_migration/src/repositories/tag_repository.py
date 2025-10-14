"""Tag repository."""
from typing import List, Optional, Dict
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import Tag, PhotosetTag, ImageTag
from src.repositories.base import BaseRepository


class TagRepository(BaseRepository[Tag]):
    """Repository for tag operations."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(Tag, db)
    
    async def get_by_name(self, name: str) -> Optional[Tag]:
        """
        Get tag by name.
        
        Args:
            name: Tag name
            
        Returns:
            Tag or None
        """
        result = await self.db.execute(
            select(Tag).where(Tag.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_or_create(self, name: str, tag_type: str = "custom") -> Tag:
        """
        Get existing tag or create new one.
        
        Args:
            name: Tag name
            tag_type: Tag type
            
        Returns:
            Tag instance
        """
        tag = await self.get_by_name(name)
        if not tag:
            tag = Tag(name=name, tag_type=tag_type)
            tag = await self.create(tag)
        return tag
    
    async def get_by_type(self, tag_type: str) -> List[Tag]:
        """
        Get all tags of a specific type.
        
        Args:
            tag_type: Tag type to filter by
            
        Returns:
            List of tags
        """
        result = await self.db.execute(
            select(Tag).where(Tag.tag_type == tag_type).order_by(Tag.name)
        )
        return list(result.scalars().all())
    
    async def get_grouped_by_type(self) -> Dict[str, List[Tag]]:
        """
        Get all tags grouped by type.
        
        Returns:
            Dictionary mapping tag types to lists of tags
        """
        result = await self.db.execute(
            select(Tag).order_by(Tag.tag_type, Tag.name)
        )
        tags = result.scalars().all()
        
        grouped = {}
        for tag in tags:
            if tag.tag_type not in grouped:
                grouped[tag.tag_type] = []
            grouped[tag.tag_type].append(tag)
        
        return grouped
    
    # Photoset Tag Operations
    
    async def add_to_photoset(self, photoset_id: UUID, tag_id: UUID) -> PhotosetTag:
        """
        Add tag to photoset.
        
        Args:
            photoset_id: Photoset UUID
            tag_id: Tag UUID
            
        Returns:
            PhotosetTag association
        """
        # Check if association already exists
        result = await self.db.execute(
            select(PhotosetTag).where(
                PhotosetTag.photoset_id == photoset_id,
                PhotosetTag.tag_id == tag_id
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            return existing
        
        photoset_tag = PhotosetTag(photoset_id=photoset_id, tag_id=tag_id)
        self.db.add(photoset_tag)
        await self.db.flush()
        return photoset_tag
    
    async def remove_from_photoset(self, photoset_id: UUID, tag_id: UUID) -> bool:
        """
        Remove tag from photoset.
        
        Args:
            photoset_id: Photoset UUID
            tag_id: Tag UUID
            
        Returns:
            True if removed, False if not found
        """
        result = await self.db.execute(
            select(PhotosetTag).where(
                PhotosetTag.photoset_id == photoset_id,
                PhotosetTag.tag_id == tag_id
            )
        )
        photoset_tag = result.scalar_one_or_none()
        
        if photoset_tag:
            await self.db.delete(photoset_tag)
            await self.db.flush()
            return True
        return False
    
    async def get_photoset_tags(self, photoset_id: UUID) -> List[Tag]:
        """
        Get all tags for a photoset.
        
        Args:
            photoset_id: Photoset UUID
            
        Returns:
            List of tags
        """
        result = await self.db.execute(
            select(Tag)
            .join(PhotosetTag)
            .where(PhotosetTag.photoset_id == photoset_id)
            .order_by(Tag.name)
        )
        return list(result.scalars().all())
    
    # Image Tag Operations
    
    async def add_to_image(self, image_id: UUID, tag_id: UUID) -> ImageTag:
        """
        Add tag to image.
        
        Args:
            image_id: Image UUID
            tag_id: Tag UUID
            
        Returns:
            ImageTag association
        """
        # Check if association already exists
        result = await self.db.execute(
            select(ImageTag).where(
                ImageTag.image_id == image_id,
                ImageTag.tag_id == tag_id
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            return existing
        
        image_tag = ImageTag(image_id=image_id, tag_id=tag_id)
        self.db.add(image_tag)
        await self.db.flush()
        return image_tag
    
    async def remove_from_image(self, image_id: UUID, tag_id: UUID) -> bool:
        """
        Remove tag from image.
        
        Args:
            image_id: Image UUID
            tag_id: Tag UUID
            
        Returns:
            True if removed, False if not found
        """
        result = await self.db.execute(
            select(ImageTag).where(
                ImageTag.image_id == image_id,
                ImageTag.tag_id == tag_id
            )
        )
        image_tag = result.scalar_one_or_none()
        
        if image_tag:
            await self.db.delete(image_tag)
            await self.db.flush()
            return True
        return False
    
    async def get_image_tags(self, image_id: UUID) -> List[Tag]:
        """
        Get all tags for an image.
        
        Args:
            image_id: Image UUID
            
        Returns:
            List of tags
        """
        result = await self.db.execute(
            select(Tag)
            .join(ImageTag)
            .where(ImageTag.image_id == image_id)
            .order_by(Tag.name)
        )
        return list(result.scalars().all())


