"""Tag service for managing tags."""
from __future__ import annotations

from typing import List, Dict, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import NotFoundException
from src.models.database import Tag
from src.repositories import TagRepository, ImageRepository, PhotosetRepository


class TagService:
    """Service for tag operations."""
    
    def __init__(self, db: AsyncSession):
        """
        Initialize tag service.
        
        Args:
            db: Database session
        """
        self.db = db
        self.tag_repo = TagRepository(db)
        self.image_repo = ImageRepository(db)
        self.photoset_repo = PhotosetRepository(db)
    
    async def get_tag(self, tag_id: UUID) -> Tag:
        """
        Get tag by ID.
        
        Args:
            tag_id: Tag UUID
            
        Returns:
            Tag model
            
        Raises:
            NotFoundException: If tag not found
        """
        return await self.tag_repo.get_by_id_or_fail(tag_id)
    
    async def get_tag_by_name(self, name: str) -> Optional[Tag]:
        """
        Get tag by name.
        
        Args:
            name: Tag name
            
        Returns:
            Tag model or None
        """
        return await self.tag_repo.get_by_name(name)
    
    async def get_all_tags(self) -> List[Tag]:
        """
        Get all tags.
        
        Returns:
            List of all tags
        """
        return await self.tag_repo.get_all()
    
    async def get_tags_by_type(self, tag_type: str) -> List[Tag]:
        """
        Get tags by type.
        
        Args:
            tag_type: Tag type (photoset, image, actor, custom)
            
        Returns:
            List of tags
        """
        return await self.tag_repo.get_by_type(tag_type)
    
    async def get_tags_grouped(self) -> Dict[str, List[Tag]]:
        """
        Get all tags grouped by type.
        
        Returns:
            Dictionary mapping tag types to tag lists
        """
        return await self.tag_repo.get_grouped_by_type()
    
    async def create_tag(self, name: str, tag_type: str = "custom") -> Tag:
        """
        Create a new tag.
        
        Args:
            name: Tag name
            tag_type: Tag type
            
        Returns:
            Created tag model
        """
        return await self.tag_repo.get_or_create(name, tag_type)
    
    # Image tag operations
    
    async def add_tag_to_image(
        self,
        image_id: UUID,
        tag_name: str,
        tag_type: str = "custom"
    ) -> Tag:
        """
        Add a tag to an image.
        
        Args:
            image_id: Image UUID
            tag_name: Tag name
            tag_type: Tag type
            
        Returns:
            Tag model
            
        Raises:
            NotFoundException: If image not found
        """
        # Verify image exists
        await self.image_repo.get_by_id_or_fail(image_id)
        
        # Get or create tag
        tag = await self.tag_repo.get_or_create(tag_name, tag_type)
        
        # Add tag to image
        await self.tag_repo.add_to_image(image_id, tag.id)
        
        return tag
    
    async def remove_tag_from_image(
        self,
        image_id: UUID,
        tag_id: UUID
    ) -> bool:
        """
        Remove a tag from an image.
        
        Args:
            image_id: Image UUID
            tag_id: Tag UUID
            
        Returns:
            True if removed
        """
        return await self.tag_repo.remove_from_image(image_id, tag_id)
    
    async def get_image_tags(self, image_id: UUID) -> List[Tag]:
        """
        Get all tags for an image.
        
        Args:
            image_id: Image UUID
            
        Returns:
            List of tags
        """
        return await self.tag_repo.get_image_tags(image_id)
    
    # Photoset tag operations
    
    async def add_tag_to_photoset(
        self,
        photoset_id: UUID,
        tag_name: str,
        tag_type: str = "photoset"
    ) -> Tag:
        """
        Add a tag to a photoset.
        
        Args:
            photoset_id: Photoset UUID
            tag_name: Tag name
            tag_type: Tag type
            
        Returns:
            Tag model
            
        Raises:
            NotFoundException: If photoset not found
        """
        # Verify photoset exists
        await self.photoset_repo.get_by_id_or_fail(photoset_id)
        
        # Get or create tag
        tag = await self.tag_repo.get_or_create(tag_name, tag_type)
        
        # Add tag to photoset
        await self.tag_repo.add_to_photoset(photoset_id, tag.id)
        
        return tag
    
    async def remove_tag_from_photoset(
        self,
        photoset_id: UUID,
        tag_id: UUID
    ) -> bool:
        """
        Remove a tag from a photoset.
        
        Args:
            photoset_id: Photoset UUID
            tag_id: Tag UUID
            
        Returns:
            True if removed
        """
        return await self.tag_repo.remove_from_photoset(photoset_id, tag_id)
    
    async def get_photoset_tags(self, photoset_id: UUID) -> List[Tag]:
        """
        Get all tags for a photoset.
        
        Args:
            photoset_id: Photoset UUID
            
        Returns:
            List of tags
        """
        return await self.tag_repo.get_photoset_tags(photoset_id)
    
    async def delete_tag(self, tag_id: UUID) -> bool:
        """
        Delete a tag (removes from all images and photosets).
        
        Args:
            tag_id: Tag UUID
            
        Returns:
            True if deleted
        """
        return await self.tag_repo.delete(tag_id)


# Fix import
from typing import Optional

