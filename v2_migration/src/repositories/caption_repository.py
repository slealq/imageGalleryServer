"""Caption repository."""
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import Caption
from src.repositories.base import BaseRepository


class CaptionRepository(BaseRepository[Caption]):
    """Repository for caption operations."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(Caption, db)
    
    async def get_by_image_id(self, image_id: UUID) -> Optional[Caption]:
        """
        Get caption by image ID.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Caption or None
        """
        result = await self.db.execute(
            select(Caption).where(Caption.image_id == image_id)
        )
        return result.scalar_one_or_none()
    
    async def upsert(
        self,
        image_id: UUID,
        caption_text: str,
        generator_type: str = "manual",
        generator_metadata: dict = None
    ) -> Caption:
        """
        Create or update caption for an image.
        
        Args:
            image_id: Image UUID
            caption_text: Caption text
            generator_type: Type of generator used
            generator_metadata: Additional generator metadata
            
        Returns:
            Created or updated caption
        """
        existing = await self.get_by_image_id(image_id)
        
        if existing:
            existing.caption = caption_text
            existing.generator_type = generator_type
            existing.generator_metadata = generator_metadata or {}
            await self.db.flush()
            await self.db.refresh(existing)
            return existing
        else:
            caption = Caption(
                image_id=image_id,
                caption=caption_text,
                generator_type=generator_type,
                generator_metadata=generator_metadata or {}
            )
            return await self.create(caption)
    
    async def delete_by_image_id(self, image_id: UUID) -> bool:
        """
        Delete caption by image ID.
        
        Args:
            image_id: Image UUID
            
        Returns:
            True if deleted, False if not found
        """
        caption = await self.get_by_image_id(image_id)
        if caption:
            return await self.delete(caption.id)
        return False
    
    async def count(self) -> int:
        """
        Count total captions.
        
        Returns:
            Total number of captions
        """
        result = await self.db.execute(
            select(func.count())
            .select_from(Caption)
        )
        return result.scalar_one()


