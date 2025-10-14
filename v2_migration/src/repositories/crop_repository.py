"""Crop repository."""
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import Crop
from src.repositories.base import BaseRepository


class CropRepository(BaseRepository[Crop]):
    """Repository for crop operations."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(Crop, db)
    
    async def get_by_image_id(self, image_id: UUID) -> Optional[Crop]:
        """
        Get crop by image ID.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Crop or None
        """
        result = await self.db.execute(
            select(Crop).where(Crop.image_id == image_id)
        )
        return result.scalar_one_or_none()
    
    async def upsert(
        self,
        image_id: UUID,
        target_size: int,
        normalized_delta_x: float,
        normalized_delta_y: float,
        crop_file_path: str
    ) -> Crop:
        """
        Create or update crop for an image.
        
        Args:
            image_id: Image UUID
            target_size: Target size in pixels
            normalized_delta_x: Normalized X delta
            normalized_delta_y: Normalized Y delta
            crop_file_path: Path to cropped image file
            
        Returns:
            Created or updated crop
        """
        existing = await self.get_by_image_id(image_id)
        
        if existing:
            existing.target_size = target_size
            existing.normalized_delta_x = normalized_delta_x
            existing.normalized_delta_y = normalized_delta_y
            existing.crop_file_path = crop_file_path
            await self.db.flush()
            await self.db.refresh(existing)
            return existing
        else:
            crop = Crop(
                image_id=image_id,
                target_size=target_size,
                normalized_delta_x=normalized_delta_x,
                normalized_delta_y=normalized_delta_y,
                crop_file_path=crop_file_path
            )
            return await self.create(crop)
    
    async def delete_by_image_id(self, image_id: UUID) -> bool:
        """
        Delete crop by image ID.
        
        Args:
            image_id: Image UUID
            
        Returns:
            True if deleted, False if not found
        """
        crop = await self.get_by_image_id(image_id)
        if crop:
            return await self.delete(crop.id)
        return False
    
    async def count(self) -> int:
        """
        Count total crops.
        
        Returns:
            Total number of crops
        """
        result = await self.db.execute(
            select(func.count())
            .select_from(Crop)
        )
        return result.scalar_one()


