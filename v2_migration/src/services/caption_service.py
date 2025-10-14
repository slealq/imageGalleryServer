"""Caption service for managing and generating captions."""
from typing import Optional, AsyncIterator
from uuid import UUID
from PIL import Image as PILImage
import io

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.exceptions import NotFoundException
from src.models.database import Caption
from src.repositories import CaptionRepository, ImageRepository
from src.caption_generators import get_caption_generator
from src.services.storage_service import StorageService


class CaptionService:
    """Service for caption operations."""
    
    def __init__(
        self,
        db: AsyncSession,
        storage: StorageService
    ):
        """
        Initialize caption service.
        
        Args:
            db: Database session
            storage: Storage service
        """
        self.db = db
        self.storage = storage
        self.caption_repo = CaptionRepository(db)
        self.image_repo = ImageRepository(db)
        
        # Initialize caption generator
        self.generator = get_caption_generator(settings.caption_generator)
    
    async def get_caption(self, image_id: UUID) -> Optional[Caption]:
        """
        Get caption for an image.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Caption model or None
        """
        return await self.caption_repo.get_by_image_id(image_id)
    
    async def save_caption(
        self,
        image_id: UUID,
        caption_text: str,
        generator_type: str = "manual"
    ) -> Caption:
        """
        Save or update a caption.
        
        Args:
            image_id: Image UUID
            caption_text: Caption text
            generator_type: Type of generator used
            
        Returns:
            Caption model
            
        Raises:
            NotFoundException: If image not found
        """
        # Verify image exists
        await self.image_repo.get_by_id_or_fail(image_id)
        
        # Upsert caption
        return await self.caption_repo.upsert(
            image_id=image_id,
            caption_text=caption_text,
            generator_type=generator_type,
            generator_metadata=self.generator.generator_metadata
        )
    
    async def generate_caption(
        self,
        image_id: UUID,
        prompt: Optional[str] = None,
        save: bool = True
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image_id: Image UUID
            prompt: Optional prompt to guide generation
            save: Whether to save the generated caption
            
        Returns:
            Generated caption text
            
        Raises:
            NotFoundException: If image not found
        """
        # Get image
        image = await self.image_repo.get_by_id_or_fail(image_id)
        image_path = self.storage.images_dir / image.file_path
        
        if not image_path.exists():
            raise NotFoundException("Image file", str(image_id))
        
        # Load image
        pil_image = PILImage.open(image_path)
        
        # Generate caption
        caption_text = await self.generator.generate_caption(pil_image, prompt)
        
        pil_image.close()
        
        # Save if requested
        if save:
            await self.caption_repo.upsert(
                image_id=image_id,
                caption_text=caption_text,
                generator_type=self.generator.generator_name,
                generator_metadata=self.generator.generator_metadata
            )
        
        return caption_text
    
    async def stream_caption(
        self,
        image_id: UUID,
        prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream caption generation.
        
        Args:
            image_id: Image UUID
            prompt: Optional prompt to guide generation
            
        Yields:
            Caption text chunks
            
        Raises:
            NotFoundException: If image not found
        """
        # Get image
        image = await self.image_repo.get_by_id_or_fail(image_id)
        image_path = self.storage.images_dir / image.file_path
        
        if not image_path.exists():
            raise NotFoundException("Image file", str(image_id))
        
        # Load image
        pil_image = PILImage.open(image_path)
        
        # Stream caption generation
        full_caption = []
        async for chunk in self.generator.stream_caption(pil_image, prompt):
            full_caption.append(chunk)
            yield chunk
        
        pil_image.close()
        
        # Save complete caption
        complete_caption = "".join(full_caption).strip()
        await self.caption_repo.upsert(
            image_id=image_id,
            caption_text=complete_caption,
            generator_type=self.generator.generator_name,
            generator_metadata=self.generator.generator_metadata
        )
    
    async def delete_caption(self, image_id: UUID) -> bool:
        """
        Delete a caption.
        
        Args:
            image_id: Image UUID
            
        Returns:
            True if deleted
        """
        return await self.caption_repo.delete_by_image_id(image_id)

