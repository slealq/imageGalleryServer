"""Dummy caption generator for testing."""
import asyncio
from typing import AsyncIterator, Optional
from PIL import Image as PILImage

from .base import BaseCaptionGenerator


class DummyCaptionGenerator(BaseCaptionGenerator):
    """Dummy caption generator that returns placeholder captions."""
    
    async def generate_caption(
        self,
        image: PILImage.Image,
        prompt: Optional[str] = None
    ) -> str:
        """
        Generate a dummy caption.
        
        Args:
            image: PIL Image object
            prompt: Optional prompt (ignored in dummy implementation)
            
        Returns:
            Dummy caption with image dimensions
        """
        width, height = image.size
        
        if prompt:
            return f"[Dummy Caption] {prompt} - Image dimensions: {width}x{height}"
        else:
            return f"[Dummy Caption] An image with dimensions {width}x{height} pixels."
    
    async def stream_caption(
        self,
        image: PILImage.Image,
        prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream a dummy caption word by word.
        
        Args:
            image: PIL Image object
            prompt: Optional prompt (ignored in dummy implementation)
            
        Yields:
            Caption chunks
        """
        caption = await self.generate_caption(image, prompt)
        words = caption.split()
        
        for word in words:
            await asyncio.sleep(0.1)  # Simulate generation delay
            yield word + " "
    
    @property
    def generator_name(self) -> str:
        """Get generator name."""
        return "dummy"
    
    @property
    def generator_metadata(self) -> dict:
        """Get generator metadata."""
        return {
            **super().generator_metadata,
            "description": "Dummy generator for testing purposes"
        }


