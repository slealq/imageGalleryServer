"""Base caption generator interface."""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from PIL import Image as PILImage


class BaseCaptionGenerator(ABC):
    """Abstract base class for caption generators."""
    
    @abstractmethod
    async def generate_caption(
        self,
        image: PILImage.Image,
        prompt: Optional[str] = None
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image object
            prompt: Optional prompt to guide generation
            
        Returns:
            Generated caption text
            
        Raises:
            CaptionGenerationException: If generation fails
        """
        pass
    
    @abstractmethod
    async def stream_caption(
        self,
        image: PILImage.Image,
        prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream caption generation token by token.
        
        Args:
            image: PIL Image object
            prompt: Optional prompt to guide generation
            
        Yields:
            Caption text chunks as they're generated
            
        Raises:
            CaptionGenerationException: If generation fails
        """
        yield ""
    
    @property
    @abstractmethod
    def generator_name(self) -> str:
        """
        Get the name of this generator.
        
        Returns:
            Generator name
        """
        pass
    
    @property
    def generator_metadata(self) -> dict:
        """
        Get metadata about this generator.
        
        Returns:
            Dictionary with generator information
        """
        return {
            "generator": self.generator_name,
            "version": "1.0"
        }


