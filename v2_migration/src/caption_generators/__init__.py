"""Pluggable caption generation system."""
from .base import BaseCaptionGenerator
from .dummy_generator import DummyCaptionGenerator
from .unsloth_generator import UnslothCaptionGenerator

__all__ = [
    "BaseCaptionGenerator",
    "DummyCaptionGenerator",
    "UnslothCaptionGenerator",
]


def get_caption_generator(generator_type: str = "dummy") -> BaseCaptionGenerator:
    """
    Factory function to get caption generator by type.
    
    Args:
        generator_type: Type of generator ('dummy', 'unsloth', 'none')
        
    Returns:
        Caption generator instance
        
    Raises:
        ValueError: If generator type is unknown
    """
    generators = {
        "dummy": DummyCaptionGenerator,
        "unsloth": UnslothCaptionGenerator,
        "none": DummyCaptionGenerator,  # Default to dummy for 'none'
    }
    
    generator_class = generators.get(generator_type.lower())
    if not generator_class:
        raise ValueError(
            f"Unknown caption generator type: {generator_type}. "
            f"Available: {', '.join(generators.keys())}"
        )
    
    return generator_class()


