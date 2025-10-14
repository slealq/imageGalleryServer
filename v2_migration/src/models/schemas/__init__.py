"""Pydantic schemas for API validation."""
from .common import PaginationParams, PaginatedResponse
from .photoset import (
    PhotosetBase,
    PhotosetCreate,
    PhotosetUpdate,
    PhotosetResponse,
    PhotosetListResponse,
)
from .image import (
    ImageBase,
    ImageCreate,
    ImageUpdate,
    ImageResponse,
    ImageListResponse,
    ImageMetadataResponse,
)
from .caption import (
    CaptionBase,
    CaptionCreate,
    CaptionUpdate,
    CaptionResponse,
    CaptionGenerateRequest,
)
from .crop import (
    CropBase,
    CropCreate,
    CropUpdate,
    CropResponse,
)
from .tag import (
    TagBase,
    TagCreate,
    TagResponse,
    TagListResponse,
)

__all__ = [
    # Common
    "PaginationParams",
    "PaginatedResponse",
    # Photoset
    "PhotosetBase",
    "PhotosetCreate",
    "PhotosetUpdate",
    "PhotosetResponse",
    "PhotosetListResponse",
    # Image
    "ImageBase",
    "ImageCreate",
    "ImageUpdate",
    "ImageResponse",
    "ImageListResponse",
    "ImageMetadataResponse",
    # Caption
    "CaptionBase",
    "CaptionCreate",
    "CaptionUpdate",
    "CaptionResponse",
    "CaptionGenerateRequest",
    # Crop
    "CropBase",
    "CropCreate",
    "CropUpdate",
    "CropResponse",
    # Tag
    "TagBase",
    "TagCreate",
    "TagResponse",
    "TagListResponse",
]


