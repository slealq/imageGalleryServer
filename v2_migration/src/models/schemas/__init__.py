"""Pydantic schemas for API validation."""
from __future__ import annotations

from .common import PaginationParams, PaginatedResponse, FilterParams
from .photoset import (
    PhotosetBase,
    PhotosetCreate,
    PhotosetUpdate,
    PhotosetResponse,
)
from .image import (
    ImageBase,
    ImageCreate,
    ImageUpdate,
    ImageResponse,
    ImageMetadataResponse,
)
from .caption import (
    CaptionBase,
    CaptionCreate,
    CaptionResponse,
    CaptionGenerateRequest,
)
from .crop import (
    CropBase,
    CropCreate,
    CropResponse,
    CropWithImageResponse,
    NormalizedDeltas,
)
from .tag import (
    TagBase,
    TagResponse,
    TagListResponse,
    AddTagRequest,
)

__all__ = [
    # Common
    "PaginationParams",
    "PaginatedResponse",
    "FilterParams",
    # Photoset
    "PhotosetBase",
    "PhotosetCreate",
    "PhotosetUpdate",
    "PhotosetResponse",
    # Image
    "ImageBase",
    "ImageCreate",
    "ImageUpdate",
    "ImageResponse",
    "ImageMetadataResponse",
    # Caption
    "CaptionBase",
    "CaptionCreate",
    "CaptionResponse",
    "CaptionGenerateRequest",
    # Crop
    "CropBase",
    "CropCreate",
    "CropResponse",
    "CropWithImageResponse",
    "NormalizedDeltas",
    # Tag
    "TagBase",
    "TagResponse",
    "TagListResponse",
    "AddTagRequest",
]


