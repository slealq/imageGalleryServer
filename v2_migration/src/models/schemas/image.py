"""Image Pydantic schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class ImageBase(BaseModel):
    """Base image schema with common fields."""
    
    original_filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
    width: Optional[int] = Field(default=None, ge=1, description="Image width in pixels")
    height: Optional[int] = Field(default=None, ge=1, description="Image height in pixels")
    file_size: Optional[int] = Field(default=None, ge=0, description="File size in bytes")
    mime_type: Optional[str] = Field(default=None, max_length=50, description="MIME type")
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ImageCreate(ImageBase):
    """Schema for creating an image."""
    
    photoset_id: Optional[UUID] = Field(default=None, description="Photoset UUID")


class ImageUpdate(BaseModel):
    """Schema for updating an image."""
    
    original_filename: Optional[str] = Field(default=None, max_length=255)
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None)


class ImageResponse(ImageBase):
    """Schema for image response."""
    
    id: UUID = Field(..., description="Image UUID")
    photoset_id: Optional[UUID] = Field(default=None, description="Photoset UUID")
    file_path: str = Field(..., description="Relative file path")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    model_config = ConfigDict(from_attributes=True)


class ImageMetadataResponse(ImageResponse):
    """Extended image response with metadata."""
    
    has_caption: bool = Field(..., description="Whether image has a caption")
    has_crop: bool = Field(..., description="Whether image has a crop")
    tags: List[str] = Field(default_factory=list, description="Image tags")
