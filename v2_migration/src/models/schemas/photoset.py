"""Photoset Pydantic schemas."""
from __future__ import annotations

from datetime import date as DateType, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class PhotosetBase(BaseModel):
    """Base photoset schema with common fields."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Photoset name")
    source_url: Optional[str] = Field(default=None, description="Source URL")
    date: Optional[DateType] = Field(default=None, description="Photoset date")
    year: Optional[int] = Field(default=None, ge=1900, le=2100, description="Year")
    original_archive_filename: Optional[str] = Field(default=None, description="Original archive filename")
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PhotosetCreate(PhotosetBase):
    """Schema for creating a photoset."""
    pass


class PhotosetUpdate(BaseModel):
    """Schema for updating a photoset."""
    
    name: Optional[str] = Field(default=None, max_length=255)
    source_url: Optional[str] = Field(default=None)
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None)


class PhotosetResponse(PhotosetBase):
    """Schema for photoset response."""
    
    id: UUID = Field(..., description="Photoset UUID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    image_count: int = Field(default=0, description="Number of images")
    tags: List[str] = Field(default_factory=list, description="Photoset tags")
    
    model_config = ConfigDict(from_attributes=True)
