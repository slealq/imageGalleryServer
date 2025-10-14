"""Photoset Pydantic schemas."""
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import UUID


class PhotosetBase(BaseModel):
    """Base photoset schema with common fields."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Photoset name")
    source_url: Optional[str] = Field(default=None, description="Source URL")
    date: Optional[date] = Field(default=None, description="Photoset date")
    year: Optional[int] = Field(default=None, ge=1900, le=2100, description="Year")
    original_archive_filename: Optional[str] = Field(default=None, description="Original archive filename")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PhotosetCreate(PhotosetBase):
    """Schema for creating a new photoset."""
    pass


class PhotosetUpdate(BaseModel):
    """Schema for updating a photoset."""
    
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    source_url: Optional[str] = None
    date: Optional[date] = None
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    original_archive_filename: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PhotosetResponse(PhotosetBase):
    """Schema for photoset responses."""
    
    id: UUID
    created_at: datetime
    updated_at: datetime
    image_count: Optional[int] = Field(default=0, description="Number of images in photoset")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    
    class Config:
        from_attributes = True


class PhotosetListResponse(BaseModel):
    """Schema for paginated photoset list."""
    
    photosets: List[PhotosetResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    
    class Config:
        from_attributes = True


class PhotosetExtractRequest(BaseModel):
    """Request to extract photoset from archive."""
    
    extract_images: bool = Field(default=True, description="Extract images from archive")
    generate_thumbnails: bool = Field(default=True, description="Generate thumbnails")
    import_metadata: bool = Field(default=True, description="Import metadata if available")


