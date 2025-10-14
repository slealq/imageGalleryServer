"""Image Pydantic schemas."""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import UUID


class ImageBase(BaseModel):
    """Base image schema with common fields."""
    
    original_filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
    width: Optional[int] = Field(default=None, ge=1, description="Image width in pixels")
    height: Optional[int] = Field(default=None, ge=1, description="Image height in pixels")
    file_size: Optional[int] = Field(default=None, ge=0, description="File size in bytes")
    mime_type: Optional[str] = Field(default=None, max_length=50, description="MIME type")
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ImageCreate(ImageBase):
    """Schema for creating a new image."""
    
    photoset_id: Optional[UUID] = Field(default=None, description="Photoset ID")
    file_path: str = Field(..., description="File path relative to images directory")


class ImageUpdate(BaseModel):
    """Schema for updating an image."""
    
    original_filename: Optional[str] = Field(default=None, min_length=1, max_length=255)
    photoset_id: Optional[UUID] = None
    width: Optional[int] = Field(default=None, ge=1)
    height: Optional[int] = Field(default=None, ge=1)
    file_size: Optional[int] = Field(default=None, ge=0)
    mime_type: Optional[str] = Field(default=None, max_length=50)
    extra_metadata: Optional[Dict[str, Any]] = None


class ImageResponse(ImageBase):
    """Schema for image responses."""
    
    id: UUID
    photoset_id: Optional[UUID]
    file_path: str
    created_at: datetime
    updated_at: datetime
    has_caption: bool = Field(default=False, description="Whether image has caption")
    has_crop: bool = Field(default=False, description="Whether image has crop")
    has_thumbnails: bool = Field(default=False, description="Whether thumbnails exist")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    
    class Config:
        from_attributes = True


class ImageMetadataResponse(BaseModel):
    """Detailed image metadata response."""
    
    id: UUID
    photoset_id: Optional[UUID]
    original_filename: str
    width: Optional[int]
    height: Optional[int]
    file_size: Optional[int]
    mime_type: Optional[str]
    created_at: datetime
    updated_at: datetime
    has_caption: bool
    has_crop: bool
    caption_text: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    photoset_name: Optional[str] = None
    
    class Config:
        from_attributes = True


class ImageListResponse(BaseModel):
    """Schema for paginated image list."""
    
    images: List[ImageResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    
    class Config:
        from_attributes = True


class ImageExportRequest(BaseModel):
    """Request to export images."""
    
    image_ids: List[UUID] = Field(..., min_length=1, description="List of image IDs to export")
    include_captions: bool = Field(default=True, description="Include captions in export")
    include_crops: bool = Field(default=True, description="Include crops in export")


