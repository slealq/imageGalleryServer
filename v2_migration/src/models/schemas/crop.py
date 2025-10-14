"""Crop Pydantic schemas."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from uuid import UUID


class NormalizedDeltas(BaseModel):
    """Normalized delta coordinates for cropping."""
    
    x: float = Field(..., ge=-1.0, le=1.0, description="Normalized X delta")
    y: float = Field(..., ge=-1.0, le=1.0, description="Normalized Y delta")


class CropBase(BaseModel):
    """Base crop schema with common fields."""
    
    target_size: int = Field(..., ge=64, le=4096, description="Target size in pixels")
    normalized_delta_x: float = Field(..., ge=-1.0, le=1.0, description="Normalized X delta")
    normalized_delta_y: float = Field(..., ge=-1.0, le=1.0, description="Normalized Y delta")


class CropCreate(BaseModel):
    """Schema for creating a new crop."""
    
    image_id: UUID = Field(..., description="Image ID")
    target_size: int = Field(..., ge=64, le=4096, description="Target size in pixels")
    normalized_deltas: NormalizedDeltas = Field(..., description="Normalized deltas")


class CropUpdate(BaseModel):
    """Schema for updating a crop."""
    
    target_size: Optional[int] = Field(default=None, ge=64, le=4096)
    normalized_deltas: Optional[NormalizedDeltas] = None


class CropResponse(CropBase):
    """Schema for crop responses."""
    
    id: UUID
    image_id: UUID
    crop_file_path: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CropWithImageResponse(BaseModel):
    """Crop response with image URL."""
    
    crop_info: CropResponse
    image_url: str = Field(..., description="URL to access cropped image")
    
    class Config:
        from_attributes = True


