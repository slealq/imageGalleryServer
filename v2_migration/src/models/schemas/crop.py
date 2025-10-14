"""Crop Pydantic schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class NormalizedDeltas(BaseModel):
    """Normalized delta coordinates."""
    
    x: float = Field(..., ge=-1.0, le=1.0, description="Normalized X delta")
    y: float = Field(..., ge=-1.0, le=1.0, description="Normalized Y delta")


class CropBase(BaseModel):
    """Base crop schema."""
    
    target_size: int = Field(..., gt=0, description="Target size in pixels")
    normalized_delta_x: float = Field(..., ge=-1.0, le=1.0, description="Normalized X delta")
    normalized_delta_y: float = Field(..., ge=-1.0, le=1.0, description="Normalized Y delta")


class CropCreate(BaseModel):
    """Schema for creating a crop."""
    
    target_size: int = Field(..., gt=0, description="Target size in pixels")
    normalized_deltas: NormalizedDeltas = Field(..., description="Normalized delta coordinates")


class CropResponse(CropBase):
    """Schema for crop response."""
    
    id: UUID = Field(..., description="Crop UUID")
    image_id: UUID = Field(..., description="Image UUID")
    crop_file_path: str = Field(..., description="Crop file path")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    model_config = ConfigDict(from_attributes=True)


class CropWithImageResponse(BaseModel):
    """Schema for crop response with image URL."""
    
    crop_info: CropResponse
    image_url: str = Field(..., description="URL to get cropped image")
