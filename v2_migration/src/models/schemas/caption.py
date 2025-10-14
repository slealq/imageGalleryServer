"""Caption Pydantic schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class CaptionBase(BaseModel):
    """Base caption schema."""
    
    caption: str = Field(..., description="Caption text")
    generator_type: str = Field(default="manual", description="Type of generator used")


class CaptionCreate(CaptionBase):
    """Schema for creating/updating a caption."""
    pass


class CaptionGenerateRequest(BaseModel):
    """Schema for caption generation request."""
    
    prompt: Optional[str] = Field(default=None, description="Optional prompt to guide generation")


class CaptionResponse(CaptionBase):
    """Schema for caption response."""
    
    id: UUID = Field(..., description="Caption UUID")
    image_id: UUID = Field(..., description="Image UUID")
    generator_metadata: Dict[str, Any] = Field(default_factory=dict, description="Generator metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    model_config = ConfigDict(from_attributes=True)
