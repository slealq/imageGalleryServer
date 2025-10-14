"""Caption Pydantic schemas."""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


class CaptionBase(BaseModel):
    """Base caption schema with common fields."""
    
    caption: str = Field(..., min_length=1, description="Caption text")
    generator_type: str = Field(default="manual", max_length=50, description="Generator type")
    generator_metadata: Dict[str, Any] = Field(default_factory=dict, description="Generator metadata")


class CaptionCreate(BaseModel):
    """Schema for creating a new caption."""
    
    image_id: UUID = Field(..., description="Image ID")
    caption: str = Field(..., min_length=1, description="Caption text")
    generator_type: str = Field(default="manual", max_length=50)
    generator_metadata: Dict[str, Any] = Field(default_factory=dict)


class CaptionUpdate(BaseModel):
    """Schema for updating a caption."""
    
    caption: Optional[str] = Field(default=None, min_length=1)
    generator_type: Optional[str] = Field(default=None, max_length=50)
    generator_metadata: Optional[Dict[str, Any]] = None


class CaptionResponse(CaptionBase):
    """Schema for caption responses."""
    
    id: UUID
    image_id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CaptionGenerateRequest(BaseModel):
    """Request to generate a caption."""
    
    prompt: Optional[str] = Field(
        default=None,
        description="Optional prompt to guide caption generation"
    )
    generator_type: Optional[str] = Field(
        default=None,
        description="Override default generator type"
    )


