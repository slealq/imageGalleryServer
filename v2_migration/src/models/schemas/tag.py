"""Tag Pydantic schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class TagBase(BaseModel):
    """Base tag schema."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Tag name")
    tag_type: str = Field(default="custom", description="Tag type (photoset, image, actor, custom)")


class AddTagRequest(TagBase):
    """Schema for adding a tag."""
    
    tag_name: str = Field(..., alias="name", description="Tag name")


class TagResponse(TagBase):
    """Schema for tag response."""
    
    id: UUID = Field(..., description="Tag UUID")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = ConfigDict(from_attributes=True)


class TagListResponse(BaseModel):
    """Schema for tag list response."""
    
    tags: List[TagResponse] = Field(..., description="All tags")
    tags_by_type: Dict[str, List[TagResponse]] = Field(..., description="Tags grouped by type")
    total: int = Field(..., description="Total tag count")
