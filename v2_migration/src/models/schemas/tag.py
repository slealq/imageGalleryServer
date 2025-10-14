"""Tag Pydantic schemas."""
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel, Field
from uuid import UUID


class TagBase(BaseModel):
    """Base tag schema with common fields."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Tag name")
    tag_type: str = Field(
        ...,
        description="Tag type: 'photoset', 'image', 'actor', or 'custom'"
    )


class TagCreate(TagBase):
    """Schema for creating a new tag."""
    pass


class TagResponse(TagBase):
    """Schema for tag responses."""
    
    id: UUID
    created_at: datetime
    
    class Config:
        from_attributes = True


class TagListResponse(BaseModel):
    """Schema for tag list grouped by type."""
    
    tags: List[TagResponse]
    tags_by_type: Dict[str, List[TagResponse]] = Field(
        default_factory=dict,
        description="Tags grouped by type"
    )
    total: int
    
    class Config:
        from_attributes = True


class AddTagRequest(BaseModel):
    """Request to add a tag."""
    
    tag_name: str = Field(..., min_length=1, max_length=100, description="Tag name")
    tag_type: str = Field(
        default="custom",
        description="Tag type: 'photoset', 'image', 'actor', or 'custom'"
    )


class RemoveTagRequest(BaseModel):
    """Request to remove a tag."""
    
    tag_id: UUID = Field(..., description="Tag ID to remove")


