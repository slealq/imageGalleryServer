"""Common Pydantic schemas used across the application."""
from typing import Generic, List, TypeVar, Optional
from pydantic import BaseModel, Field
from uuid import UUID


class PaginationParams(BaseModel):
    """Pagination parameters."""
    
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=100, description="Items per page")


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    
    class Config:
        from_attributes = True


class IDResponse(BaseModel):
    """Simple response with just an ID."""
    
    id: UUID
    
    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """Simple message response."""
    
    message: str
    
    class Config:
        from_attributes = True


class FilterParams(BaseModel):
    """Common filtering parameters."""
    
    actor: Optional[str] = Field(default=None, description="Filter by actor")
    tag: Optional[str] = Field(default=None, description="Filter by tag")
    year: Optional[int] = Field(default=None, description="Filter by year")
    has_caption: Optional[bool] = Field(default=None, description="Filter by caption presence")
    has_crop: Optional[bool] = Field(default=None, description="Filter by crop presence")
    photoset_id: Optional[UUID] = Field(default=None, description="Filter by photoset")


