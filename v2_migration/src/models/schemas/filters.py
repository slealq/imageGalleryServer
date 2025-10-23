"""Filters schema."""
from typing import List

from pydantic import BaseModel, Field


class FiltersResponse(BaseModel):
    """Response model for available filters."""
    
    actors: List[str] = Field(
        default_factory=list,
        description="List of available actor names"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of available tag names"
    )
    years: List[int] = Field(
        default_factory=list,
        description="List of available years"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "actors": ["Actor One", "Actor Two"],
                "tags": ["tag1", "tag2", "custom-tag"],
                "years": [2023, 2024, 2025]
            }
        }

