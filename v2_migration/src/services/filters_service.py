"""Filters service for getting available filter options."""
from __future__ import annotations

from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories import TagRepository, PhotosetRepository


class FiltersService:
    """Service for filter operations."""
    
    def __init__(self, db: AsyncSession):
        """
        Initialize filters service.
        
        Args:
            db: Database session
        """
        self.db = db
        self.tag_repo = TagRepository(db)
        self.photoset_repo = PhotosetRepository(db)
    
    async def get_available_filters(self) -> dict[str, List[str | int]]:
        """
        Get all available filter options.
        
        Returns:
            Dictionary with actors, tags, and years lists
        """
        # Get all tags grouped by type
        tags_grouped = await self.tag_repo.get_grouped_by_type()
        
        # Extract actors (tags with type 'actor')
        actors = sorted([tag.name for tag in tags_grouped.get('actor', [])])
        
        # Extract all tag names (excluding actors for tags list)
        all_tags = []
        for tag_type, tags_list in tags_grouped.items():
            if tag_type != 'actor':  # Don't include actors in tags list
                all_tags.extend([tag.name for tag in tags_list])
        tags = sorted(set(all_tags))
        
        # Get distinct years from photosets
        years = await self.photoset_repo.get_distinct_years()
        
        return {
            "actors": actors,
            "tags": tags,
            "years": years
        }

