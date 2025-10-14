"""Photoset database model."""
import uuid
from datetime import datetime
from typing import List, TYPE_CHECKING

from sqlalchemy import Column, String, Text, Date, Integer, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.core.database import Base

if TYPE_CHECKING:
    from .image import Image
    from .tag import PhotosetTag


class Photoset(Base):
    """Photoset model representing a collection of images."""
    
    __tablename__ = "photosets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    source_url = Column(Text, nullable=True)
    date = Column(Date, nullable=True)
    year = Column(Integer, nullable=True, index=True)
    original_archive_filename = Column(String(255), nullable=True)
    metadata = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Relationships
    images = relationship("Image", back_populates="photoset", cascade="all, delete-orphan")
    photoset_tags = relationship("PhotosetTag", back_populates="photoset", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Photoset(id={self.id}, name='{self.name}', year={self.year})>"


