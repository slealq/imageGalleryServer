"""Tag database models."""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.core.database import Base

if TYPE_CHECKING:
    from .photoset import Photoset
    from .image import Image


class Tag(Base):
    """Tag model for categorization."""
    
    __tablename__ = "tags"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True, index=True)
    tag_type = Column(
        String(20),
        nullable=False,
        index=True,
        # Valid types: 'photoset', 'image', 'actor', 'custom'
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    photoset_tags = relationship("PhotosetTag", back_populates="tag", cascade="all, delete-orphan")
    image_tags = relationship("ImageTag", back_populates="tag", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Tag(id={self.id}, name='{self.name}', type='{self.tag_type}')>"


class PhotosetTag(Base):
    """Association table for photoset-tag many-to-many relationship."""
    
    __tablename__ = "photoset_tags"
    
    photoset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("photosets.id", ondelete="CASCADE"),
        primary_key=True
    )
    tag_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True
    )
    
    # Relationships
    photoset = relationship("Photoset", back_populates="photoset_tags")
    tag = relationship("Tag", back_populates="photoset_tags")
    
    def __repr__(self) -> str:
        return f"<PhotosetTag(photoset_id={self.photoset_id}, tag_id={self.tag_id})>"


class ImageTag(Base):
    """Association table for image-tag many-to-many relationship."""
    
    __tablename__ = "image_tags"
    
    image_id = Column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="CASCADE"),
        primary_key=True
    )
    tag_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True
    )
    
    # Relationships
    image = relationship("Image", back_populates="image_tags")
    tag = relationship("Tag", back_populates="image_tags")
    
    def __repr__(self) -> str:
        return f"<ImageTag(image_id={self.image_id}, tag_id={self.tag_id})>"


