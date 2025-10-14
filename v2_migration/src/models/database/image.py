"""Image database model."""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, String, Text, Integer, BigInteger, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.core.database import Base

if TYPE_CHECKING:
    from .photoset import Photoset
    from .caption import Caption
    from .crop import Crop
    from .thumbnail import Thumbnail
    from .tag import ImageTag
    from .embedding import Embedding


class Image(Base):
    """Image model representing individual image files."""
    
    __tablename__ = "images"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    photoset_id = Column(UUID(as_uuid=True), ForeignKey("photosets.id", ondelete="CASCADE"), nullable=True, index=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False, unique=True, index=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(BigInteger, nullable=True)
    mime_type = Column(String(50), nullable=True)
    metadata = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Relationships
    photoset = relationship("Photoset", back_populates="images")
    caption = relationship("Caption", back_populates="image", uselist=False, cascade="all, delete-orphan")
    crop = relationship("Crop", back_populates="image", uselist=False, cascade="all, delete-orphan")
    thumbnails = relationship("Thumbnail", back_populates="image", cascade="all, delete-orphan")
    image_tags = relationship("ImageTag", back_populates="image", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="image", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Image(id={self.id}, filename='{self.original_filename}')>"


