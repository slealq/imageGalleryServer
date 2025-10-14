"""Crop database model."""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, Text, Integer, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.core.database import Base

if TYPE_CHECKING:
    from .image import Image


class Crop(Base):
    """Crop model for cropped image data."""
    
    __tablename__ = "crops"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    target_size = Column(Integer, nullable=False)
    normalized_delta_x = Column(Float, nullable=False)
    normalized_delta_y = Column(Float, nullable=False)
    crop_file_path = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Relationships
    image = relationship("Image", back_populates="crop")
    
    def __repr__(self) -> str:
        return f"<Crop(id={self.id}, target_size={self.target_size})>"


