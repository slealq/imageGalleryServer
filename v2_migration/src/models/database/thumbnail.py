"""Thumbnail database model."""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.core.database import Base

if TYPE_CHECKING:
    from .image import Image


class Thumbnail(Base):
    """Thumbnail model for image thumbnails at different sizes."""
    
    __tablename__ = "thumbnails"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    size_name = Column(String(20), nullable=False)  # 'small', 'medium', 'large'
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    file_path = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    image = relationship("Image", back_populates="thumbnails")
    
    def __repr__(self) -> str:
        return f"<Thumbnail(id={self.id}, size='{self.size_name}', dimensions={self.width}x{self.height})>"


