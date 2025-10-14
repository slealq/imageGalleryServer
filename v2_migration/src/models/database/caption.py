"""Caption database model."""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.core.database import Base

if TYPE_CHECKING:
    from .image import Image


class Caption(Base):
    """Caption model for image descriptions."""
    
    __tablename__ = "captions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    caption = Column(Text, nullable=False)
    generator_type = Column(String(50), default="manual", nullable=False)
    generator_metadata = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Relationships
    image = relationship("Image", back_populates="caption")
    
    def __repr__(self) -> str:
        caption_preview = self.caption[:50] + "..." if len(self.caption) > 50 else self.caption
        return f"<Caption(id={self.id}, type='{self.generator_type}', caption='{caption_preview}')>"


