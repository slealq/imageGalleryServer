"""Embedding database model (Future feature)."""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, String, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.core.database import Base

if TYPE_CHECKING:
    from .image import Image


class Embedding(Base):
    """Embedding model for storing vector embedding metadata."""
    
    __tablename__ = "embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    model_name = Column(String(100), nullable=False, index=True)
    vector_dimension = Column(Integer, nullable=False)
    external_vector_id = Column(UUID(as_uuid=True), nullable=True)  # Reference to Qdrant
    extra_metadata = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    image = relationship("Image", back_populates="embeddings")
    
    def __repr__(self) -> str:
        return f"<Embedding(id={self.id}, model='{self.model_name}', dim={self.vector_dimension})>"


