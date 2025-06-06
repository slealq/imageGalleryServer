from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ImageMetadata(BaseModel):
    id: str
    filename: str
    size: int
    created_at: datetime
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    has_caption: bool = False
    collection_name: str = "Default Collection"
    has_tags: bool = False
    has_crop: bool = False
    year: Optional[str] = None
    tags: List[str] = []
    actors: List[str] = []

class ImageResponse(BaseModel):
    images: List[ImageMetadata]
    total: int
    page: int
    page_size: int
    total_pages: int

class CaptionRequest(BaseModel):
    prompt: Optional[str] = None
    caption: Optional[str] = None

class CaptionResponse(BaseModel):
    caption: str

class NormalizedDeltas(BaseModel):
    x: float
    y: float

class CropRequest(BaseModel):
    imageId: str
    targetSize: int
    normalizedDeltas: NormalizedDeltas

class ExportRequest(BaseModel):
    imageIds: List[str]
