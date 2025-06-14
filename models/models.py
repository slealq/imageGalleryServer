from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

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
    has_custom_tags: bool = False
    custom_tags: List[str] = []

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

class TagResponse(BaseModel):
    tags: List[str]

class AddTagRequest(BaseModel):
    tag: str

class ExportRequest(BaseModel):
    imageIds: List[str] 