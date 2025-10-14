"""Application configuration management."""
from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = Field(..., description="PostgreSQL connection URL")
    database_pool_size: int = Field(default=20, description="Database connection pool size")
    database_echo: bool = Field(default=False, description="Echo SQL statements")
    
    # Redis
    redis_url: str = Field(..., description="Redis connection URL")
    
    # Storage paths (all from .env, no hardcoding)
    storage_root: Path = Field(..., description="Root storage directory")
    images_dir: Path = Field(..., description="Images storage directory")
    thumbnails_dir: Path = Field(..., description="Thumbnails storage directory")
    crops_dir: Path = Field(..., description="Crops storage directory")
    archives_dir: Path = Field(..., description="Archives storage directory")
    
    # API
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8002, description="API port")
    api_version: str = Field(default="v2", description="API version")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"], 
        description="CORS allowed origins"
    )
    
    # Caption Generation
    caption_generator: str = Field(
        default="dummy", 
        description="Caption generator type (unsloth, dummy, none)"
    )
    unsloth_model_path: Optional[Path] = Field(
        default=None, 
        description="Path to Unsloth model"
    )
    unsloth_load_in_4bit: bool = Field(
        default=True, 
        description="Load Unsloth model in 4-bit mode"
    )
    
    # Caching
    image_cache_size_mb: int = Field(
        default=10240, 
        description="Image cache size in MB"
    )
    metadata_cache_ttl_seconds: int = Field(
        default=3600, 
        description="Metadata cache TTL in seconds"
    )
    
    # Thumbnails (scaled down for browsing, maintains aspect ratio)
    thumbnail_max_dimension: int = Field(default=512, description="Max width or height for thumbnails")
    thumbnail_quality: int = Field(default=75, description="JPEG quality (1-100, lower = smaller file)")
    
    # Performance
    max_workers: int = Field(default=4, description="Max worker threads")
    prefetch_pages: int = Field(default=2, description="Number of pages to prefetch")
    
    # Logging
    log_level: str = Field(default="DEBUG", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    
    # Future: Embeddings
    qdrant_url: Optional[str] = Field(
        default=None, 
        description="Qdrant vector database URL"
    )
    embedding_model: str = Field(
        default="openai/clip-vit-base-patch32", 
        description="Embedding model name"
    )
    embedding_batch_size: int = Field(
        default=32, 
        description="Embedding generation batch size"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def ensure_directories_exist(self):
        """Ensure all storage directories exist."""
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.archives_dir.mkdir(parents=True, exist_ok=True)
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v
    
    def __init__(self, **kwargs):
        """Initialize settings and resolve all paths."""
        super().__init__(**kwargs)
        # Ensure all paths are absolute and resolved
        self.storage_root = self.storage_root.resolve()
        self.images_dir = self.images_dir.resolve()
        self.thumbnails_dir = self.thumbnails_dir.resolve()
        self.crops_dir = self.crops_dir.resolve()
        self.archives_dir = self.archives_dir.resolve()
        
        if self.unsloth_model_path:
            self.unsloth_model_path = self.unsloth_model_path.resolve()
        
        if self.log_file:
            self.log_file = self.log_file.resolve()
    
    def ensure_directories_exist(self):
        """Create all storage directories if they don't exist."""
        for directory in [
            self.storage_root,
            self.images_dir,
            self.thumbnails_dir,
            self.crops_dir,
            self.archives_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory if log_file is specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Singleton instance - will be initialized when config is imported
# In production, this reads from .env file
# In tests, this can be overridden with test configuration
try:
    settings = Settings()
except Exception as e:
    # If .env doesn't exist or is incomplete, provide helpful error
    print(f"Error loading configuration: {e}")
    print("Please run 'python setup/setup_wizard.py' to configure the application.")
    raise


