import os
from pathlib import Path

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

# Directory configuration
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", BASE_DIR / "images"))
CAPTIONS_DIR = Path(os.getenv("CAPTIONS_DIR", BASE_DIR / "captions"))
CROPS_DIR = Path(os.getenv("CROPS_DIR", BASE_DIR / "crops"))
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", BASE_DIR / "exports"))
PHOTOSET_METADATA_DIRECTORY = Path(os.getenv("PHOTOSET_METADATA_DIRECTORY", BASE_DIR / "metadata"))

# Cache configuration
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "10 * 1024 * 1024 * 1024"))  # 10GB default
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default

# Image configuration
IMAGES_PER_PAGE = int(os.getenv("IMAGES_PER_PAGE", "20"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10 * 1024 * 1024"))  # 10MB default
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif"}

# Performance configuration
SLOW_REQUEST_THRESHOLD = float(os.getenv("SLOW_REQUEST_THRESHOLD", "1.0"))  # 1 second
SLOW_IMAGE_PROCESSING_THRESHOLD = float(os.getenv("SLOW_IMAGE_PROCESSING_THRESHOLD", "0.5"))  # 500ms
SLOW_BATCH_PROCESSING_THRESHOLD = float(os.getenv("SLOW_BATCH_PROCESSING_THRESHOLD", "2.0"))  # 2 seconds

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = os.getenv("LOG_FILE", BASE_DIR / "logs" / "image_server.log")

# Create necessary directories
for directory in [IMAGES_DIR, CAPTIONS_DIR, CROPS_DIR, EXPORT_DIR, PHOTOSET_METADATA_DIRECTORY, BASE_DIR / "logs"]:
    directory.mkdir(parents=True, exist_ok=True) 