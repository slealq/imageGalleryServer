from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path

from .controllers.image_controller import router as image_router
from .controllers.crop_controller import router as crop_router
from .controllers.caption_controller import router as caption_router
from .controllers.export_controller import router as export_router
from .middleware.timing_middleware import RequestTimingMiddleware
from .services.cache_service import (
    init_caption_cache,
    init_crop_cache,
    init_dimension_cache,
    image_cache
)
from config import (
    IMAGES_DIR,
    CAPTIONS_DIR,
    CROPS_DIR,
    EXPORT_DIR,
    HOST,
    PORT
)

# Create necessary directories
for directory in [IMAGES_DIR, CAPTIONS_DIR, CROPS_DIR, EXPORT_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Image Server",
    description="A FastAPI server for managing and serving images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add timing middleware
app.add_middleware(RequestTimingMiddleware)

# Include routers
app.include_router(image_router, prefix="/api", tags=["images"])
app.include_router(crop_router, prefix="/api", tags=["crops"])
app.include_router(caption_router, prefix="/api", tags=["captions"])
app.include_router(export_router, prefix="/api", tags=["exports"])

@app.on_event("startup")
async def startup_event():
    """Initialize caches on startup."""
    print("Initializing caches...")
    init_caption_cache()
    init_crop_cache()
    init_dimension_cache()
    print("Caches initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    print("Cleaning up resources...")
    image_cache.clear()
    print("Resources cleaned up successfully")

if __name__ == "__main__":
    uvicorn.run(
        "image_server.main:app",
        host=HOST,
        port=PORT,
        reload=True
    ) 