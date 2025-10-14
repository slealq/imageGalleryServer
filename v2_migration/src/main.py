"""Main FastAPI application."""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.database import close_db
from src.api.middleware import RequestTimingMiddleware
from src.api.routes import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    print("🚀 Starting Image Gallery v2...")
    print(f"📁 Storage root: {settings.storage_root}")
    print(f"🗄️  Database: {settings.database_url.split('@')[1] if '@' in settings.database_url else 'configured'}")
    print(f"📝 Caption generator: {settings.caption_generator}")
    
    # Ensure directories exist
    settings.ensure_directories_exist()
    
    yield
    
    # Shutdown
    print("👋 Shutting down Image Gallery v2...")
    await close_db()


# Create FastAPI application
app = FastAPI(
    title="Image Gallery API v2",
    description="Database-backed image gallery with photosets, captions, crops, and tags",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
app.add_middleware(RequestTimingMiddleware)

# Include API routes
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Image Gallery API v2",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/v2/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

