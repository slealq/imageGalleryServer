"""Main FastAPI application."""
import sys
import logging
import traceback
from pathlib import Path
from contextlib import asynccontextmanager

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.database import close_db
from src.api.middleware import RequestTimingMiddleware
from src.api.routes import api_router
from src.api.routes import legacy
from src.api.dependencies import get_filters_service
from src.models.schemas.filters import FiltersResponse
from src.services import FiltersService

# Configure detailed logging
# Note: File logging disabled during development to prevent reload loops
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


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
    lifespan=lifespan,
    debug=True  # Enable debug mode to show more error details
)

# Global exception handler to catch and log all unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and log them with full traceback."""
    logger.error(f"Unhandled exception on {request.method} {request.url}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    
    # Return a proper error response
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc().split('\n') if settings.log_level == "DEBUG" else None
        }
    )

# HTTP exception handler for more details
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with logging."""
    logger.warning(f"HTTP {exc.status_code} on {request.method} {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Add CORS middleware - Allow all origins for development
# Note: allow_credentials cannot be True when allow_origins is ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
app.add_middleware(RequestTimingMiddleware)

# Include API routes
app.include_router(api_router)

# Include legacy routes for backward compatibility
app.include_router(legacy.router, tags=["legacy"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Image Gallery API v2",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/v2/health"
    }


@app.get("/filters", response_model=FiltersResponse)
async def get_filters(
    filters_service: FiltersService = Depends(get_filters_service)
):
    """
    Get all available filter options.
    
    Returns available actors, tags, and years that can be used to filter images.
    
    **Returns:**
    - **actors**: List of available actor names
    - **tags**: List of available tag names (excluding actors)
    - **years**: List of available years from photosets
    """
    filters_data = await filters_service.get_available_filters()
    
    return FiltersResponse(
        actors=filters_data["actors"],
        tags=filters_data["tags"],
        years=filters_data["years"]
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

