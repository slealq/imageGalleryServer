# Image Gallery v2 - Database-Backed Architecture

A complete rewrite of the image gallery service with database-backed metadata, modular architecture, and native Windows support.

## 🌟 Features

### Core Gallery Features
- **Photoset Management**: Organize images in collections
- **Image Browsing**: Efficient pagination and filtering
- **Caption Generation**: Pluggable caption generators (Unsloth, OpenAI, etc.)
- **Smart Cropping**: Store crop parameters and generate on-demand
- **Tagging System**: Tag photosets and images for easy organization
- **Thumbnail Generation**: Multiple sizes with caching
- **Export to ZIP**: Export selected images

### Advanced Features (Planned)
- **Embedding Generation**: Support for CLIP, SigLIP models
- **Semantic Search**: Find similar images using vector databases (Qdrant)
- **Batch Processing**: Async processing for large operations

## 🏗️ Architecture

### Technology Stack

**Core:**
- Python 3.10+
- FastAPI (async web framework)
- PostgreSQL 14+ (relational database)
- Redis 7+ / Memurai (caching)
- SQLAlchemy 2.0 (async ORM)
- Alembic (database migrations)

**Image Processing:**
- Pillow (PIL) for image manipulation
- NumPy (for future embedding support)

**Testing:**
- Pytest with async support
- Factory Boy for test fixtures

### Project Structure

```
v2_migration/
├── src/
│   ├── api/                    # FastAPI routes and middleware
│   │   ├── routes/             # Route modules
│   │   ├── dependencies.py     # Dependency injection
│   │   └── middleware.py       # Request timing, logging
│   ├── core/                   # Core configuration
│   │   ├── config.py           # Settings management
│   │   ├── database.py         # DB session management
│   │   ├── redis.py            # Redis client
│   │   └── exceptions.py       # Custom exceptions
│   ├── models/
│   │   ├── database/           # SQLAlchemy ORM models
│   │   └── schemas/            # Pydantic schemas
│   ├── repositories/           # Data access layer
│   ├── services/               # Business logic layer
│   ├── caption_generators/     # Pluggable caption system
│   └── main.py                 # FastAPI application
├── scripts/                    # Utility scripts
├── tests/                      # Test suite
├── migrations/                 # Alembic migrations
└── setup/                      # Installation scripts
```

### Layered Architecture

```
┌─────────────────────────────────────┐
│     FastAPI Routes (API Layer)      │
│  - Request validation               │
│  - Response serialization           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    Services (Business Logic)        │
│  - ImageService                     │
│  - PhotosetService                  │
│  - CaptionService                   │
│  - CropService, etc.                │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Repositories (Data Access)        │
│  - ImageRepository                  │
│  - PhotosetRepository               │
│  - Generic BaseRepository           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      SQLAlchemy ORM Models          │
│  - Database schema definitions      │
└─────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- PostgreSQL 14+ or Docker
- Redis 7+ or Memurai (for Windows)

### Installation (Windows)

1. **Clone the repository**
   ```batch
   cd v2_migration
   ```

2. **Run the setup wizard**
   ```batch
   setup\install_windows.bat
   ```

   This will:
   - Create a virtual environment
   - Install dependencies
   - Run the interactive setup wizard
   - Configure `.env` file
   - Initialize the database

3. **Start the service**
   ```batch
   run_windows.bat
   ```

4. **Access the API**
   - API Docs: http://localhost:8002/docs
   - Health Check: http://localhost:8002/api/v2/health

### Manual Setup

1. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   
   Copy `.env.example` to `.env` and configure:
   ```env
   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/gallery_v2
   REDIS_URL=redis://localhost:6379/0
   STORAGE_ROOT=C:/gallery_storage
   CAPTION_GENERATOR=dummy  # or 'unsloth'
   ```

4. **Initialize database**
   ```bash
   alembic upgrade head
   ```

5. **Run the application**
   ```bash
   python src/main.py
   ```

## 📚 API Documentation

### Core Endpoints

#### Health Check
```
GET /api/v2/health
```
Returns service health status.

#### Photosets

```
GET    /api/v2/photosets          # List photosets
POST   /api/v2/photosets          # Create photoset
GET    /api/v2/photosets/{id}     # Get photoset
PUT    /api/v2/photosets/{id}     # Update photoset
DELETE /api/v2/photosets/{id}     # Delete photoset
```

#### Images

```
GET    /api/v2/images/{id}                 # Get image file
GET    /api/v2/images/{id}/metadata        # Get image metadata
POST   /api/v2/images                      # Upload image
DELETE /api/v2/images/{id}                 # Delete image
```

#### Thumbnails

```
GET /api/v2/images/{id}/thumbnail/{size}   # Get thumbnail (small/medium/large)
```

#### Captions

```
GET  /api/v2/images/{id}/caption           # Get caption
POST /api/v2/images/{id}/caption           # Save caption
POST /api/v2/images/{id}/caption/generate  # Generate caption
```

#### Crops

```
GET  /api/v2/images/{id}/crop              # Get crop metadata
GET  /api/v2/images/{id}/cropped           # Get cropped image
POST /api/v2/images/{id}/crop              # Create/update crop
```

#### Tags

```
GET  /api/v2/tags                          # List all tags
POST /api/v2/images/{id}/tags              # Add tag to image
```

### Example Requests

**Create a photoset:**
```bash
curl -X POST http://localhost:8002/api/v2/photosets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Beach Photos 2024",
    "year": 2024,
    "source_url": "https://example.com/beach-2024"
  }'
```

**Generate a caption:**
```bash
curl -X POST http://localhost:8002/api/v2/images/{image_id}/caption/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Describe this image in detail"}'
```

## 🔧 Configuration

All configuration is managed through environment variables (`.env` file):

### Required Settings

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db

# Redis
REDIS_URL=redis://host:port/db

# Storage (configured by setup wizard)
STORAGE_ROOT=/path/to/storage
IMAGES_DIR=${STORAGE_ROOT}/images
THUMBNAILS_DIR=${STORAGE_ROOT}/thumbnails
CROPS_DIR=${STORAGE_ROOT}/crops
ARCHIVES_DIR=${STORAGE_ROOT}/archives
```

### Optional Settings

```env
# API
API_HOST=0.0.0.0
API_PORT=8002
CORS_ORIGINS=["http://localhost:3000"]

# Caption Generation
CAPTION_GENERATOR=dummy  # dummy, unsloth, openai
UNSLOTH_MODEL_PATH=/path/to/model

# Caching
IMAGE_CACHE_SIZE_MB=10240
METADATA_CACHE_TTL_SECONDS=3600

# Thumbnails
THUMBNAIL_SMALL_SIZE=256
THUMBNAIL_MEDIUM_SIZE=512
THUMBNAIL_LARGE_SIZE=1024
```

## 🗄️ Database Schema

### Core Tables

- **photosets**: Photoset metadata
- **images**: Image records
- **captions**: Generated/manual captions
- **crops**: Crop parameters
- **tags**: Tag definitions
- **photoset_tags**: Photoset-tag relationships
- **image_tags**: Image-tag relationships
- **thumbnails**: Thumbnail metadata
- **embeddings**: Future: embedding vectors

### Migrations

Create a new migration:
```bash
alembic revision --autogenerate -m "description"
```

Apply migrations:
```bash
alembic upgrade head
```

Rollback:
```bash
alembic downgrade -1
```

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/unit/services/
```

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── services/
│   ├── repositories/
│   └── caption_generators/
├── integration/       # Integration tests
│   ├── api/
│   └── database/
├── fixtures/          # Test fixtures
└── conftest.py        # Pytest configuration
```

## 📦 Dependencies

### Production

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
sqlalchemy[asyncio]>=2.0.0
asyncpg>=0.28.0
alembic>=1.11.0
redis[hiredis]>=5.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
pillow>=10.0.0
python-multipart
```

### Development

```
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.7.0
isort>=5.12.0
mypy>=1.5.0
factory-boy>=3.3.0
```

## 🛠️ Development

### Code Style

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/
```

### Adding a New Service

1. Create service in `src/services/`
2. Add dependency injection in `src/api/dependencies.py`
3. Create routes in `src/api/routes/`
4. Write tests in `tests/unit/services/`

### Adding a Caption Generator

1. Create class in `src/caption_generators/`
2. Inherit from `BaseCaptionGenerator`
3. Implement `generate_caption()` and `stream_caption()`
4. Register in `caption_generators/__init__.py`

## 🔍 Troubleshooting

### Database Connection Issues

```bash
# Test PostgreSQL connection
psql -U postgres -h localhost -p 5432 -d gallery_v2

# Check if database exists
psql -U postgres -l | grep gallery_v2
```

### Redis Connection Issues

```bash
# Test Redis connection
redis-cli ping

# Windows (Memurai)
memurai-cli ping
```

### Migration Issues

```bash
# Check current migration
alembic current

# View migration history
alembic history

# Force stamp to specific version
alembic stamp head
```

## 📖 Additional Documentation

- [Windows Setup Guide](WINDOWS_SETUP_GUIDE.md) - Detailed Windows installation
- [Architecture Plan](v2-database-migration-architecture.plan.md) - Full architecture
- [Bug Fixes](BUGFIX_METADATA.md) - Known issues and fixes

## 🤝 Contributing

### Development Workflow

1. Create a feature branch
2. Implement changes with tests
3. Run tests and linting
4. Submit pull request

### Code Quality

- Write comprehensive tests
- Follow PEP 8 style guide
- Use type hints
- Document public APIs

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- FastAPI for the excellent async web framework
- SQLAlchemy for robust ORM capabilities
- Pillow for image processing

---

**Status**: Phase 2 Complete ✅
- ✅ Service Layer Implementation
- ✅ API Routes (Core Endpoints)
- ✅ Database Models & Migrations
- ✅ Repository Pattern
- ✅ Pluggable Caption Generators
- 🚧 Testing Framework
- 🚧 Data Migration Scripts
- 📝 Documentation (In Progress)
