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

## 📦 Data Migration

If you're migrating from the old file-based system, use these scripts to import your existing data.

### Overview

The migration process consists of three main steps:

1. **Extract Archives**: Extract ZIP/RAR photoset files into organized storage
2. **Import Metadata**: Parse JSON metadata files and populate the database
3. **Generate Thumbnails**: Create thumbnail images for all imported photos

### Quick Migration

Use the bootstrap script to run the entire migration:

```bash
# Dry run to preview what will happen
python scripts/bootstrap_data.py --dry-run --archives "D:\old_archives" --metadata "D:\old_metadata"

# Actual migration
python scripts/bootstrap_data.py --archives "D:\old_archives" --metadata "D:\old_metadata"
```

### Step-by-Step Migration

If you prefer to run each step individually:

#### 1. Extract Archives

Extract ZIP/RAR files containing photoset images:

```bash
# Extract archives from a specific directory
python scripts/extract_archives.py --source "D:\photosets_archives"

# Dry run to preview
python scripts/extract_archives.py --source "D:\photosets_archives" --dry-run
```

This will:
- Find all `.zip` and `.rar` files in the source directory
- Extract only image files (jpg, png, gif, webp, etc.)
- Organize extracted images into `{IMAGES_DIR}/{photoset_name}/`
- Skip photosets that have already been extracted

#### 2. Import Metadata

Import photoset metadata from JSON files into the database:

```bash
# Import from metadata directory
python scripts/import_metadata.py --source "D:\photoset_metadata"

# Dry run to preview
python scripts/import_metadata.py --source "D:\photoset_metadata" --dry-run
```

This will:
- Read all JSON files from the source directory
- Create photoset records in the database
- Link images to photosets
- Import tags and actors
- Skip photosets that already exist

**Expected JSON format:**
```json
{
  "url": "https://example.com/photoset",
  "date": "2024-01-15",
  "year": 2024,
  "original_filename": "photoset_archive.zip",
  "actors": ["Actor Name"],
  "tags": ["tag1", "tag2"]
}
```

#### 3. Generate Thumbnails

Generate thumbnails for all images in the database:

```bash
# Generate thumbnails for all images
python scripts/generate_thumbnails.py

# Force regenerate even if thumbnails exist
python scripts/generate_thumbnails.py --force

# Process in larger batches (default is 100)
python scripts/generate_thumbnails.py --batch-size 200
```

This will:
- Generate 3 sizes for each image: small (256px), medium (512px), large (1024px)
- Store thumbnails in `{THUMBNAILS_DIR}/{image_id}/{size}.jpg`
- Skip images that already have thumbnails (unless `--force` is used)
- Process images in batches for better performance

#### 4. Verify Migration

Verify that the migration completed successfully:

```bash
python scripts/verify_migration.py
```

This will check:
- Database record counts (photosets, images, captions, crops, tags)
- File existence (verify images exist on disk)
- Thumbnail coverage
- Database relationship integrity
- Storage directory structure

### Migration Options

**Bootstrap Script Options:**
- `--dry-run`: Preview changes without modifying anything
- `--skip-extraction`: Skip archive extraction (if already done)
- `--skip-thumbnails`: Skip thumbnail generation
- `--archives PATH`: Path to archive directory
- `--metadata PATH`: Path to metadata JSON directory

**Resumability:**
All migration scripts are designed to be resumable:
- They skip items that have already been processed
- They can be safely interrupted and rerun
- They won't create duplicates

### Example: Complete Migration Workflow

```bash
# 1. Run a dry run first to verify everything
python scripts/bootstrap_data.py --dry-run \
  --archives "D:\old_data\archives" \
  --metadata "D:\old_data\metadata"

# 2. Run the actual migration
python scripts/bootstrap_data.py \
  --archives "D:\old_data\archives" \
  --metadata "D:\old_data\metadata"

# 3. Verify migration completed successfully
python scripts/verify_migration.py

# 4. Start the server
python src/main.py
```

### Troubleshooting Migration

**No archives found:**
- Verify the `--archives` path exists and contains `.zip` or `.rar` files
- Check file permissions

**No metadata files found:**
- Verify the `--metadata` path exists and contains `.json` files
- Check that JSON files are named to match photoset names

**Images not found during metadata import:**
- Make sure you run extract_archives.py first
- Verify that photoset names match between archives and metadata files

**Thumbnails not generating:**
- Check that images exist in the database (run import_metadata.py first)
- Verify PIL/Pillow can open the image files
- Check disk space in the thumbnails directory

**RAR extraction fails:**
- Install WinRAR or unrar: `choco install winrar`
- Or extract RAR files manually and convert to ZIP

## 🧪 Testing

### Quick Start

```bash
# Create test database (first time only)
python scripts/create_test_db.py

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit

# Run only integration tests
pytest tests/integration

# Stop on first failure
pytest -x
```

### Using the Test Runner

```bash
# Convenient shortcuts
python scripts/run_tests.py              # All tests
python scripts/run_tests.py --unit       # Unit tests only
python scripts/run_tests.py --integration # Integration tests
python scripts/run_tests.py --cov        # With coverage
python scripts/run_tests.py --fast       # Stop on first failure
```

### Test Structure

```
tests/
├── conftest.py          # Pytest configuration & fixtures
├── factories.py         # Test data factories
├── unit/                # Unit tests
│   └── test_repositories.py
└── integration/         # Integration tests
    ├── test_api_photosets.py
    └── test_api_health.py
```

### Key Fixtures

- `db_session` - Clean database session for each test
- `client` - Async HTTP client for API testing
- `sample_photoset_data` - Sample photoset data
- `sample_image_data` - Sample image data
- `sample_caption_data` - Sample caption data

### Test Factories

```python
from tests.factories import PhotosetFactory, ImageFactory, CaptionFactory

# Create test data
photoset = PhotosetFactory.create(name="Custom Name")
image = ImageFactory.create(photoset_id=photoset.id)
caption = CaptionFactory.create(image_id=image.id)
```

### Coverage Reports

After running tests with `--cov`, open `htmlcov/index.html` in your browser to see detailed coverage.

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
