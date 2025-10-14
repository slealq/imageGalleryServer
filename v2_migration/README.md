# Image Gallery v2 - Database Migration

A scalable, database-backed image gallery service with native Windows support, featuring photoset management, image browsing, caption generation, cropping, tagging, and future-ready for embeddings and semantic search.

## Key Features

- 🗄️ **PostgreSQL Database**: Robust relational storage for all metadata
- ⚡ **Redis Caching**: High-performance caching layer
- 🖼️ **Image Management**: Upload, organize, and browse images by photosets
- ✂️ **Cropping**: Create and manage image crops
- 📝 **Captions**: Pluggable caption generation (Unsloth, OpenAI, etc.)
- 🏷️ **Tagging**: Flexible tagging system for photosets and images
- 🔍 **Filtering**: Search by actors, tags, years, and more
- 📦 **Export**: Export selected images with crops and captions
- 🚀 **Future Ready**: Database schema supports embeddings and vector search

## Installation

### Prerequisites

- Python 3.10 or higher
- PostgreSQL 14+
- Redis 7+

### Windows Installation

Run the automated installation script:

```powershell
.\setup\install_windows.ps1
```

This will:
1. Check Python version
2. Offer to install PostgreSQL and Redis via Chocolatey
3. Create a virtual environment
4. Install Python dependencies
5. Run the interactive setup wizard

### Manual Installation

1. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup wizard**:
   ```bash
   python setup/setup_wizard.py
   ```

## Configuration

The interactive setup wizard will guide you through:

1. **Storage paths**: Where to store images, thumbnails, crops
2. **Migration sources**: Paths to existing data (optional)
3. **Database**: PostgreSQL connection details
4. **Redis**: Redis connection details
5. **Caption generation**: Choose generator type (Unsloth, dummy, etc.)
6. **Performance tuning**: Cache sizes, thumbnail dimensions

All configuration is stored in `.env` file (never commit this file).

## Running the Service

### Windows

```batch
run_windows.bat
```

### Cross-platform

```bash
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8002
```

Access the API documentation at: `http://localhost:8002/docs`

## Data Migration

If you have existing data, run the migration script after setup:

```bash
python scripts/bootstrap_data.py
```

This will:
1. Extract photoset archives (zip/rar)
2. Import metadata from JSON files
3. Migrate existing captions and crops
4. Generate thumbnails
5. Validate data integrity

## Testing

Run the test suite:

```bash
pytest
```

With coverage report:

```bash
pytest --cov=src --cov-report=html
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`

## Project Structure

```
v2_migration/
├── src/                  # Application source code
│   ├── api/             # API routes and middleware
│   ├── core/            # Configuration and core utilities
│   ├── models/          # Database and Pydantic models
│   ├── repositories/    # Data access layer
│   ├── services/        # Business logic
│   └── caption_generators/  # Pluggable caption system
├── scripts/             # Migration and utility scripts
├── tests/               # Test suite
├── migrations/          # Alembic database migrations
├── setup/              # Installation and setup scripts
└── storage/            # Runtime storage (created by wizard)
```

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# With coverage
pytest --cov=src --cov-report=term-missing
```

### Code Quality

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint
flake8 src tests

# Type checking
mypy src
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## License

[Your License Here]

## Support

For issues and questions, please open an issue on GitHub.


