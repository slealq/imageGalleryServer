# Setup Guide

This guide explains how to set up and configure the Image Gallery v2 service.

## Prerequisites

- **Python 3.10 or higher**
- **PostgreSQL 14+**
- **Redis 7+** (optional but recommended)

## Installation Methods

### Method 1: Automated Installation (Windows)

The easiest way to install on Windows:

1. Double-click `setup/install_windows.bat`
2. Follow the interactive prompts
3. The script will:
   - Check Python version
   - Offer to install PostgreSQL and Redis via Chocolatey
   - Create a virtual environment
   - Install all dependencies
   - Run the configuration wizard

### Method 2: Manual Installation

For other platforms or manual control:

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup wizard:**
   ```bash
   python setup/setup_wizard.py
   ```

## Configuration Wizard

The interactive wizard will guide you through:

### 1. Storage Configuration
- Where to store images, thumbnails, crops, and archives
- Creates all necessary directories
- Validates write permissions

### 2. Data Migration (Optional)
- Paths to existing zip/rar archives
- Paths to existing metadata JSON files
- Paths to existing images, captions, crops

### 3. Database Configuration
- PostgreSQL connection details (host, port, database, credentials)
- Tests connection
- Creates database if it doesn't exist

### 4. Redis Configuration
- Redis connection details (host, port, database number)
- Tests connection
- Optional but recommended for performance

### 5. Caption Generation
- Choose caption generator:
  - **dummy**: For testing only
  - **unsloth**: Local AI model (requires model path)
  - **none**: Manual captions only
- Configure model settings if using unsloth

### 6. Performance Tuning
- Image cache size
- Thumbnail dimensions
- Worker threads
- Logging preferences

### 7. Review & Confirm
- Review all settings
- Confirm to proceed with initialization

### 8. System Initialization
- Creates all storage directories
- Generates `.env` configuration file
- Runs database migrations
- Saves migration configuration

## Running the Service

### Windows

Simply run:
```batch
run_windows.bat
```

### Other Platforms

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the service
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8002
```

## Data Migration

If you configured existing data during setup:

```bash
python scripts/bootstrap_data.py
```

This will:
1. Extract photoset archives
2. Import metadata
3. Migrate captions and crops
4. Generate thumbnails
5. Validate data integrity

## Accessing the Service

Once running, access:

- **API Documentation (Swagger)**: http://localhost:8002/docs
- **API Documentation (ReDoc)**: http://localhost:8002/redoc
- **Health Check**: http://localhost:8002/api/v2/health

## Troubleshooting

### Virtual Environment Issues

If activation fails:
```bash
# Windows
python -m venv --clear .venv

# Ensure ExecutionPolicy allows scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Database Connection Issues

1. Ensure PostgreSQL is running:
   ```bash
   # Check status
   pg_isready
   
   # Windows service
   net start postgresql-x64-14
   ```

2. Verify credentials and connection:
   ```bash
   psql -h localhost -U postgres -d gallery_v2
   ```

### Redis Issues

Redis is optional. If not available:
- The service will work with reduced performance
- Some caching features will be disabled

To start Redis:
```bash
# Windows
redis-server

# Linux
sudo systemctl start redis
```

### Migration Issues

If database migrations fail:
```bash
# Check current migration status
alembic current

# Run migrations manually
alembic upgrade head

# Rollback if needed
alembic downgrade -1
```

## Configuration Files

After setup, you'll have:

- **`.env`**: Environment configuration (DO NOT commit)
- **`config.json`**: Migration source paths (if configured)
- **`alembic.ini`**: Database migration configuration

## Next Steps

1. **Start the service**: `run_windows.bat` or `uvicorn` command
2. **Access API docs**: http://localhost:8002/docs
3. **Run data migration**: `python scripts/bootstrap_data.py` (if applicable)
4. **Test the API**: Use Swagger UI to explore endpoints

## Support

For issues or questions:
- Check the logs in `storage/logs/app.log` (if configured)
- Review the main README.md
- Open an issue on GitHub


