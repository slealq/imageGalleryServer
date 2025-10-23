# Quick Start - Image Gallery v2

Get up and running in 5 minutes!

## Prerequisites

✅ Python 3.10+
✅ PostgreSQL (running)
✅ Redis/Memurai (running)
✅ Virtual environment activated

## Step 1: Setup (First Time Only)

```bash
# Navigate to v2_migration directory
cd v2_migration

# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Ensure migrations are up to date
alembic upgrade head
```

## Step 2: Start the Server

```bash
python src/main.py
```

You should see:
```
🚀 Starting Image Gallery v2...
📁 Storage root: C:/gallery_storage
🗄️  Database: localhost:5432/gallery_v2
📝 Caption generator: dummy
INFO:     Uvicorn running on http://0.0.0.0:8002
```

## Step 3: Access the API

Open your browser to:
- **Interactive Docs**: http://localhost:8002/docs
- **Health Check**: http://localhost:8002/api/v2/health

## Step 4: Try the API

### Create a Photoset

```bash
curl -X POST http://localhost:8002/api/v2/photosets \
  -H "Content-Type: application/json" \
  -d '{"name": "My First Photoset", "year": 2024}'
```

You'll get a response with the photoset ID.

### Check Health

```bash
curl http://localhost:8002/api/v2/health
```

Should return:
```json
{
  "status": "healthy",
  "database": "healthy",
  "cache": "healthy"
}
```

## Next Steps

1. **Explore the API**: Visit http://localhost:8002/docs
2. **Read the full README**: [README.md](README.md)
3. **Check API usage examples**: [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md)
4. **Review implementation details**: [PHASE_2_COMPLETE.md](PHASE_2_COMPLETE.md)

## Common Issues

### "alembic: command not found"
Make sure virtual environment is activated:
```bash
.\.venv\Scripts\Activate.ps1
```

### "Database connection failed"
Ensure PostgreSQL is running and `.env` is configured correctly.

### "Redis connection unavailable"
Check if Redis/Memurai is running:
```bash
redis-cli ping  # Should return PONG
```

## What's Available Now

✅ Photoset management
✅ Image retrieval
✅ Thumbnail generation
✅ Caption generation (dummy mode)
✅ Crop creation
✅ Tag management
✅ Full API documentation

## What's Coming

🚧 Image upload
🚧 Data migration scripts
🚧 Testing framework
🚧 Batch operations
🚧 Export to ZIP

---

**Need help?** Check the [WINDOWS_SETUP_GUIDE.md](WINDOWS_SETUP_GUIDE.md)






