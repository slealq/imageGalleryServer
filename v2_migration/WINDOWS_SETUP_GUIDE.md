# Windows Setup Guide - Image Gallery v2

Complete step-by-step guide for setting up the Image Gallery v2 service on Windows.

## Prerequisites

Before you begin, ensure you have:
- Windows 10/11
- Administrator access (required for some installations)
- Internet connection

---

## Step 1: Install Python 3.10+

1. Download Python from [python.org](https://www.python.org/downloads/)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```powershell
   python --version
   # Should show Python 3.10.x or higher
   ```

---

## Step 2: Install PostgreSQL

### Option A: Direct Download (Recommended)

1. Download PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
2. Run the installer
3. During installation:
   - Set a password for the `postgres` user (remember this!)
   - Default port: `5432`
   - Install pgAdmin 4 (graphical tool)
4. Verify installation:
   ```powershell
   # Add PostgreSQL to PATH if not done automatically
   # Default location: C:\Program Files\PostgreSQL\14\bin
   
   # Test connection
   psql --version
   ```

### Option B: Using Chocolatey

If you have [Chocolatey](https://chocolatey.org/) installed:

```powershell
# Run as Administrator
choco install postgresql14 -y
```

### Create the Database

Using pgAdmin or command line:

```powershell
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE gallery_v2;

# Exit
\q
```

---

## Step 3: Install Redis (Memurai)

Redis is not natively available on Windows. Use Memurai instead (Redis-compatible).

### Download and Install

1. Download Memurai from [memurai.com](https://www.memurai.com/get-memurai)
2. **Important**: You must install as Administrator

### Installation Steps

```powershell
# Open PowerShell as Administrator (Right-click → Run as Administrator)

# Navigate to your downloads folder
cd ~\Downloads

# Install using msiexec
msiexec /i memurai-developer-x.x.x.msi

# Follow the installation wizard
```

### Verify Installation

```powershell
# Check if Redis service is running
redis-cli ping
# Should return: PONG
```

### Troubleshooting Redis

If `redis-cli` is not found:
```powershell
# Add Memurai to PATH
# Default location: C:\Program Files\Memurai\

# Or start the service manually
net start Memurai
```

---

## Step 4: Setup the Project

### Clone/Navigate to Project

```powershell
cd C:\playground\imageGalleryServer\v2_migration
```

### Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

**If you get an execution policy error:**

```powershell
# Run this first
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating again
.\.venv\Scripts\Activate.ps1
```

**Successful activation**: You'll see `(.venv)` at the start of your prompt:
```
(.venv) PS C:\playground\imageGalleryServer\v2_migration>
```

### Install Python Dependencies

```powershell
# Make sure virtual environment is activated!

# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

This will install:
- FastAPI
- SQLAlchemy
- PostgreSQL drivers (asyncpg, psycopg2)
- Redis client
- Alembic
- PIL/Pillow
- And all other dependencies

---

## Step 5: Run Setup Wizard

The setup wizard will configure all paths and settings:

```powershell
# Make sure you're in the v2_migration directory with venv activated
python setup/setup_wizard.py
```

### What the Wizard Will Ask:

1. **Storage Location**: Where to store images, thumbnails, etc.
   - Suggested: `C:\gallery_storage` or `.\storage`

2. **Existing Data** (Optional): Paths to migrate existing data
   - Raw archives (zip/rar files)
   - Metadata JSON files
   - Existing images, captions, crops

3. **Database Configuration**:
   - Host: `localhost`
   - Port: `5432`
   - Database: `gallery_v2`
   - User: `postgres`
   - Password: (the one you set during PostgreSQL installation)

4. **Redis Configuration**:
   - Host: `localhost`
   - Port: `6379`
   - Database: `0`

5. **Caption Generator**:
   - Choose `dummy` for testing
   - Choose `unsloth` if you have the model
   - Choose `none` for manual captions only

6. **Performance Settings**: Use defaults or customize

The wizard will:
- ✅ Create `.env` file with your configuration
- ✅ Create storage directories
- ✅ Test database connection
- ✅ Test Redis connection

---

## Step 6: Setup Database Migrations

### Create Migrations Directory

**Important**: This directory must exist before creating migrations:

```powershell
# Create the versions directory
mkdir migrations\versions
```

### Generate Initial Migration

```powershell
# Generate migration from your models
alembic revision --autogenerate -m "initial schema"
```

**Expected output:**
```
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.autogenerate.compare] Detected added table 'photosets'
INFO  [alembic.autogenerate.compare] Detected added table 'images'
... (more tables)
Generating ...\migrations\versions\xxxxx_initial_schema.py ... done
```

### Run Migrations

```powershell
# Apply migrations to create tables
alembic upgrade head
```

**Expected output:**
```
INFO  [alembic.runtime.migration] Running upgrade  -> xxxxx, initial schema
```

### Verify Migrations

```powershell
# Check current migration status
alembic current

# Should show something like:
# xxxxx (head)
```

---

## Step 7: Test the Foundation

Run the comprehensive foundation tests:

```powershell
python scripts/test_foundation.py
```

**Expected output:**
```
╭─────────────────────────────────────────────╮
│  Foundation Layer Test Suite                │
╰─────────────────────────────────────────────╯

1. Testing Configuration
✓ Configuration loaded successfully

2. Testing Database Connection
✓ Database connection successful

3. Testing Redis Connection
✓ Redis connection successful

4. Testing Database Schema
✓ All tables exist

5. Testing Models & Repositories
✓ All CRUD operations successful

6. Testing Caption Generators
✓ Dummy generator working

==================================================
Test Summary

Results: 6/6 tests passed

✓ All foundation tests passed!
```

---

## Complete Setup Summary

Here's the full command sequence from scratch:

```powershell
# 1. Navigate to project
cd C:\playground\imageGalleryServer\v2_migration

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run setup wizard
python setup/setup_wizard.py

# 5. Create migrations directory
mkdir migrations\versions

# 6. Generate and run migrations
alembic revision --autogenerate -m "initial schema"
alembic upgrade head

# 7. Test everything
python scripts/test_foundation.py
```

---

## Common Issues and Solutions

### Issue: "execution policy" error when activating venv

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "alembic: command not found"

**Solution:** Virtual environment is not activated. Run:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Issue: PostgreSQL connection fails

**Solutions:**
1. Check if PostgreSQL is running:
   ```powershell
   # Windows service
   net start postgresql-x64-14
   ```

2. Verify password is correct in `.env` file

3. Test connection manually:
   ```powershell
   psql -U postgres -d gallery_v2
   ```

### Issue: Redis/Memurai not responding

**Solutions:**
1. Start Memurai service:
   ```powershell
   net start Memurai
   ```

2. Or start manually:
   ```powershell
   redis-server
   ```

### Issue: "FileNotFoundError" when running alembic

**Solution:** Create the migrations/versions directory:
```powershell
mkdir migrations\versions
```

### Issue: "metadata is reserved" error

**Solution:** This was fixed in the codebase. Make sure you have the latest code where `metadata` columns are renamed to `extra_metadata`.

---

## Directory Structure After Setup

```
v2_migration/
├── .venv/                    # Virtual environment
├── .env                      # Your configuration (DO NOT COMMIT)
├── migrations/
│   └── versions/
│       └── xxxxx_initial_schema.py
├── storage/                  # Created by wizard
│   ├── images/
│   ├── thumbnails/
│   ├── crops/
│   └── archives/
└── ... (source code)
```

---

## Next Steps

After successful setup:

### Option 1: Start Building (Developer)
Continue with service layer and API implementation

### Option 2: Migrate Data (User)
If you have existing data:
```powershell
python scripts/bootstrap_data.py
```

### Option 3: Start the Service
Once API is implemented:
```powershell
run_windows.bat
# Or
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8002
```

Access the API at: http://localhost:8002/docs

---

## Maintenance Commands

### Update Dependencies
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --upgrade
```

### Create New Migration (after model changes)
```powershell
alembic revision --autogenerate -m "description of changes"
alembic upgrade head
```

### Rollback Migration
```powershell
alembic downgrade -1
```

### Check Database Status
```powershell
alembic current
alembic history
```

### Deactivate Virtual Environment
```powershell
deactivate
```

---

## Automated Setup (Future)

You can also use the automated installation script:

```powershell
.\setup\install_windows.bat
```

This will:
- Check Python version
- Offer to install PostgreSQL and Redis
- Create virtual environment
- Install dependencies
- Run setup wizard

---

## Troubleshooting Checklist

Before asking for help, verify:

- [ ] Python 3.10+ is installed and in PATH
- [ ] Virtual environment is activated (see `(.venv)` in prompt)
- [ ] PostgreSQL is installed and running
- [ ] Redis/Memurai is installed and running
- [ ] Database `gallery_v2` exists
- [ ] `.env` file exists in v2_migration directory
- [ ] `migrations/versions/` directory exists
- [ ] All dependencies installed (`pip install -r requirements.txt`)

Run the verification script:
```powershell
python scripts/verify_setup.py
```

---

## Support

For issues:
1. Check this guide first
2. Run `python scripts/verify_setup.py`
3. Check error messages carefully
4. Search for similar issues in documentation

---

## Summary

✅ Install Python 3.10+  
✅ Install PostgreSQL 14+ (with pgAdmin)  
✅ Install Memurai (Redis for Windows) as Administrator  
✅ Create virtual environment  
✅ Install Python dependencies  
✅ Run setup wizard  
✅ Create migrations/versions directory  
✅ Generate and apply database migrations  
✅ Test with foundation test suite  

**You're ready to go!** 🎉


