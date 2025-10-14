@echo off
echo ========================================
echo   Image Gallery v2 - Starting Service
echo ========================================
echo.

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found
    echo Please run setup/install_windows.bat first
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo ERROR: Configuration file .env not found
    echo Please run setup/setup_wizard.py first
    pause
    exit /b 1
)

echo.
echo Checking services...
echo.

REM Check PostgreSQL
pg_isready >nul 2>&1
if errorlevel 1 (
    echo WARNING: PostgreSQL is not responding
    echo Please make sure PostgreSQL is running
    choice /C YN /M "Continue anyway"
    if errorlevel 2 exit /b 1
)

REM Check Redis
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo WARNING: Redis is not responding
    echo Attempting to start Redis...
    start /B redis-server --port 6379
    timeout /t 2 >nul
    
    redis-cli ping >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Could not start Redis
        echo The service will run with reduced performance
        timeout /t 3
    )
)

echo.
echo ========================================
echo   Starting FastAPI server...
echo ========================================
echo.
echo API Documentation: http://localhost:8002/docs
echo.

REM Start the application
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8002

pause


