# Image Gallery v2 - Windows Installation Script

Write-Host "=== Image Gallery v2 - Windows Installation ===" -ForegroundColor Cyan
Write-Host ""

# Check Python version (3.10+)
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonCmd = Get-Command python -ErrorAction Stop
    $pythonVersion = python --version 2>&1 | Out-String
    
    if ($pythonVersion -match "Python 3\.(\d+)") {
        $minorVersion = [int]$Matches[1]
        if ($minorVersion -lt 10) {
            Write-Host "Error: Python 3.10+ required (found $pythonVersion)" -ForegroundColor Red
            Write-Host "Please install Python 3.10 or higher from https://www.python.org/" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "✓ Python version OK: $pythonVersion" -ForegroundColor Green
    }
    else {
        Write-Host "Error: Could not determine Python version" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error: Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check if Chocolatey is installed
$chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue

# Offer to install PostgreSQL
Write-Host "Checking PostgreSQL..." -ForegroundColor Yellow
$pgInstalled = Get-Command psql -ErrorAction SilentlyContinue

if (-not $pgInstalled) {
    Write-Host "PostgreSQL not found" -ForegroundColor Yellow
    
    if ($chocoInstalled) {
        $installPg = Read-Host "Install PostgreSQL via Chocolatey? (y/n)"
        if ($installPg -eq 'y') {
            Write-Host "Installing PostgreSQL..." -ForegroundColor Yellow
            choco install postgresql14 -y
            Write-Host "✓ PostgreSQL installed" -ForegroundColor Green
        }
        else {
            Write-Host "Please install PostgreSQL manually: https://www.postgresql.org/download/windows/" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "Please install PostgreSQL manually: https://www.postgresql.org/download/windows/" -ForegroundColor Yellow
        Write-Host "Or install Chocolatey first: https://chocolatey.org/install" -ForegroundColor Cyan
    }
}
else {
    Write-Host "✓ PostgreSQL found" -ForegroundColor Green
}

Write-Host ""

# Offer to install Redis/Memurai
Write-Host "Checking Redis..." -ForegroundColor Yellow
$redisInstalled = Get-Command redis-server -ErrorAction SilentlyContinue

if (-not $redisInstalled) {
    Write-Host "Redis not found" -ForegroundColor Yellow
    
    if ($chocoInstalled) {
        $installRedis = Read-Host "Install Memurai (Redis for Windows) via Chocolatey? (y/n)"
        if ($installRedis -eq 'y') {
            Write-Host "Installing Memurai..." -ForegroundColor Yellow
            choco install memurai-developer -y
            Write-Host "✓ Memurai installed" -ForegroundColor Green
        }
        else {
            Write-Host "Please install Redis/Memurai manually: https://www.memurai.com/" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "Please install Redis/Memurai manually: https://www.memurai.com/" -ForegroundColor Yellow
        Write-Host "Or install Chocolatey first: https://chocolatey.org/install" -ForegroundColor Cyan
    }
}
else {
    Write-Host "✓ Redis found" -ForegroundColor Green
}

Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists" -ForegroundColor Cyan
}
else {
    python -m venv .venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    }
    else {
        Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "(This may take a few minutes)" -ForegroundColor Cyan
pip install -r requirements.txt --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
}
else {
    Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Run interactive setup wizard
Write-Host "Running interactive setup wizard..." -ForegroundColor Green
Write-Host "(Press Ctrl+C to cancel)" -ForegroundColor Cyan
Write-Host ""

python setup/setup_wizard.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Installation Complete ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start the service, run:" -ForegroundColor Cyan
    Write-Host "  run_windows.bat" -ForegroundColor White
    Write-Host ""
}
else {
    Write-Host ""
    Write-Host "Setup wizard failed or was cancelled" -ForegroundColor Yellow
    Write-Host "You can run it again with:" -ForegroundColor Cyan
    Write-Host "  .venv\Scripts\activate" -ForegroundColor White
    Write-Host "  python setup/setup_wizard.py" -ForegroundColor White
    Write-Host ""
}

# Keep window open
Read-Host "Press Enter to exit"


