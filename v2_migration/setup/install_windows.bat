@echo off
REM Batch wrapper for PowerShell installation script

echo Starting installation...
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running with administrator privileges
) else (
    echo WARNING: Not running as administrator
    echo Some installations may fail
    echo.
)

REM Run PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1"

pause


