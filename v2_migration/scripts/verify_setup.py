"""
Quick setup verification script.

Run this before test_foundation.py to check prerequisites.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

console = Console()


def check_python_version():
    """Check Python version."""
    import sys
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        console.print(f"[green]✓ Python {version.major}.{version.minor}.{version.micro}[/green]")
        return True
    else:
        console.print(f"[red]✗ Python {version.major}.{version.minor} (need 3.10+)[/red]")
        return False


def check_dependencies():
    """Check if key dependencies are installed."""
    required = [
        "fastapi",
        "sqlalchemy",
        "asyncpg",
        "redis",
        "alembic",
        "pydantic",
        "pydantic_settings",
        "pillow",
        "questionary",
        "rich",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            console.print(f"[green]✓ {package}[/green]")
        except ImportError:
            console.print(f"[red]✗ {package}[/red]")
            missing.append(package)
    
    if missing:
        console.print(f"\n[yellow]Missing packages: {', '.join(missing)}[/yellow]")
        console.print("[cyan]Run: pip install -r requirements.txt[/cyan]")
        return False
    
    return True


def check_env_file():
    """Check if .env file exists."""
    env_file = Path(".env")
    if env_file.exists():
        console.print("[green]✓ .env file exists[/green]")
        return True
    else:
        console.print("[red]✗ .env file not found[/red]")
        console.print("[cyan]Run: python setup/setup_wizard.py[/cyan]")
        return False


def check_directories():
    """Check if storage directories exist."""
    try:
        from src.core.config import settings
        
        dirs = [
            settings.storage_root,
            settings.images_dir,
            settings.thumbnails_dir,
            settings.crops_dir,
            settings.archives_dir,
        ]
        
        all_exist = True
        for dir_path in dirs:
            if dir_path.exists():
                console.print(f"[green]✓ {dir_path}[/green]")
            else:
                console.print(f"[yellow]⚠ {dir_path} (will be created)[/yellow]")
                all_exist = False
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Cannot load configuration: {e}[/red]")
        return False


def main():
    """Run all checks."""
    console.print(Panel.fit(
        "[bold cyan]Setup Verification[/bold cyan]\n"
        "Checking prerequisites before testing",
        border_style="cyan"
    ))
    
    console.print("\n[bold]1. Python Version[/bold]")
    python_ok = check_python_version()
    
    console.print("\n[bold]2. Dependencies[/bold]")
    deps_ok = check_dependencies()
    
    console.print("\n[bold]3. Configuration[/bold]")
    config_ok = check_env_file()
    
    if config_ok:
        console.print("\n[bold]4. Storage Directories[/bold]")
        dirs_ok = check_directories()
    else:
        dirs_ok = False
    
    # Summary
    console.print("\n" + "="*50)
    
    if python_ok and deps_ok and config_ok:
        console.print("[bold green]✓ Setup verification passed![/bold green]")
        console.print("\n[cyan]You can now run:[/cyan]")
        console.print("  [white]python scripts/test_foundation.py[/white]")
    else:
        console.print("[bold yellow]⚠ Some checks failed[/bold yellow]")
        console.print("\n[cyan]Steps to fix:[/cyan]")
        
        if not python_ok:
            console.print("  1. Install Python 3.10+")
        if not deps_ok:
            console.print("  2. Install dependencies: pip install -r requirements.txt")
        if not config_ok:
            console.print("  3. Run setup wizard: python setup/setup_wizard.py")


if __name__ == "__main__":
    main()


