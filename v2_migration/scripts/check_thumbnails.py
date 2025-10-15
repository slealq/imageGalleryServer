"""
Check if thumbnails are being stored to disk.

Usage:
    python scripts/check_thumbnails.py
"""
from __future__ import annotations

import sys
import asyncio
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.database import get_db_context
from src.repositories import ImageRepository
from src.services.storage_service import StorageService

console = Console()


async def main():
    """Check thumbnail storage status."""
    console.print("[bold cyan]Thumbnail Storage Check[/bold cyan]")
    console.print()
    
    # Show configuration
    console.print(f"[bold]Configuration:[/bold]")
    console.print(f"  Thumbnails directory: {settings.thumbnails_dir}")
    console.print(f"  Directory exists: {settings.thumbnails_dir.exists()}")
    console.print()
    
    storage = StorageService()
    
    # Check if thumbnails directory exists
    if not settings.thumbnails_dir.exists():
        console.print("[yellow]Thumbnails directory doesn't exist. Creating it...[/yellow]")
        settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    
    async with get_db_context() as db:
        image_repo = ImageRepository(db)
        
        # Get sample of images
        console.print("Getting sample of images...")
        images = await image_repo.get_all(limit=10)
        
        if not images:
            console.print("[yellow]No images found in database![/yellow]")
            return
        
        console.print(f"Checking first {len(images)} images...")
        console.print()
        
        # Check each image for thumbnails
        table = Table()
        table.add_column("Image", style="cyan", width=40)
        table.add_column("Thumbnail Path", style="green", width=50)
        table.add_column("Exists", justify="center")
        table.add_column("Size", justify="right")
        
        found_count = 0
        total_size = 0
        
        for image in images:
            thumb_path = storage.get_thumbnail_path(image.id, image.photoset_id)
            
            if thumb_path.exists():
                size_mb = thumb_path.stat().st_size / (1024 * 1024)
                table.add_row(
                    image.original_filename[:40],
                    str(thumb_path)[-50:],  # Show last 50 chars
                    "[green]✓[/green]",
                    f"{size_mb:.1f} MB"
                )
                found_count += 1
                total_size += size_mb
            else:
                table.add_row(
                    image.original_filename[:40],
                    str(thumb_path)[-50:],
                    "[red]✗[/red]",
                    "-"
                )
        
        console.print(table)
        console.print()
        
        # Summary
        console.print(f"[bold]Summary:[/bold]")
        console.print(f"  Found thumbnails: {found_count}/{len(images)}")
        console.print(f"  Total size: {total_size:.1f} MB")
        
        if found_count == 0:
            console.print(f"\n[yellow]No thumbnails found. Run the generator:[/yellow]")
            console.print(f"  python scripts/generate_thumbnails_simple.py")
        elif found_count < len(images):
            console.print(f"\n[yellow]Some thumbnails missing. Run generator to complete:[/yellow]")
            console.print(f"  python scripts/generate_thumbnails_simple.py")
        else:
            console.print(f"\n[green]✓ All sample images have thumbnails![/green]")
        
        # Count total thumbnail files
        if settings.thumbnails_dir.exists():
            thumbnail_files = list(settings.thumbnails_dir.glob("*.jpg"))
            console.print(f"\n[bold]Total thumbnail files on disk: {len(thumbnail_files)}[/bold]")
            
            if thumbnail_files:
                total_disk_size = sum(f.stat().st_size for f in thumbnail_files) / (1024 * 1024)
                console.print(f"Total disk usage: {total_disk_size:.1f} MB")
                
                # Show sample paths
                console.print(f"\nSample thumbnail files:")
                for thumb_file in thumbnail_files[:5]:
                    size_mb = thumb_file.stat().st_size / (1024 * 1024)
                    console.print(f"  {thumb_file.name}: {size_mb:.1f} MB")
                if len(thumbnail_files) > 5:
                    console.print(f"  ... and {len(thumbnail_files) - 5} more")


if __name__ == "__main__":
    asyncio.run(main())
