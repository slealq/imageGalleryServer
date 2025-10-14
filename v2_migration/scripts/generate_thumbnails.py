"""
Generate thumbnails for all images in database.

Usage:
    python scripts/generate_thumbnails.py [--force] [--batch-size N]
"""
from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.database import get_db_context
from src.repositories import ImageRepository
from src.services import ThumbnailService, StorageService, CacheService

console = Console()


class ThumbnailGenerator:
    """Generate thumbnails for all images."""
    
    def __init__(self, force: bool = False, batch_size: int = 100):
        self.force = force
        self.batch_size = batch_size
        self.stats = {
            "total": 0,
            "generated": 0,
            "skipped": 0,
            "errors": 0
        }
    
    async def generate_for_image(
        self,
        image_id: str,
        thumbnail_service: ThumbnailService
    ) -> tuple[bool, str]:
        """
        Generate thumbnail for a single image.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Generate thumbnail (single scaled-down version)
            thumbnail = await thumbnail_service.generate_thumbnail(
                image_id,
                force=self.force
            )
            
            if thumbnail:
                return True, f"{thumbnail.width}x{thumbnail.height}"
            else:
                return False, "No thumbnail generated"
                
        except Exception as e:
            return False, str(e)
    
    async def run(self):
        """Run the thumbnail generation process."""
        console.print(Panel.fit(
            "[bold cyan]Thumbnail Generation[/bold cyan]\n"
            f"Mode: {'FORCE REGENERATE' if self.force else 'SKIP EXISTING'}\n"
            f"Batch size: {self.batch_size}\n"
            f"Max dimension: {settings.thumbnail_max_dimension}px\n"
            f"Quality: {settings.thumbnail_quality}%",
            border_style="cyan"
        ))
        
        async with get_db_context() as db:
            image_repo = ImageRepository(db)
            storage = StorageService()
            cache = CacheService()
            thumbnail_service = ThumbnailService(db, storage, cache)
            
            # Get total image count
            total_images = await image_repo.count()
            self.stats["total"] = total_images
            
            if total_images == 0:
                console.print("\n[yellow]No images found in database![/yellow]")
                return
            
            console.print(f"\n[bold]Processing {total_images} images...[/bold]\n")
            
            # Process in batches
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Generating thumbnails...", total=total_images)
                
                offset = 0
                while offset < total_images:
                    # Get batch of images
                    images = await image_repo.get_all(skip=offset, limit=self.batch_size)
                    
                    for image in images:
                        progress.update(task, description=f"Processing: {image.original_filename[:40]}...")
                        
                        success, message = await self.generate_for_image(
                            image.id,
                            thumbnail_service
                        )
                        
                        if success:
                            self.stats["generated"] += 1
                        elif message == "Already exists":
                            self.stats["skipped"] += 1
                        else:
                            self.stats["errors"] += 1
                            console.print(f"[red]✗[/red] {image.original_filename}: {message}")
                        
                        progress.update(task, advance=1)
                    
                    offset += self.batch_size
                    
                    # Commit after each batch
                    await db.commit()
        
        # Display summary
        self.display_summary()
    
    def display_summary(self):
        """Display generation summary."""
        table = Table(title="\n[bold]Generation Summary[/bold]")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Total images", str(self.stats["total"]))
        table.add_row("Thumbnails generated", str(self.stats["generated"]))
        table.add_row("Skipped (already exists)", str(self.stats["skipped"]))
        table.add_row("Errors", str(self.stats["errors"]), style="red" if self.stats["errors"] > 0 else "green")
        
        console.print(table)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate thumbnails for all images")
    parser.add_argument("--force", action="store_true", help="Regenerate existing thumbnails")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of images per batch")
    
    args = parser.parse_args()
    
    generator = ThumbnailGenerator(
        force=args.force,
        batch_size=args.batch_size
    )
    asyncio.run(generator.run())


if __name__ == "__main__":
    main()

