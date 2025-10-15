"""
Generate thumbnails for all images in database - PARALLEL VERSION.

Usage:
    python scripts/generate_thumbnails_parallel.py [--force] [--batch-size N] [--workers N]
"""
from __future__ import annotations

import sys
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Optional, Tuple
from uuid import UUID
import time

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


class ParallelThumbnailGenerator:
    """Generate thumbnails using parallel processing for speed."""
    
    def __init__(self, force: bool = False, batch_size: int = 200, max_workers: int = 4):
        self.force = force
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.stats = {
            "total": 0,
            "generated": 0,
            "skipped": 0,
            "errors": 0
        }
        self.start_time = time.time()
    
    def generate_thumbnail_sync(self, image_data: Tuple[UUID, str]) -> Tuple[bool, str, str]:
        """
        Generate thumbnail for a single image (sync function for multiprocessing).
        
        Args:
            image_data: Tuple of (image_id, original_filename)
            
        Returns:
            Tuple of (success, message, filename)
        """
        image_id, filename = image_data
        
        try:
            # Import here to avoid issues with multiprocessing
            from PIL import Image as PILImage
            import asyncio
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run async thumbnail generation
            result = loop.run_until_complete(self._generate_single_thumbnail(image_id))
            loop.close()
            
            return result + (filename,)
            
        except Exception as e:
            return False, str(e), filename
    
    async def _generate_single_thumbnail(self, image_id: UUID) -> Tuple[bool, str]:
        """Generate thumbnail using async services."""
        try:
            async with get_db_context() as db:
                storage = StorageService()
                cache = CacheService()
                thumbnail_service = ThumbnailService(db, storage, cache)
                
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
        """Run the parallel thumbnail generation process."""
        console.print(Panel.fit(
            "[bold cyan]Parallel Thumbnail Generation[/bold cyan]\n"
            f"Mode: {'FORCE REGENERATE' if self.force else 'SKIP EXISTING'}\n"
            f"Batch size: {self.batch_size}\n"
            f"Workers: {self.max_workers}\n"
            f"Max dimension: {settings.thumbnail_max_dimension}px\n"
            f"Quality: {settings.thumbnail_quality}%",
            border_style="cyan"
        ))
        
        async with get_db_context() as db:
            image_repo = ImageRepository(db)
            
            # Get total image count
            total_images = await image_repo.count()
            self.stats["total"] = total_images
            
            if total_images == 0:
                console.print("\n[yellow]No images found in database![/yellow]")
                return
            
            console.print(f"\n[bold]Processing {total_images:,} images with {self.max_workers} workers...[/bold]\n")
            
            # Process in batches with parallel workers
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[blue]{task.fields[speed]} img/s"),
                console=console
            ) as progress:
                
                task = progress.add_task("Generating thumbnails...", total=total_images, speed="0.0")
                processed = 0
                
                # Create thread pool for parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    
                    # Process in batches
                    offset = 0
                    while offset < total_images:
                        # Get batch of images
                        images = await image_repo.get_all(skip=offset, limit=self.batch_size)
                        
                        if not images:
                            break
                        
                        # Prepare data for parallel processing
                        image_data = [(img.id, img.original_filename) for img in images]
                        
                        # Submit all images in this batch to thread pool
                        future_to_data = {
                            executor.submit(self.generate_thumbnail_sync, data): data 
                            for data in image_data
                        }
                        
                        # Process results as they complete
                        batch_start = time.time()
                        for future in concurrent.futures.as_completed(future_to_data):
                            success, message, filename = future.result()
                            processed += 1
                            
                            if success:
                                self.stats["generated"] += 1
                            elif "Already exists" in message or "exists" in message.lower():
                                self.stats["skipped"] += 1
                            else:
                                self.stats["errors"] += 1
                                console.print(f"[red]✗[/red] {filename[:40]}: {message}")
                            
                            # Update progress with speed calculation
                            elapsed = time.time() - self.start_time
                            speed = processed / elapsed if elapsed > 0 else 0
                            progress.update(task, 
                                          advance=1,
                                          speed=f"{speed:.1f}",
                                          description=f"Processing: {filename[:40]}...")
                        
                        offset += self.batch_size
                        
                        # Commit after each batch (if using database session)
                        await db.commit()
        
        # Display summary
        self.display_summary()
    
    def display_summary(self):
        """Display generation summary."""
        elapsed = time.time() - self.start_time
        speed = self.stats["total"] / elapsed if elapsed > 0 else 0
        
        table = Table(title="\n[bold]Generation Summary[/bold]")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Total images", f"{self.stats['total']:,}")
        table.add_row("Thumbnails generated", f"{self.stats['generated']:,}")
        table.add_row("Skipped (already exists)", f"{self.stats['skipped']:,}")
        table.add_row("Errors", f"{self.stats['errors']:,}", 
                     style="red" if self.stats["errors"] > 0 else "green")
        table.add_row("", "")
        table.add_row("Total time", f"{elapsed:.1f} seconds")
        table.add_row("Average speed", f"{speed:.1f} images/second")
        
        console.print(table)
        
        # Performance stats
        if self.stats["generated"] > 0:
            avg_per_thumbnail = elapsed / self.stats["generated"]
            console.print(f"\n[green]✓ Average time per thumbnail: {avg_per_thumbnail:.2f} seconds[/green]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate thumbnails for all images (parallel)")
    parser.add_argument("--force", action="store_true", help="Regenerate existing thumbnails")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of images per batch")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    generator = ParallelThumbnailGenerator(
        force=args.force,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
    asyncio.run(generator.run())


if __name__ == "__main__":
    main()
