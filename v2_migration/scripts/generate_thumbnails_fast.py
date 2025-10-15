"""
Generate thumbnails for all images - FAST VERSION (no Redis dependency).

Usage:
    python scripts/generate_thumbnails_fast.py [--force] [--batch-size N] [--workers N]
"""
from __future__ import annotations

import sys
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Optional, Tuple
from uuid import UUID
import time
from PIL import Image as PILImage

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.database import get_db_context
from src.repositories import ImageRepository
from src.services.storage_service import StorageService
from src.models.database import Thumbnail

console = Console()


class FastThumbnailGenerator:
    """Generate thumbnails without Redis dependency for speed."""
    
    def __init__(self, force: bool = False, batch_size: int = 500, max_workers: int = 6):
        self.force = force
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.storage = StorageService()
        self.stats = {
            "total": 0,
            "generated": 0,
            "skipped": 0,
            "errors": 0
        }
        self.start_time = time.time()
    
    def generate_thumbnail_sync(self, image_data: Tuple[UUID, str, str, UUID]) -> Tuple[bool, str, str]:
        """
        Generate thumbnail for a single image (sync function for multiprocessing).
        
        Args:
            image_data: Tuple of (image_id, filename, file_path, photoset_id)
            
        Returns:
            Tuple of (success, message, filename)
        """
        image_id, filename, file_path, photoset_id = image_data
        
        try:
            # Check if thumbnail already exists
            thumb_path = self.storage.get_thumbnail_path(image_id, photoset_id)
            if not self.force and thumb_path.exists():
                return False, "Already exists", filename
            
            # Get full image path
            full_image_path = self.storage.images_dir / file_path
            if not full_image_path.exists():
                return False, f"Image file not found: {file_path}", filename
            
            # Generate thumbnail
            with PILImage.open(full_image_path) as img:
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')
                
                original_width, original_height = img.size
                
                # Calculate scaled dimensions maintaining aspect ratio
                thumb_width, thumb_height = self._calculate_scaled_size(
                    original_width, original_height
                )
                
                # Resize with high-quality downsampling
                img = img.resize(
                    (thumb_width, thumb_height),
                    PILImage.Resampling.LANCZOS
                )
                
                # Save to storage
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save with high quality - 100% quality is actually faster to encode
                img.save(
                    thumb_path,
                    'JPEG',
                    quality=settings.thumbnail_quality,
                    optimize=False,  # Disable optimization for speed
                    progressive=False  # Disable progressive for faster encoding
                )
            
            return True, f"{thumb_width}x{thumb_height}", filename
            
        except Exception as e:
            return False, str(e), filename
    
    def _calculate_scaled_size(
        self,
        original_width: int,
        original_height: int
    ) -> Tuple[int, int]:
        """
        Calculate thumbnail dimensions maintaining aspect ratio.
        """
        max_dimension = settings.thumbnail_max_dimension
        
        # If image is already smaller than max dimension, keep original size
        if original_width <= max_dimension and original_height <= max_dimension:
            return (original_width, original_height)
        
        # Calculate scale factor based on longest dimension
        if original_width > original_height:
            scale_factor = max_dimension / original_width
        else:
            scale_factor = max_dimension / original_height
        
        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        return (new_width, new_height)
    
    async def create_thumbnail_records(self, generated_thumbnails: list):
        """Create thumbnail records in database for successfully generated thumbnails."""
        if not generated_thumbnails:
            return
        
        console.print(f"\n[yellow]Creating {len(generated_thumbnails)} thumbnail records in database...[/yellow]")
        
        async with get_db_context() as db:
            for image_id, thumb_width, thumb_height in generated_thumbnails:
                try:
                    # Create thumbnail record
                    thumb_path = self.storage.get_thumbnail_path(image_id, photoset_id)
                    relative_path = str(thumb_path.relative_to(self.storage.thumbnails_dir))
                    
                    thumbnail = Thumbnail(
                        image_id=image_id,
                        size_name="preview",  # Single size for browsing
                        width=thumb_width,
                        height=thumb_height,
                        file_path=relative_path
                    )
                    
                    db.add(thumbnail)
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not create DB record for {image_id}: {e}[/yellow]")
            
            await db.commit()
            console.print("[green]✓ Database records created[/green]")
    
    async def run(self):
        """Run the fast thumbnail generation process."""
        console.print(Panel.fit(
            "[bold cyan]Fast Thumbnail Generation (No Redis)[/bold cyan]\n"
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
            
            # Track thumbnails that were generated for database records
            generated_thumbnails = []
            
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
                        image_data = [(img.id, img.original_filename, img.file_path, img.photoset_id) for img in images]
                        
                        # Submit all images in this batch to thread pool
                        future_to_data = {
                            executor.submit(self.generate_thumbnail_sync, data): data 
                            for data in image_data
                        }
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_data):
                            success, message, filename = future.result()
                            image_id, _, _ = future_to_data[future]
                            processed += 1
                            
                            if success:
                                self.stats["generated"] += 1
                                # Parse dimensions from message (e.g., "1024x768")
                                try:
                                    width, height = map(int, message.split('x'))
                                    generated_thumbnails.append((image_id, width, height))
                                except ValueError:
                                    pass
                            elif "Already exists" in message:
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
                        
                        # Create thumbnail records for this batch
                        if generated_thumbnails:
                            await self.create_thumbnail_records(generated_thumbnails)
                            generated_thumbnails.clear()
                        
                        offset += self.batch_size
        
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
        table.add_row("Total time", f"{elapsed/60:.1f} minutes")
        table.add_row("Average speed", f"{speed:.1f} images/second")
        
        console.print(table)
        
        # Performance stats
        if self.stats["generated"] > 0:
            avg_per_thumbnail = elapsed / self.stats["generated"]
            console.print(f"\n[green]✓ Average time per thumbnail: {avg_per_thumbnail:.2f} seconds[/green]")
            
            # Estimate remaining time for 145k images
            if self.stats["total"] < 145000:
                remaining = 145000 - self.stats["total"]
                eta_minutes = (remaining * avg_per_thumbnail) / 60
                console.print(f"[yellow]⏱ Estimated time for 145k images: {eta_minutes:.0f} minutes[/yellow]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate thumbnails fast (no Redis)")
    parser.add_argument("--force", action="store_true", help="Regenerate existing thumbnails")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of images per batch")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    generator = FastThumbnailGenerator(
        force=args.force,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
    asyncio.run(generator.run())


if __name__ == "__main__":
    main()
