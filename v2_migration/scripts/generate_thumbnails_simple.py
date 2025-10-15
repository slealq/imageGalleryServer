"""
Simple thumbnail generator - just scale images down to 1200px max.

Usage:
    python scripts/generate_thumbnails_simple.py [--force]
"""
from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from PIL import Image as PILImage

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.database import get_db_context
from src.repositories import ImageRepository
from src.services.storage_service import StorageService

console = Console()


async def main():
    """Generate thumbnails for all images - simple version."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate thumbnails (simple)")
    parser.add_argument("--force", action="store_true", help="Regenerate existing thumbnails")
    args = parser.parse_args()
    
    console.print("[bold cyan]Simple Thumbnail Generator[/bold cyan]")
    console.print(f"Max dimension: {settings.thumbnail_max_dimension}px")
    console.print(f"Quality: {settings.thumbnail_quality}%")
    console.print(f"Mode: {'FORCE REGENERATE' if args.force else 'SKIP EXISTING'}")
    console.print()
    
    storage = StorageService()
    
    async with get_db_context() as db:
        image_repo = ImageRepository(db)
        
        # Get total count first
        console.print("Counting images in database...")
        total = await image_repo.count()
        
        if total == 0:
            console.print("[yellow]No images found![/yellow]")
            return
        
        console.print(f"Found {total:,} images - processing in batches...")
        
        # We'll process all images by getting them in large batches
        batch_size = 10000  # Process 10k images at a time to avoid memory issues
        all_images = []
        
        console.print("Loading images from database...")
        with Progress(
            TextColumn("Loading images..."),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as load_progress:
            load_task = load_progress.add_task("Loading...", total=total)
            
            offset = 0
            while offset < total:
                batch = await image_repo.get_all(skip=offset, limit=batch_size)
                if not batch:
                    break
                all_images.extend(batch)
                offset += len(batch)
                load_progress.update(load_task, completed=len(all_images))
        
        images = all_images
        console.print(f"Loaded {len(images):,} images into memory")
        
        console.print()
        
        generated = 0
        skipped = 0
        errors = 0
        
        # Process images one by one
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed:,}/{task.total:,})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing images...", total=total)
            
            for i, image in enumerate(images):
                # Update progress
                progress.update(task, description=f"Processing: {image.original_filename[:40]}...")
                
                try:
                    # Check if thumbnail exists
                    thumb_path = storage.get_thumbnail_path(image.id, image.photoset_id)
                    
                    if not args.force and thumb_path.exists():
                        skipped += 1
                        progress.advance(task)
                        continue
                    
                    # Get source image path
                    source_path = storage.images_dir / image.file_path
                    
                    if not source_path.exists():
                        console.print(f"[red]Missing: {image.file_path}[/red]")
                        errors += 1
                        progress.advance(task)
                        continue
                    
                    # Generate thumbnail
                    with PILImage.open(source_path) as img:
                        # Convert to RGB if needed
                        if img.mode in ('RGBA', 'P', 'LA'):
                            img = img.convert('RGB')
                        
                        # Calculate new size (maintain aspect ratio)
                        width, height = img.size
                        max_dim = settings.thumbnail_max_dimension
                        
                        if width <= max_dim and height <= max_dim:
                            # Image is already small enough
                            new_width, new_height = width, height
                        else:
                            # Scale down
                            if width > height:
                                new_width = max_dim
                                new_height = int(height * max_dim / width)
                            else:
                                new_height = max_dim
                                new_width = int(width * max_dim / height)
                        
                        # Resize
                        img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                        
                    # Save thumbnail
                    thumb_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(
                        thumb_path,
                        'JPEG',
                        quality=settings.thumbnail_quality,
                        optimize=False
                    )
                    
                    # Verify file was actually created
                    if not thumb_path.exists():
                        console.print(f"[red]ERROR: Thumbnail not saved: {thumb_path}[/red]")
                        errors += 1
                    else:
                        file_size_mb = thumb_path.stat().st_size / (1024 * 1024)
                        if file_size_mb < 0.01:  # Less than 10KB is suspicious
                            console.print(f"[yellow]WARNING: Very small thumbnail ({file_size_mb:.2f}MB): {image.original_filename}[/yellow]")
                        
                        # Show occasional confirmations
                        if generated % 100 == 0:  # Every 100th thumbnail
                            console.print(f"[green]✓ Created {thumb_path.name} ({file_size_mb:.1f}MB)[/green]")
                
                    generated += 1
                    
                except Exception as e:
                    console.print(f"[red]Error processing {image.original_filename}: {e}[/red]")
                    errors += 1
                
                progress.advance(task)
        
        # Summary
        console.print()
        console.print("[bold]Summary:[/bold]")
        console.print(f"  Generated: {generated:,}")
        console.print(f"  Skipped: {skipped:,}")
        console.print(f"  Errors: {errors:,}")
        console.print(f"  Total: {total:,}")
        
        if errors == 0:
            console.print("[green]✓ All done![/green]")
        else:
            console.print(f"[yellow]⚠ Completed with {errors} errors[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
