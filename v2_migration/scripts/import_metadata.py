"""
Import photoset metadata from JSON files into database.

Usage:
    python scripts/import_metadata.py [--source PATH] [--dry-run] [--debug] [--scan]
"""
from __future__ import annotations

import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, date

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.database import get_db_context
from src.repositories import PhotosetRepository, ImageRepository, TagRepository
from src.models.database import Photoset, Image

console = Console()


class MetadataImporter:
    """Import photoset metadata from JSON files."""
    
    def __init__(self, source_path: Optional[Path] = None, dry_run: bool = False, debug: bool = False):
        self.source_path = source_path
        self.images_path = settings.images_dir
        self.dry_run = dry_run
        self.debug = debug
        self.stats = {
            "found": 0,
            "imported": 0,
            "skipped": 0,
            "errors": 0,
            "images_created": 0,
            "dirs_not_found": 0
        }
        self.missing_dirs = []
    
    def find_metadata_files(self) -> list[Path]:
        """Find all JSON metadata files."""
        if not self.source_path:
            console.print("[yellow]No metadata source path configured[/yellow]")
            return []
        
        if not self.source_path.exists():
            console.print(f"[red]Source path does not exist: {self.source_path}[/red]")
            return []
        
        return list(self.source_path.glob("**/*.json"))
    
    def parse_metadata(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """Parse JSON metadata file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]Error parsing {json_path.name}: {e}[/red]")
            return None
    
    def find_photoset_directory(self, photoset_name: str, metadata: Dict[str, Any]) -> Optional[Path]:
        """
        Find the photoset directory, trying multiple naming strategies.
        
        Args:
            photoset_name: Name derived from JSON filename
            metadata: Parsed JSON metadata (may contain 'name' field)
            
        Returns:
            Path to photoset directory if found, None otherwise
        """
        # Try multiple potential directory names
        potential_names = [
            photoset_name,  # JSON filename without extension
        ]
        
        # Try the 'name' field from metadata if it exists
        if 'name' in metadata:
            potential_names.append(metadata['name'])
        
        # Try the original filename without extension if present
        if 'original_filename' in metadata:
            original = Path(metadata['original_filename']).stem
            potential_names.append(original)
        
        # Try with cleaned variations
        for name in list(potential_names):
            # Try cleaned version (spaces, underscores)
            cleaned = name.replace('_', ' ').replace('-', ' ').strip()
            if cleaned not in potential_names:
                potential_names.append(cleaned)
            
            # Try with underscores instead of spaces
            underscored = name.replace(' ', '_').replace('-', '_')
            if underscored not in potential_names:
                potential_names.append(underscored)
        
        # Check each potential name
        for name in potential_names:
            photoset_dir = self.images_path / name
            if photoset_dir.exists() and photoset_dir.is_dir():
                if self.debug:
                    console.print(f"[dim]Found directory: {photoset_dir}[/dim]")
                return photoset_dir
        
        # Not found - log details
        if self.debug:
            console.print(f"[yellow]Could not find directory for '{photoset_name}'[/yellow]")
            console.print(f"[dim]  Tried: {', '.join(potential_names)}[/dim]")
            console.print(f"[dim]  Looking in: {self.images_path}[/dim]")
        
        return None
    
    def find_photoset_images(self, photoset_dir: Path) -> list[Path]:
        """Find all images in a photoset directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        images = []
        
        for ext in image_extensions:
            images.extend(photoset_dir.glob(f"*{ext}"))
            images.extend(photoset_dir.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    async def import_photoset(
        self,
        metadata: Dict[str, Any],
        photoset_name: str
    ) -> tuple[bool, str]:
        """
        Import a single photoset with its images.
        
        Returns:
            Tuple of (success, error_message)
        """
        if self.dry_run:
            # Find directory and count images
            photoset_dir = self.find_photoset_directory(photoset_name, metadata)
            if not photoset_dir:
                return False, "Directory not found"
            images = self.find_photoset_images(photoset_dir)
            return True, f"Would import {len(images)} images"
        
        try:
            async with get_db_context() as db:
                photoset_repo = PhotosetRepository(db)
                image_repo = ImageRepository(db)
                tag_repo = TagRepository(db)
                
                # Check if photoset already exists
                existing = await photoset_repo.get_by_name(photoset_name)
                if existing:
                    # If exists but has no images, try to import images
                    image_count = await photoset_repo.get_image_count(existing.id)
                    if image_count == 0:
                        # Find and import images for existing photoset
                        photoset_dir = self.find_photoset_directory(photoset_name, metadata)
                        if not photoset_dir:
                            return False, f"Already exists (no images - dir not found)"
                        
                        images = self.find_photoset_images(photoset_dir)
                        if not images:
                            return False, f"Already exists (no images found in {photoset_dir.name})"
                        
                        # Import images for this photoset
                        imported_count = await self._import_images(db, existing.id, images, photoset_dir)
                        await db.commit()
                        
                        self.stats["images_created"] += imported_count
                        return True, f"Added {imported_count} images to existing photoset"
                    else:
                        return False, f"Already exists ({image_count} images)"
                
                # Parse date if available
                photoset_date = None
                if 'date' in metadata:
                    try:
                        photoset_date = datetime.strptime(metadata['date'], '%Y-%m-%d').date()
                    except:
                        pass
                
                # Create photoset
                photoset = Photoset(
                    name=photoset_name,
                    source_url=metadata.get('url') or metadata.get('source_url'),
                    date=photoset_date,
                    year=metadata.get('year'),
                    original_archive_filename=metadata.get('original_filename'),
                    extra_metadata={
                        k: v for k, v in metadata.items()
                        if k not in ['name', 'url', 'source_url', 'date', 'year', 'original_filename', 'actors', 'tags']
                    }
                )
                
                created_photoset = await photoset_repo.create(photoset)
                await db.flush()
                
                # Add tags (actors, custom tags)
                if 'actors' in metadata and isinstance(metadata['actors'], list):
                    for actor in metadata['actors']:
                        tag = await tag_repo.get_or_create(actor, "actor")
                        await tag_repo.add_to_photoset(created_photoset.id, tag.id)
                
                if 'tags' in metadata and isinstance(metadata['tags'], list):
                    for tag_name in metadata['tags']:
                        tag = await tag_repo.get_or_create(tag_name, "custom")
                        await tag_repo.add_to_photoset(created_photoset.id, tag.id)
                
                # Find and import images
                photoset_dir = self.find_photoset_directory(photoset_name, metadata)
                if not photoset_dir:
                    self.stats["dirs_not_found"] += 1
                    self.missing_dirs.append(photoset_name)
                    await db.commit()
                    return True, "Created photoset (no images dir found)"
                
                images = self.find_photoset_images(photoset_dir)
                if not images:
                    await db.commit()
                    return True, f"Created photoset (no images in {photoset_dir.name})"
                
                # Import images
                image_count = await self._import_images(db, created_photoset.id, images, photoset_dir)
                await db.commit()
                
                self.stats["images_created"] += image_count
                return True, f"{image_count} images"
                
        except Exception as e:
            import traceback
            if self.debug:
                console.print(f"[red]Exception: {traceback.format_exc()}[/red]")
            return False, str(e)
    
    async def _import_images(
        self,
        db,
        photoset_id,
        image_paths: list[Path],
        photoset_dir: Path
    ) -> int:
        """Import a list of images for a photoset."""
        from sqlalchemy import select
        image_repo = ImageRepository(db)
        count = 0
        skipped = 0
        
        for img_path in image_paths:
            try:
                # Create relative path from images_dir
                try:
                    relative_path = str(img_path.relative_to(self.images_path))
                except ValueError:
                    # If not relative to images_path, use photoset_name/filename
                    relative_path = f"{photoset_dir.name}/{img_path.name}"
                
                # Check if image already exists (by file_path which is unique)
                result = await db.execute(
                    select(Image).where(Image.file_path == relative_path)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    skipped += 1
                    if self.debug:
                        console.print(f"[dim]Skipping {img_path.name} (already exists)[/dim]")
                    continue
                
                # Get image info
                from PIL import Image as PILImage
                with PILImage.open(img_path) as pil_img:
                    width, height = pil_img.size
                
                file_size = img_path.stat().st_size
                
                # Create image record
                image = Image(
                    photoset_id=photoset_id,
                    original_filename=img_path.name,
                    file_path=relative_path,
                    width=width,
                    height=height,
                    file_size=file_size,
                    mime_type=f"image/{img_path.suffix[1:].lower()}"
                )
                
                await image_repo.create(image)
                count += 1
                
            except Exception as e:
                if self.debug:
                    console.print(f"[yellow]Warning: Could not import {img_path.name}: {e}[/yellow]")
        
        if self.debug and skipped > 0:
            console.print(f"[dim]Skipped {skipped} existing images[/dim]")
        
        return count
    
    async def scan_directories(self):
        """Scan and report directory matching issues."""
        console.print(Panel.fit(
            "[bold cyan]Directory Scan Mode[/bold cyan]\n"
            "Checking for directory/metadata mismatches",
            border_style="cyan"
        ))
        
        # Get all directories in images_path
        if not self.images_path.exists():
            console.print(f"[red]Images directory doesn't exist: {self.images_path}[/red]")
            return
        
        existing_dirs = {d.name: d for d in self.images_path.iterdir() if d.is_dir()}
        console.print(f"\nFound {len(existing_dirs)} directories in {self.images_path}\n")
        
        # Find metadata files
        metadata_files = self.find_metadata_files()
        console.print(f"Found {len(metadata_files)} metadata files\n")
        
        # Check matches
        matches = []
        mismatches = []
        
        for json_path in metadata_files:
            photoset_name = json_path.stem
            metadata = self.parse_metadata(json_path)
            if not metadata:
                continue
            
            photoset_dir = self.find_photoset_directory(photoset_name, metadata)
            
            if photoset_dir:
                image_count = len(self.find_photoset_images(photoset_dir))
                matches.append((photoset_name, photoset_dir.name, image_count))
            else:
                mismatches.append((photoset_name, json_path))
        
        # Display results
        if matches:
            table = Table(title="[green]Matched Photosets[/green]")
            table.add_column("JSON Name", style="cyan")
            table.add_column("Directory Name", style="green")
            table.add_column("Images", justify="right", style="yellow")
            
            for json_name, dir_name, img_count in matches[:20]:  # Show first 20
                table.add_row(json_name, dir_name, str(img_count))
            
            if len(matches) > 20:
                table.add_row("...", "...", f"(+{len(matches)-20} more)")
            
            console.print(table)
            console.print(f"\n[green]Total matches: {len(matches)}[/green]")
        
        if mismatches:
            table = Table(title="\n[red]Mismatched Photosets (No Directory Found)[/red]")
            table.add_column("JSON Filename", style="cyan")
            table.add_column("Path", style="dim")
            
            for json_name, json_path in mismatches[:20]:  # Show first 20
                table.add_row(json_name, str(json_path.parent))
            
            if len(mismatches) > 20:
                table.add_row("...", f"(+{len(mismatches)-20} more)")
            
            console.print(table)
            console.print(f"\n[red]Total mismatches: {len(mismatches)}[/red]")
            
            # Suggest fixes
            console.print("\n[yellow]Suggestions:[/yellow]")
            console.print("1. Check if directory names match JSON filenames")
            console.print("2. Run extract_archives.py if archives haven't been extracted")
            console.print("3. Manually rename directories to match JSON filenames")
    
    async def run(self):
        """Run the import process."""
        console.print(Panel.fit(
            "[bold cyan]Metadata Import[/bold cyan]\n"
            f"Source: {self.source_path or 'Not configured'}\n"
            f"Images: {self.images_path}\n"
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}{'  (DEBUG)' if self.debug else ''}",
            border_style="cyan"
        ))
        
        # Find metadata files
        console.print("\n[bold]Finding metadata files...[/bold]")
        metadata_files = self.find_metadata_files()
        self.stats["found"] = len(metadata_files)
        
        if not metadata_files:
            console.print("[yellow]No metadata files found![/yellow]")
            return
        
        console.print(f"Found {len(metadata_files)} file(s)\n")
        
        # Import metadata
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Importing metadata...", total=len(metadata_files))
            
            for json_path in metadata_files:
                photoset_name = json_path.stem
                progress.update(task, description=f"Importing: {photoset_name[:40]}...")
                
                metadata = self.parse_metadata(json_path)
                if not metadata:
                    self.stats["errors"] += 1
                    progress.update(task, advance=1)
                    continue
                
                success, message = await self.import_photoset(metadata, photoset_name)
                
                if success:
                    self.stats["imported"] += 1
                    if not self.dry_run and self.debug:
                        console.print(f"[green]✓[/green] {photoset_name}: {message}")
                elif "Already exists" in message:
                    self.stats["skipped"] += 1
                    if self.debug:
                        console.print(f"[yellow]⊘[/yellow] {photoset_name}: {message}")
                else:
                    self.stats["errors"] += 1
                    console.print(f"[red]✗[/red] {photoset_name}: {message}")
                
                progress.update(task, advance=1)
        
        # Display summary
        self.display_summary()
        
        # Show missing directories if any
        if self.missing_dirs:
            console.print(f"\n[yellow]⚠ {len(self.missing_dirs)} photosets created without images (directory not found)[/yellow]")
            if self.debug:
                console.print("[dim]Run with --scan to see directory matching details[/dim]")
    
    def display_summary(self):
        """Display import summary."""
        table = Table(title="\n[bold]Import Summary[/bold]")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Metadata files found", str(self.stats["found"]))
        table.add_row("Photosets imported", str(self.stats["imported"]))
        table.add_row("Images created", str(self.stats["images_created"]))
        table.add_row("Skipped (already exists)", str(self.stats["skipped"]))
        table.add_row("Directories not found", str(self.stats["dirs_not_found"]), style="yellow")
        table.add_row("Errors", str(self.stats["errors"]), style="red" if self.stats["errors"] > 0 else "green")
        
        console.print(table)
        
        if self.dry_run:
            console.print("\n[yellow]This was a dry run. No data was imported.[/yellow]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import photoset metadata")
    parser.add_argument("--source", type=Path, help="Source directory containing JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without importing")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    parser.add_argument("--scan", action="store_true", help="Scan for directory/metadata mismatches")
    
    args = parser.parse_args()
    
    importer = MetadataImporter(
        source_path=args.source,
        dry_run=args.dry_run,
        debug=args.debug
    )
    
    if args.scan:
        asyncio.run(importer.scan_directories())
    else:
        asyncio.run(importer.run())


if __name__ == "__main__":
    main()
