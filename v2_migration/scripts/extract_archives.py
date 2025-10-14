"""
Extract photoset archives (ZIP/RAR) to organized storage.

Usage:
    python scripts/extract_archives.py [--source PATH] [--dry-run]
"""
from __future__ import annotations

import sys
import zipfile
import rarfile
from pathlib import Path
from typing import Optional
import asyncio

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings

console = Console()


class ArchiveExtractor:
    """Extract photoset archives to storage."""
    
    def __init__(self, source_path: Optional[Path] = None, dry_run: bool = False):
        self.source_path = source_path or settings.archives_dir
        self.target_path = settings.images_dir
        self.dry_run = dry_run
        self.stats = {
            "found": 0,
            "extracted": 0,
            "skipped": 0,
            "errors": 0
        }
    
    def find_archives(self) -> list[Path]:
        """Find all ZIP and RAR files in source directory."""
        archives = []
        
        if not self.source_path.exists():
            console.print(f"[red]Source path does not exist: {self.source_path}[/red]")
            return archives
        
        # Find ZIP files
        archives.extend(self.source_path.glob("**/*.zip"))
        archives.extend(self.source_path.glob("**/*.ZIP"))
        
        # Find RAR files
        archives.extend(self.source_path.glob("**/*.rar"))
        archives.extend(self.source_path.glob("**/*.RAR"))
        
        return sorted(archives)
    
    def get_photoset_name(self, archive_path: Path) -> str:
        """Extract photoset name from archive filename."""
        # Remove extension
        name = archive_path.stem
        
        # Clean up common patterns
        name = name.replace("_", " ").replace("-", " ")
        
        return name.strip()
    
    def extract_archive(self, archive_path: Path, photoset_name: str) -> tuple[bool, int, str]:
        """
        Extract a single archive.
        
        Returns:
            Tuple of (success, file_count, error_message)
        """
        # Create target directory for this photoset
        target_dir = self.target_path / photoset_name
        
        # Check if already extracted
        if target_dir.exists() and any(target_dir.iterdir()):
            return False, 0, "Already extracted"
        
        if self.dry_run:
            return True, 0, "Dry run"
        
        try:
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract based on file type
            if archive_path.suffix.lower() == '.zip':
                file_count = self._extract_zip(archive_path, target_dir)
            elif archive_path.suffix.lower() == '.rar':
                file_count = self._extract_rar(archive_path, target_dir)
            else:
                return False, 0, "Unknown archive type"
            
            return True, file_count, ""
            
        except Exception as e:
            return False, 0, str(e)
    
    def _extract_zip(self, archive_path: Path, target_dir: Path) -> int:
        """Extract ZIP archive."""
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Filter image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
            image_files = [
                name for name in zip_ref.namelist()
                if Path(name).suffix.lower() in image_extensions
                and not name.startswith('__MACOSX')
            ]
            
            # Extract only images
            for file_name in image_files:
                # Get just the filename, ignore directory structure in archive
                base_name = Path(file_name).name
                zip_ref.extract(file_name, target_dir)
                
                # Move to root of target dir if in subdirectory
                extracted_path = target_dir / file_name
                if extracted_path.parent != target_dir:
                    extracted_path.rename(target_dir / base_name)
            
            # Clean up empty directories
            self._cleanup_empty_dirs(target_dir)
            
            return len(image_files)
    
    def _extract_rar(self, archive_path: Path, target_dir: Path) -> int:
        """Extract RAR archive."""
        with rarfile.RarFile(archive_path, 'r') as rar_ref:
            # Filter image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
            image_files = [
                name for name in rar_ref.namelist()
                if Path(name).suffix.lower() in image_extensions
            ]
            
            # Extract only images
            for file_name in image_files:
                base_name = Path(file_name).name
                rar_ref.extract(file_name, target_dir)
                
                # Move to root of target dir if in subdirectory
                extracted_path = target_dir / file_name
                if extracted_path.parent != target_dir:
                    extracted_path.rename(target_dir / base_name)
            
            # Clean up empty directories
            self._cleanup_empty_dirs(target_dir)
            
            return len(image_files)
    
    def _cleanup_empty_dirs(self, directory: Path):
        """Remove empty subdirectories."""
        for subdir in directory.rglob("*"):
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()
    
    def run(self):
        """Run the extraction process."""
        console.print(Panel.fit(
            "[bold cyan]Archive Extraction[/bold cyan]\n"
            f"Source: {self.source_path}\n"
            f"Target: {self.target_path}\n"
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}",
            border_style="cyan"
        ))
        
        # Find archives
        console.print("\n[bold]Finding archives...[/bold]")
        archives = self.find_archives()
        self.stats["found"] = len(archives)
        
        if not archives:
            console.print("[yellow]No archives found![/yellow]")
            return
        
        console.print(f"Found {len(archives)} archive(s)\n")
        
        # Extract archives
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Extracting archives...", total=len(archives))
            
            for archive_path in archives:
                photoset_name = self.get_photoset_name(archive_path)
                progress.update(task, description=f"Extracting: {photoset_name}")
                
                success, file_count, error = self.extract_archive(archive_path, photoset_name)
                
                if success:
                    self.stats["extracted"] += 1
                    if not self.dry_run:
                        console.print(f"[green]✓[/green] {photoset_name}: {file_count} files")
                elif error == "Already extracted":
                    self.stats["skipped"] += 1
                    console.print(f"[yellow]⊘[/yellow] {photoset_name}: Already extracted")
                else:
                    self.stats["errors"] += 1
                    console.print(f"[red]✗[/red] {photoset_name}: {error}")
                
                progress.update(task, advance=1)
        
        # Display summary
        self.display_summary()
    
    def display_summary(self):
        """Display extraction summary."""
        table = Table(title="\n[bold]Extraction Summary[/bold]")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Archives found", str(self.stats["found"]))
        table.add_row("Successfully extracted", str(self.stats["extracted"]))
        table.add_row("Skipped (already exists)", str(self.stats["skipped"]))
        table.add_row("Errors", str(self.stats["errors"]), style="red" if self.stats["errors"] > 0 else "green")
        
        console.print(table)
        
        if self.dry_run:
            console.print("\n[yellow]This was a dry run. No files were extracted.[/yellow]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract photoset archives")
    parser.add_argument("--source", type=Path, help="Source directory containing archives")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without extracting")
    
    args = parser.parse_args()
    
    extractor = ArchiveExtractor(
        source_path=args.source,
        dry_run=args.dry_run
    )
    extractor.run()


if __name__ == "__main__":
    main()

