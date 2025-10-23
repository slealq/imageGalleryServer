"""
Verify data migration completeness and integrity.

Usage:
    python scripts/verify_migration.py
"""
from __future__ import annotations

import sys
import asyncio
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.database import get_db_context
from src.repositories import PhotosetRepository, ImageRepository, CaptionRepository, CropRepository, TagRepository

console = Console()


class MigrationVerifier:
    """Verify migration completeness."""
    
    def __init__(self):
        self.checks = []
    
    async def verify_database_counts(self):
        """Verify database record counts."""
        console.print("\n[bold]Database Record Counts[/bold]")
        
        async with get_db_context() as db:
            photoset_repo = PhotosetRepository(db)
            image_repo = ImageRepository(db)
            caption_repo = CaptionRepository(db)
            crop_repo = CropRepository(db)
            tag_repo = TagRepository(db)
            
            counts = {
                "Photosets": await photoset_repo.count(),
                "Images": await image_repo.count(),
                "Captions": await caption_repo.count(),
                "Crops": await crop_repo.count(),
                "Tags": len(await tag_repo.get_all())
            }
            
            table = Table()
            table.add_column("Entity", style="cyan")
            table.add_column("Count", justify="right", style="green")
            
            for entity, count in counts.items():
                table.add_row(entity, str(count))
                self.checks.append((f"{entity} in database", count > 0 if entity != "Captions" else True, ""))
            
            console.print(table)
            
            return counts
    
    async def verify_file_existence(self):
        """Verify that image files exist on disk."""
        console.print("\n[bold]File Existence Check[/bold]")
        
        async with get_db_context() as db:
            image_repo = ImageRepository(db)
            
            total_images = await image_repo.count()
            missing_files = []
            
            if total_images == 0:
                console.print("[yellow]No images in database to verify[/yellow]")
                return
            
            # Sample check (first 100 images)
            images = await image_repo.get_all(limit=min(100, total_images))
            
            for image in images:
                full_path = settings.images_dir / image.file_path
                if not full_path.exists():
                    missing_files.append(image.file_path)
            
            if missing_files:
                console.print(f"[red]✗ {len(missing_files)} files missing (sample of {len(images)})[/red]")
                for path in missing_files[:5]:
                    console.print(f"  [red]Missing: {path}[/red]")
                if len(missing_files) > 5:
                    console.print(f"  [red]... and {len(missing_files) - 5} more[/red]")
                self.checks.append(("Image files exist", False, f"{len(missing_files)} missing"))
            else:
                console.print(f"[green]✓ All sampled files exist ({len(images)} checked)[/green]")
                self.checks.append(("Image files exist", True, ""))
    
    async def verify_thumbnails(self):
        """Verify thumbnail generation."""
        console.print("\n[bold]Thumbnail Check[/bold]")
        
        async with get_db_context() as db:
            image_repo = ImageRepository(db)
            
            total_images = await image_repo.count()
            
            if total_images == 0:
                console.print("[yellow]No images in database[/yellow]")
                return
            
            # Check if thumbnail directories exist
            thumb_dirs_exist = settings.thumbnails_dir.exists()
            
            if not thumb_dirs_exist:
                console.print(f"[yellow]⚠ Thumbnail directory doesn't exist: {settings.thumbnails_dir}[/yellow]")
                self.checks.append(("Thumbnail directory exists", False, "Directory missing"))
                return
            
            # Count thumbnail files
            thumb_count = sum(1 for _ in settings.thumbnails_dir.rglob("*.jpg"))
            expected_count = total_images * 3  # 3 sizes per image
            coverage = (thumb_count / expected_count * 100) if expected_count > 0 else 0
            
            console.print(f"Thumbnail files: {thumb_count} / {expected_count} expected ({coverage:.1f}%)")
            
            if coverage >= 90:
                console.print(f"[green]✓ Thumbnail coverage is good ({coverage:.1f}%)[/green]")
                self.checks.append(("Thumbnails generated", True, f"{coverage:.1f}% coverage"))
            elif coverage >= 50:
                console.print(f"[yellow]⚠ Thumbnail coverage is partial ({coverage:.1f}%)[/yellow]")
                self.checks.append(("Thumbnails generated", True, f"{coverage:.1f}% coverage (partial)"))
            else:
                console.print(f"[red]✗ Thumbnail coverage is low ({coverage:.1f}%)[/red]")
                self.checks.append(("Thumbnails generated", False, f"Only {coverage:.1f}% coverage"))
    
    async def verify_relationships(self):
        """Verify database relationships."""
        console.print("\n[bold]Relationship Integrity Check[/bold]")
        
        async with get_db_context() as db:
            photoset_repo = PhotosetRepository(db)
            image_repo = ImageRepository(db)
            
            # Check that all images have valid photoset references
            total_images = await image_repo.count()
            
            if total_images == 0:
                console.print("[yellow]No images to check[/yellow]")
                return
            
            # Sample check
            images = await image_repo.get_all(limit=min(50, total_images))
            orphaned = 0
            
            for image in images:
                if image.photoset_id:
                    photoset = await photoset_repo.get_by_id(image.photoset_id)
                    if not photoset:
                        orphaned += 1
            
            if orphaned > 0:
                console.print(f"[red]✗ Found {orphaned} orphaned images (sample of {len(images)})[/red]")
                self.checks.append(("Image relationships valid", False, f"{orphaned} orphaned"))
            else:
                console.print(f"[green]✓ All image relationships valid ({len(images)} checked)[/green]")
                self.checks.append(("Image relationships valid", True, ""))
    
    async def verify_storage_directories(self):
        """Verify storage directory structure."""
        console.print("\n[bold]Storage Directory Check[/bold]")
        
        dirs_to_check = {
            "Storage root": settings.storage_root,
            "Images": settings.images_dir,
            "Thumbnails": settings.thumbnails_dir,
            "Crops": settings.crops_dir,
            "Archives": settings.archives_dir
        }
        
        table = Table()
        table.add_column("Directory", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Path", style="dim")
        
        all_exist = True
        for name, path in dirs_to_check.items():
            exists = path.exists()
            status = "[green]✓ Exists[/green]" if exists else "[red]✗ Missing[/red]"
            table.add_row(name, status, str(path))
            
            if not exists:
                all_exist = False
        
        console.print(table)
        self.checks.append(("Storage directories exist", all_exist, ""))
    
    async def run(self):
        """Run all verification checks."""
        console.print(Panel.fit(
            "[bold cyan]Migration Verification[/bold cyan]\n"
            "Verifying data migration completeness and integrity",
            border_style="cyan"
        ))
        
        # Run all checks
        await self.verify_database_counts()
        await self.verify_file_existence()
        await self.verify_thumbnails()
        await self.verify_relationships()
        await self.verify_storage_directories()
        
        # Display summary
        self.display_summary()
    
    def display_summary(self):
        """Display verification summary."""
        console.print("\n" + "="*60)
        console.print("[bold]Verification Summary[/bold]")
        console.print("="*60 + "\n")
        
        table = Table()
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Notes", style="dim")
        
        passed = 0
        failed = 0
        
        for check_name, success, notes in self.checks:
            status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
            table.add_row(check_name, status, notes)
            
            if success:
                passed += 1
            else:
                failed += 1
        
        console.print(table)
        
        total = passed + failed
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        console.print(f"\n[bold]Result: {passed}/{total} checks passed ({pass_rate:.1f}%)[/bold]")
        
        if failed == 0:
            console.print("\n[bold green]✓ All verification checks passed![/bold green]")
            console.print("\n[cyan]Your data has been successfully migrated to v2.[/cyan]")
        else:
            console.print(f"\n[bold yellow]⚠ {failed} check(s) failed.[/bold yellow]")
            console.print("Review the details above and re-run migration scripts as needed.")


async def main():
    """Main entry point."""
    verifier = MigrationVerifier()
    await verifier.run()


if __name__ == "__main__":
    asyncio.run(main())





