"""
Bootstrap all existing data into v2 system.

This script orchestrates the complete data migration:
1. Extract archives
2. Import metadata
3. Generate thumbnails

Usage:
    python scripts/bootstrap_data.py [--dry-run] [--skip-extraction] [--skip-thumbnails]
"""
from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from typing import Optional
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings

console = Console()


class DataBootstrap:
    """Orchestrate the complete data migration."""
    
    def __init__(
        self,
        dry_run: bool = False,
        skip_extraction: bool = False,
        skip_thumbnails: bool = False,
        archives_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ):
        self.dry_run = dry_run
        self.skip_extraction = skip_extraction
        self.skip_thumbnails = skip_thumbnails
        self.archives_path = archives_path
        self.metadata_path = metadata_path
    
    def run_step(self, name: str, command: list[str]) -> bool:
        """Run a migration step."""
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]STEP: {name}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
        
        try:
            result = subprocess.run(command, check=True)
            console.print(f"\n[green]✓ {name} completed successfully[/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"\n[red]✗ {name} failed with code {e.returncode}[/red]")
            return False
        except KeyboardInterrupt:
            console.print(f"\n[yellow]⊘ {name} interrupted by user[/yellow]")
            return False
    
    def run(self):
        """Run the complete bootstrap process."""
        console.print(Panel.fit(
            "[bold cyan]Data Migration Bootstrap[/bold cyan]\n\n"
            "This will migrate all your existing data into the v2 system:\n"
            "  1. Extract photoset archives\n"
            "  2. Import metadata from JSON files\n"
            "  3. Generate thumbnails\n\n"
            f"Mode: {'DRY RUN (no changes)' if self.dry_run else 'LIVE (will make changes)'}\n"
            f"Storage root: {settings.storage_root}",
            border_style="cyan"
        ))
        
        # Show configuration
        console.print("\n[bold]Configuration:[/bold]")
        console.print(f"  Archives path: {self.archives_path or settings.archives_dir}")
        console.print(f"  Metadata path: {self.metadata_path or 'Not configured'}")
        console.print(f"  Images dir: {settings.images_dir}")
        console.print(f"  Thumbnails dir: {settings.thumbnails_dir}")
        
        # Confirm before proceeding
        if not self.dry_run:
            console.print("\n[yellow]WARNING: This will modify your database and file system![/yellow]")
            if not Confirm.ask("Do you want to proceed?"):
                console.print("[yellow]Migration cancelled.[/yellow]")
                return
        
        steps_completed = []
        steps_failed = []
        
        # Step 1: Extract archives
        if not self.skip_extraction:
            cmd = ["python", "scripts/extract_archives.py"]
            if self.archives_path:
                cmd.extend(["--source", str(self.archives_path)])
            if self.dry_run:
                cmd.append("--dry-run")
            
            if self.run_step("Extract Archives", cmd):
                steps_completed.append("Extract Archives")
            else:
                steps_failed.append("Extract Archives")
                if not Confirm.ask("\nContinue despite extraction failure?"):
                    self.show_summary(steps_completed, steps_failed)
                    return
        else:
            console.print("\n[yellow]⊘ Skipping archive extraction[/yellow]")
        
        # Step 2: Import metadata
        if self.metadata_path or not self.dry_run:
            cmd = ["python", "scripts/import_metadata.py"]
            if self.metadata_path:
                cmd.extend(["--source", str(self.metadata_path)])
            if self.dry_run:
                cmd.append("--dry-run")
            
            if self.run_step("Import Metadata", cmd):
                steps_completed.append("Import Metadata")
            else:
                steps_failed.append("Import Metadata")
                if not Confirm.ask("\nContinue despite import failure?"):
                    self.show_summary(steps_completed, steps_failed)
                    return
        else:
            console.print("\n[yellow]⊘ No metadata path configured, skipping metadata import[/yellow]")
        
        # Step 3: Generate thumbnails
        if not self.skip_thumbnails and not self.dry_run:
            cmd = ["python", "scripts/generate_thumbnails.py"]
            
            if self.run_step("Generate Thumbnails", cmd):
                steps_completed.append("Generate Thumbnails")
            else:
                steps_failed.append("Generate Thumbnails")
        elif self.skip_thumbnails:
            console.print("\n[yellow]⊘ Skipping thumbnail generation[/yellow]")
        elif self.dry_run:
            console.print("\n[yellow]⊘ Dry run mode - skipping thumbnail generation[/yellow]")
        
        # Show final summary
        self.show_summary(steps_completed, steps_failed)
    
    def show_summary(self, completed: list[str], failed: list[str]):
        """Show migration summary."""
        console.print("\n" + "="*60)
        console.print("[bold]Migration Summary[/bold]")
        console.print("="*60 + "\n")
        
        if completed:
            console.print("[bold green]Completed Steps:[/bold green]")
            for step in completed:
                console.print(f"  [green]✓[/green] {step}")
        
        if failed:
            console.print("\n[bold red]Failed Steps:[/bold red]")
            for step in failed:
                console.print(f"  [red]✗[/red] {step}")
        
        if not failed:
            console.print("\n[bold green]✓ Migration completed successfully![/bold green]")
            console.print("\n[cyan]Next steps:[/cyan]")
            console.print("  • Run verification: python scripts/verify_migration.py")
            console.print("  • Start the server: python src/main.py")
            console.print("  • Access API docs: http://localhost:8002/docs")
        else:
            console.print("\n[bold red]✗ Migration completed with errors.[/bold red]")
            console.print("Please review the logs above and retry failed steps.")
        
        if self.dry_run:
            console.print("\n[yellow]This was a dry run. No actual changes were made.[/yellow]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bootstrap all data into v2 system")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without making changes")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip archive extraction")
    parser.add_argument("--skip-thumbnails", action="store_true", help="Skip thumbnail generation")
    parser.add_argument("--archives", type=Path, help="Path to archives directory")
    parser.add_argument("--metadata", type=Path, help="Path to metadata JSON directory")
    
    args = parser.parse_args()
    
    bootstrap = DataBootstrap(
        dry_run=args.dry_run,
        skip_extraction=args.skip_extraction,
        skip_thumbnails=args.skip_thumbnails,
        archives_path=args.archives,
        metadata_path=args.metadata
    )
    bootstrap.run()


if __name__ == "__main__":
    main()





