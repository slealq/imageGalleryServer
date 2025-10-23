"""
Quick database diagnostic - check what's actually in the database.

Usage:
    python scripts/check_database.py
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

from src.core.database import get_db_context
from src.repositories import PhotosetRepository, ImageRepository
from sqlalchemy import select, func, text
from src.models.database import Photoset, Image

console = Console()


async def main():
    """Check database contents."""
    console.print(Panel.fit(
        "[bold cyan]Database Diagnostic[/bold cyan]\n"
        "Checking what's actually in the database",
        border_style="cyan"
    ))
    
    async with get_db_context() as db:
        # Count totals
        photoset_count = await db.scalar(select(func.count()).select_from(Photoset))
        image_count = await db.scalar(select(func.count()).select_from(Image))
        
        console.print(f"\n[bold]Total Counts:[/bold]")
        console.print(f"  Photosets: {photoset_count}")
        console.print(f"  Images: {image_count}")
        
        # Sample photosets with their image counts
        console.print(f"\n[bold]Sample Photosets (first 20):[/bold]\n")
        
        result = await db.execute(
            select(Photoset).limit(20)
        )
        photosets = result.scalars().all()
        
        table = Table()
        table.add_column("Name", style="cyan", width=60)
        table.add_column("Images", justify="right", style="yellow")
        table.add_column("ID", style="dim", width=36)
        
        for ps in photosets:
            # Count images for this photoset
            img_count = await db.scalar(
                select(func.count()).select_from(Image).where(Image.photoset_id == ps.id)
            )
            table.add_row(ps.name[:60], str(img_count), str(ps.id))
        
        console.print(table)
        
        # Photosets with 0 images
        console.print(f"\n[bold]Checking photosets with 0 images...[/bold]")
        
        # Get all photoset IDs and their image counts
        result = await db.execute(
            text("""
                SELECT p.id, p.name, COUNT(i.id) as image_count
                FROM photosets p
                LEFT JOIN images i ON p.id = i.photoset_id
                GROUP BY p.id, p.name
                ORDER BY image_count ASC
                LIMIT 10
            """)
        )
        
        table2 = Table(title="Photosets with Fewest Images")
        table2.add_column("Name", style="cyan", width=60)
        table2.add_column("Images", justify="right", style="yellow")
        
        empty_count = 0
        for row in result:
            table2.add_row(row.name[:60], str(row.image_count))
            if row.image_count == 0:
                empty_count += 1
        
        console.print(table2)
        
        # Count photosets with 0 images
        result = await db.execute(
            text("""
                SELECT COUNT(*) as count
                FROM photosets p
                LEFT JOIN images i ON p.id = i.photoset_id
                GROUP BY p.id
                HAVING COUNT(i.id) = 0
            """)
        )
        total_empty = len(result.all())
        
        console.print(f"\n[bold yellow]Total photosets with 0 images: {total_empty}[/bold yellow]")
        
        # Sample actual image records
        if image_count > 0:
            console.print(f"\n[bold]Sample Image Records (first 10):[/bold]\n")
            
            result = await db.execute(
                select(Image).limit(10)
            )
            images = result.scalars().all()
            
            table3 = Table()
            table3.add_column("Filename", style="cyan", width=40)
            table3.add_column("File Path", style="green", width=60)
            table3.add_column("Photoset ID", style="dim", width=36)
            
            for img in images:
                table3.add_row(
                    img.original_filename[:40],
                    img.file_path[:60],
                    str(img.photoset_id)
                )
            
            console.print(table3)


if __name__ == "__main__":
    asyncio.run(main())





