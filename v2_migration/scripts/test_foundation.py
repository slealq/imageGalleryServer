"""
Test script to validate the foundation layer.

This script tests:
- Configuration loading
- Database connection
- Redis connection
- Database models (create/read/update/delete)
- Repositories
- Caption generators

Run this after completing the setup wizard.
"""
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def test_configuration():
    """Test configuration loading."""
    console.print("\n[bold]1. Testing Configuration[/bold]")
    
    try:
        from src.core.config import settings
        
        table = Table(title="Configuration Status")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        # Check key settings
        checks = [
            ("Database URL", settings.database_url.split('@')[1] if '@' in settings.database_url else "Not configured", "✓"),
            ("Redis URL", settings.redis_url, "✓"),
            ("Storage Root", str(settings.storage_root), "✓" if settings.storage_root.exists() else "✗"),
            ("Images Dir", str(settings.images_dir), "✓" if settings.images_dir.exists() else "✗"),
            ("Caption Generator", settings.caption_generator, "✓"),
        ]
        
        for name, value, status in checks:
            table.add_row(name, value, status)
        
        console.print(table)
        console.print("[green]✓ Configuration loaded successfully[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Configuration error: {e}[/red]")
        console.print("[yellow]Have you run 'python setup/setup_wizard.py' yet?[/yellow]")
        return False


async def test_database():
    """Test database connection and operations."""
    console.print("\n[bold]2. Testing Database Connection[/bold]")
    
    try:
        from src.core.database import engine, get_db_context
        from sqlalchemy import text
        
        # Test connection
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        console.print("[green]✓ Database connection successful[/green]")
        
        # Test session creation
        async with get_db_context() as db:
            result = await db.execute(text("SELECT current_database()"))
            db_name = result.scalar()
            console.print(f"[green]✓ Connected to database: {db_name}[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Database connection error: {e}[/red]")
        console.print("[yellow]Make sure PostgreSQL is running and database is created[/yellow]")
        return False


async def test_redis():
    """Test Redis connection."""
    console.print("\n[bold]3. Testing Redis Connection[/bold]")
    
    try:
        from src.core.redis import redis_client
        
        # Test connection
        if await redis_client.ping():
            console.print("[green]✓ Redis connection successful[/green]")
            
            # Test basic operations
            await redis_client.set("test_key", "test_value", ex=10)
            value = await redis_client.get("test_key")
            assert value == "test_value"
            
            await redis_client.delete("test_key")
            console.print("[green]✓ Redis operations working[/green]")
            return True
        else:
            console.print("[yellow]⚠ Redis not available (optional)[/yellow]")
            return True
            
    except Exception as e:
        console.print(f"[yellow]⚠ Redis error: {e}[/yellow]")
        console.print("[dim]Redis is optional. Service will work with reduced performance.[/dim]")
        return True


async def test_database_schema():
    """Test if database schema is created."""
    console.print("\n[bold]4. Testing Database Schema[/bold]")
    
    try:
        from src.core.database import engine
        from sqlalchemy import text, inspect
        
        async with engine.connect() as conn:
            # Check if tables exist
            inspector = inspect(engine.sync_engine)
            tables = inspector.get_table_names()
            
            expected_tables = [
                'photosets', 'images', 'captions', 'crops', 
                'tags', 'photoset_tags', 'image_tags', 'thumbnails', 'embeddings'
            ]
            
            table_status = Table(title="Database Tables")
            table_status.add_column("Table", style="cyan")
            table_status.add_column("Status", style="green")
            
            all_exist = True
            for table in expected_tables:
                exists = table in tables
                status = "✓ Exists" if exists else "✗ Missing"
                style = "green" if exists else "red"
                table_status.add_row(table, f"[{style}]{status}[/{style}]")
                if not exists:
                    all_exist = False
            
            console.print(table_status)
            
            if all_exist:
                console.print("[green]✓ All tables exist[/green]")
                return True
            else:
                console.print("[yellow]⚠ Some tables missing. Run: alembic upgrade head[/yellow]")
                return False
                
    except Exception as e:
        console.print(f"[red]✗ Schema check error: {e}[/red]")
        console.print("[yellow]Run: alembic upgrade head[/yellow]")
        return False


async def test_models_and_repositories():
    """Test ORM models and repositories."""
    console.print("\n[bold]5. Testing Models & Repositories[/bold]")
    
    try:
        from src.core.database import get_db_context
        from src.models.database import Photoset, Image, Tag
        from src.repositories import PhotosetRepository, ImageRepository, TagRepository
        
        async with get_db_context() as db:
            # Create repositories
            photoset_repo = PhotosetRepository(db)
            image_repo = ImageRepository(db)
            tag_repo = TagRepository(db)
            
            # Test photoset creation
            test_photoset = Photoset(
                name="Test Photoset",
                year=2024,
                metadata={"test": True}
            )
            created_photoset = await photoset_repo.create(test_photoset)
            console.print(f"[green]✓ Created photoset: {created_photoset.id}[/green]")
            
            # Test image creation
            test_image = Image(
                photoset_id=created_photoset.id,
                original_filename="test.jpg",
                file_path="/test/path/test.jpg",
                width=1920,
                height=1080
            )
            created_image = await image_repo.create(test_image)
            console.print(f"[green]✓ Created image: {created_image.id}[/green]")
            
            # Test tag creation
            test_tag = await tag_repo.get_or_create("test_tag", "custom")
            console.print(f"[green]✓ Created/retrieved tag: {test_tag.id}[/green]")
            
            # Test reading
            retrieved_photoset = await photoset_repo.get_by_id(created_photoset.id)
            assert retrieved_photoset is not None
            console.print("[green]✓ Read operations working[/green]")
            
            # Test update
            await photoset_repo.update(created_photoset.id, {"name": "Updated Photoset"})
            updated = await photoset_repo.get_by_id(created_photoset.id)
            assert updated.name == "Updated Photoset"
            console.print("[green]✓ Update operations working[/green]")
            
            # Test delete (cleanup)
            await photoset_repo.delete(created_photoset.id)
            await tag_repo.delete(test_tag.id)
            console.print("[green]✓ Delete operations working[/green]")
            
        console.print("[green]✓ All CRUD operations successful[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Model/Repository error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False


async def test_caption_generators():
    """Test caption generators."""
    console.print("\n[bold]6. Testing Caption Generators[/bold]")
    
    try:
        from PIL import Image
        from src.caption_generators import get_caption_generator
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Test dummy generator
        dummy_gen = get_caption_generator("dummy")
        caption = await dummy_gen.generate_caption(test_image)
        console.print(f"[green]✓ Dummy generator: {caption}[/green]")
        
        # Test streaming
        console.print("[dim]Testing streaming...[/dim]")
        stream_result = []
        async for chunk in dummy_gen.stream_caption(test_image):
            stream_result.append(chunk)
        console.print(f"[green]✓ Streaming works: {len(stream_result)} chunks[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Caption generator error: {e}[/red]")
        return False


async def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold cyan]Foundation Layer Test Suite[/bold cyan]\n"
        "Testing core infrastructure components",
        border_style="cyan"
    ))
    
    results = []
    
    # Run tests in order
    results.append(("Configuration", await test_configuration()))
    
    if not results[0][1]:
        console.print("\n[red]Cannot proceed without valid configuration[/red]")
        console.print("[yellow]Run: python setup/setup_wizard.py[/yellow]")
        return
    
    results.append(("Database", await test_database()))
    results.append(("Redis", await test_redis()))
    results.append(("Schema", await test_database_schema()))
    
    if results[3][1]:  # Only test if schema exists
        results.append(("Models & Repos", await test_models_and_repositories()))
    
    results.append(("Caption Generators", await test_caption_generators()))
    
    # Summary
    console.print("\n" + "="*50)
    console.print("[bold]Test Summary[/bold]\n")
    
    summary_table = Table()
    summary_table.add_column("Component", style="cyan")
    summary_table.add_column("Status", style="green")
    
    passed = 0
    total = len(results)
    
    for name, success in results:
        status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
        summary_table.add_row(name, status)
        if success:
            passed += 1
    
    console.print(summary_table)
    console.print(f"\n[bold]Results: {passed}/{total} tests passed[/bold]")
    
    if passed == total:
        console.print("\n[bold green]✓ All foundation tests passed![/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  1. Continue with service layer implementation")
        console.print("  2. Or start migrating your existing data")
    else:
        console.print("\n[yellow]⚠ Some tests failed. Check the output above.[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())


