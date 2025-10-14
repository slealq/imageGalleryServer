"""Interactive setup wizard for configuring the application."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import questionary
import psycopg2
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import set_key

console = Console()


class SetupWizard:
    """Interactive setup wizard to configure the application."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.migration_config: Dict[str, Any] = {
            "migration_configured": False,
            "migration_sources": {},
            "setup_date": datetime.now().isoformat(),
            "wizard_version": "1.0"
        }
        self.project_root = Path.cwd()
        self.env_file = self.project_root / ".env"
        self.config_file = self.project_root / "config.json"
    
    def run(self):
        """Run the interactive setup wizard."""
        console.print(Panel.fit(
            "[bold cyan]Image Gallery v2 - Setup Wizard[/bold cyan]\n"
            "Configure your installation step by step",
            border_style="cyan"
        ))
        
        try:
            # Run configuration steps
            self.configure_storage()
            self.configure_migration_sources()
            self.configure_database()
            self.configure_redis()
            self.configure_captions()
            self.configure_performance()
            self.review_and_confirm()
            self.initialize_system()
            self.show_next_steps()
            
            console.print("\n[bold green]✓ Setup completed successfully![/bold green]")
            return True
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Setup cancelled by user[/yellow]")
            return False
        except Exception as e:
            console.print(f"\n[bold red]✗ Setup failed: {e}[/bold red]")
            import traceback
            traceback.print_exc()
            return False
    
    def configure_storage(self):
        """Configure storage paths."""
        console.print("\n[bold]Step 1: Storage Configuration[/bold]")
        console.print("Specify where all image data will be stored.")
        
        storage_root = questionary.text(
            "Storage root directory:",
            default="./storage",
            validate=lambda x: len(x) > 0 or "Path cannot be empty"
        ).ask()
        
        storage_root_path = Path(storage_root).absolute()
        
        # Check if path is writable
        try:
            storage_root_path.mkdir(parents=True, exist_ok=True)
            test_file = storage_root_path / ".write_test"
            test_file.touch()
            test_file.unlink()
            console.print(f"[green]✓ Storage directory is writable: {storage_root_path}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Cannot write to storage directory: {e}[/red]")
            sys.exit(1)
        
        self.config['storage_root'] = str(storage_root_path)
        self.config['images_dir'] = str(storage_root_path / 'images')
        self.config['thumbnails_dir'] = str(storage_root_path / 'thumbnails')
        self.config['crops_dir'] = str(storage_root_path / 'crops')
        self.config['archives_dir'] = str(storage_root_path / 'archives')
    
    def configure_migration_sources(self):
        """Configure paths for existing data migration."""
        console.print("\n[bold]Step 2: Existing Data Migration (Optional)[/bold]")
        console.print("If you have existing data, specify the paths here.")
        
        has_data = questionary.confirm(
            "Do you have existing data to migrate?",
            default=False
        ).ask()
        
        if not has_data:
            console.print("[dim]Skipping data migration configuration[/dim]")
            return
        
        self.migration_config["migration_configured"] = True
        
        # Raw archives
        has_archives = questionary.confirm(
            "Do you have raw zip/rar photoset archives?",
            default=False
        ).ask()
        
        if has_archives:
            archives_path = questionary.text(
                "Path to raw archives directory:",
                validate=lambda x: Path(x).exists() or "Path does not exist"
            ).ask()
            self.migration_config["migration_sources"]["raw_archives_path"] = archives_path
        
        # Metadata JSON files
        has_metadata = questionary.confirm(
            "Do you have photoset metadata JSON files?",
            default=False
        ).ask()
        
        if has_metadata:
            metadata_path = questionary.text(
                "Path to metadata directory:",
                validate=lambda x: Path(x).exists() or "Path does not exist"
            ).ask()
            self.migration_config["migration_sources"]["metadata_json_path"] = metadata_path
        
        # Existing images
        has_images = questionary.confirm(
            "Do you have existing extracted images?",
            default=False
        ).ask()
        
        if has_images:
            images_path = questionary.text(
                "Path to existing images directory:",
                validate=lambda x: Path(x).exists() or "Path does not exist"
            ).ask()
            self.migration_config["migration_sources"]["existing_images_path"] = images_path
        
        # Captions
        has_captions = questionary.confirm(
            "Do you have existing caption files?",
            default=False
        ).ask()
        
        if has_captions:
            captions_path = questionary.text(
                "Path to captions directory:",
                validate=lambda x: Path(x).exists() or "Path does not exist"
            ).ask()
            self.migration_config["migration_sources"]["existing_captions_path"] = captions_path
        
        # Crops
        has_crops = questionary.confirm(
            "Do you have existing crop files?",
            default=False
        ).ask()
        
        if has_crops:
            crops_path = questionary.text(
                "Path to crops directory:",
                validate=lambda x: Path(x).exists() or "Path does not exist"
            ).ask()
            self.migration_config["migration_sources"]["existing_crops_path"] = crops_path
    
    def configure_database(self):
        """Configure PostgreSQL connection."""
        console.print("\n[bold]Step 3: Database Configuration[/bold]")
        console.print("Configure PostgreSQL database connection.")
        
        db_host = questionary.text("PostgreSQL host:", default="localhost").ask()
        db_port = questionary.text("PostgreSQL port:", default="5432").ask()
        db_name = questionary.text("Database name:", default="gallery_v2").ask()
        db_user = questionary.text("Database user:", default="postgres").ask()
        db_pass = questionary.password("Database password:").ask()
        
        # Test connection
        console.print("\n[dim]Testing database connection...[/dim]")
        try:
            # Connect to default postgres database
            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_pass,
                database="postgres"
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (db_name,)
            )
            exists = cursor.fetchone()
            
            if not exists:
                create_db = questionary.confirm(
                    f"Database '{db_name}' does not exist. Create it?",
                    default=True
                ).ask()
                
                if create_db:
                    cursor.execute(f'CREATE DATABASE "{db_name}"')
                    console.print(f"[green]✓ Created database '{db_name}'[/green]")
                else:
                    console.print("[red]✗ Database is required to continue[/red]")
                    sys.exit(1)
            else:
                console.print(f"[green]✓ Database '{db_name}' exists[/green]")
            
            cursor.close()
            conn.close()
            console.print("[green]✓ Database connection successful![/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Database connection failed: {e}[/red]")
            retry = questionary.confirm("Retry database configuration?", default=True).ask()
            if retry:
                return self.configure_database()
            sys.exit(1)
        
        self.config['database_url'] = (
            f"postgresql+asyncpg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        )
        self.config['database_pool_size'] = 20
        self.config['database_echo'] = False
    
    def configure_redis(self):
        """Configure Redis connection."""
        console.print("\n[bold]Step 4: Redis Configuration[/bold]")
        console.print("Configure Redis for caching.")
        
        redis_host = questionary.text("Redis host:", default="localhost").ask()
        redis_port = questionary.text("Redis port:", default="6379").ask()
        redis_db = questionary.text("Redis database number:", default="0").ask()
        
        # Test connection
        console.print("\n[dim]Testing Redis connection...[/dim]")
        try:
            import redis
            r = redis.Redis(
                host=redis_host,
                port=int(redis_port),
                db=int(redis_db),
                socket_connect_timeout=5
            )
            r.ping()
            console.print("[green]✓ Redis connection successful![/green]")
            r.close()
        except Exception as e:
            console.print(f"[yellow]⚠ Redis connection failed: {e}[/yellow]")
            console.print("[dim]Redis is optional. The application will work without it but with reduced performance.[/dim]")
            continue_anyway = questionary.confirm(
                "Continue without Redis?",
                default=False
            ).ask()
            if not continue_anyway:
                retry = questionary.confirm("Retry Redis configuration?", default=True).ask()
                if retry:
                    return self.configure_redis()
                sys.exit(1)
        
        self.config['redis_url'] = f"redis://{redis_host}:{redis_port}/{redis_db}"
    
    def configure_captions(self):
        """Configure caption generation."""
        console.print("\n[bold]Step 5: Caption Generation Setup[/bold]")
        console.print("Choose how captions will be generated.")
        
        generator_type = questionary.select(
            "Caption generator:",
            choices=[
                "dummy (testing only)",
                "unsloth (local AI model)",
                "none (manual captions only)",
            ]
        ).ask()
        
        if "unsloth" in generator_type:
            self.config['caption_generator'] = "unsloth"
            
            model_path = questionary.text(
                "Path to Unsloth model:",
                validate=lambda x: len(x) == 0 or Path(x).exists() or "Path does not exist"
            ).ask()
            
            if model_path:
                self.config['unsloth_model_path'] = str(Path(model_path).absolute())
            
            load_4bit = questionary.confirm(
                "Load model in 4-bit mode? (saves memory)",
                default=True
            ).ask()
            self.config['unsloth_load_in_4bit'] = load_4bit
        elif "none" in generator_type:
            self.config['caption_generator'] = "none"
        else:
            self.config['caption_generator'] = "dummy"
    
    def configure_performance(self):
        """Configure performance settings."""
        console.print("\n[bold]Step 6: Performance Tuning[/bold]")
        console.print("Configure caching and performance settings.")
        
        cache_size = questionary.text(
            "Image cache size (MB):",
            default="10240",
            validate=lambda x: x.isdigit() and int(x) > 0 or "Must be a positive number"
        ).ask()
        self.config['image_cache_size_mb'] = int(cache_size)
        
        self.config['metadata_cache_ttl_seconds'] = 3600
        
        # Thumbnail sizes
        self.config['thumbnail_small_size'] = 256
        self.config['thumbnail_medium_size'] = 512
        self.config['thumbnail_large_size'] = 1024
        
        # Performance
        self.config['max_workers'] = 4
        self.config['prefetch_pages'] = 2
        
        # Logging
        self.config['log_level'] = "INFO"
        log_to_file = questionary.confirm(
            "Enable file logging?",
            default=False
        ).ask()
        
        if log_to_file:
            log_file = Path(self.config['storage_root']) / 'logs' / 'app.log'
            self.config['log_file'] = str(log_file)
        
        # API settings
        self.config['api_host'] = "0.0.0.0"
        self.config['api_port'] = 8002
        self.config['api_version'] = "v2"
        self.config['cors_origins'] = '["http://localhost:3000"]'
        
        # Future: Embeddings
        self.config['embedding_model'] = "openai/clip-vit-base-patch32"
        self.config['embedding_batch_size'] = 32
    
    def review_and_confirm(self):
        """Review configuration and ask for confirmation."""
        console.print("\n[bold]Step 7: Review Configuration[/bold]")
        
        table = Table(title="Configuration Summary", show_header=True, header_style="bold")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        # Show key configuration items
        table.add_row("Storage Root", self.config['storage_root'])
        table.add_row("Database", self.config['database_url'].split('@')[1] if '@' in self.config['database_url'] else self.config['database_url'])
        table.add_row("Redis", self.config['redis_url'])
        table.add_row("Caption Generator", self.config['caption_generator'])
        table.add_row("Image Cache (MB)", str(self.config['image_cache_size_mb']))
        
        if self.migration_config["migration_configured"]:
            table.add_row("Migration", "Configured ✓", style="yellow")
        
        console.print(table)
        
        confirmed = questionary.confirm(
            "\nProceed with this configuration?",
            default=True
        ).ask()
        
        if not confirmed:
            console.print("[yellow]Configuration cancelled[/yellow]")
            sys.exit(0)
    
    def initialize_system(self):
        """Initialize the system with the configuration."""
        console.print("\n[bold]Step 8: Initializing System[/bold]")
        
        # Create storage directories
        console.print("[dim]Creating storage directories...[/dim]")
        for key in ['storage_root', 'images_dir', 'thumbnails_dir', 'crops_dir', 'archives_dir']:
            path = Path(self.config[key])
            path.mkdir(parents=True, exist_ok=True)
            console.print(f"  ✓ {path}")
        
        # Create logs directory if needed
        if 'log_file' in self.config:
            log_path = Path(self.config['log_file'])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            console.print(f"  ✓ {log_path.parent}")
        
        # Generate .env file
        console.print("\n[dim]Generating .env file...[/dim]")
        self.generate_env_file()
        console.print(f"  ✓ {self.env_file}")
        
        # Save migration configuration
        if self.migration_config["migration_configured"]:
            console.print("\n[dim]Saving migration configuration...[/dim]")
            with open(self.config_file, 'w') as f:
                json.dump(self.migration_config, f, indent=2)
            console.print(f"  ✓ {self.config_file}")
        
        # Run database migrations
        console.print("\n[dim]Running database migrations...[/dim]")
        try:
            os.system("alembic upgrade head")
            console.print("  ✓ Database schema created")
        except Exception as e:
            console.print(f"  [yellow]⚠ Could not run migrations automatically: {e}[/yellow]")
            console.print("  [dim]You can run them manually with: alembic upgrade head[/dim]")
        
        console.print("\n[green]✓ System initialized successfully![/green]")
    
    def generate_env_file(self):
        """Generate .env file from collected configuration."""
        # Write header
        with open(self.env_file, 'w') as f:
            f.write(f"# Generated by setup wizard on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# DO NOT COMMIT THIS FILE\n\n")
        
        # Write all configuration
        for key, value in self.config.items():
            env_key = key.upper()
            set_key(self.env_file, env_key, str(value))
    
    def show_next_steps(self):
        """Display next steps to the user."""
        console.print("\n[bold cyan]Next Steps[/bold cyan]\n")
        
        if self.migration_config["migration_configured"]:
            console.print("1. Run data migration:")
            console.print("   [bold]python scripts/bootstrap_data.py[/bold]\n")
        
        console.print("2. Start the service:")
        console.print("   [bold]run_windows.bat[/bold] (Windows)")
        console.print("   or")
        console.print("   [bold]python -m uvicorn src.main:app --reload[/bold] (any platform)\n")
        
        console.print("3. Access the API documentation:")
        console.print(f"   [bold]http://localhost:{self.config['api_port']}/docs[/bold]\n")


def main():
    """Main entry point."""
    wizard = SetupWizard()
    success = wizard.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


