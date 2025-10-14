"""Test database connection and check for tables."""

import sys
import asyncio
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.database import get_db_context
from sqlalchemy import text

async def test_database():
    """Test database connection and table existence."""
    print("=== Database Connection Test ===\n")
    
    print(f"Database URL: {settings.database_url}")
    
    try:
        print("1. Testing database connection...")
        async with get_db_context() as db:
            # Test basic connection
            result = await db.execute(text("SELECT 1"))
            assert result.scalar() == 1
            print("+ Database connection successful")
            
            # Check if tables exist
            print("\n2. Checking for tables...")
            result = await db.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            
            expected_tables = [
                'photosets', 'images', 'captions', 'crops', 
                'tags', 'photoset_tags', 'image_tags', 'thumbnails'
            ]
            
            print(f"Found {len(tables)} tables:")
            for table in tables:
                print(f"  + {table}")
            
            missing_tables = [t for t in expected_tables if t not in tables]
            if missing_tables:
                print(f"\n[ERROR] Missing tables: {missing_tables}")
                print("Run: alembic upgrade head")
                return False
            
            # Check data counts
            print("\n3. Checking data counts...")
            for table in ['photosets', 'images']:
                if table in tables:
                    result = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    print(f"  {table}: {count} records")
            
            print("\n[SUCCESS] Database is ready!")
            return True
            
    except Exception as e:
        print(f"[ERROR] Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_database())
    if not success:
        print("\nFix database issues before starting the server.")
        sys.exit(1)
