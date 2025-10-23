"""
Create the test database.

Run this before running tests for the first time.
"""
import sys
from pathlib import Path

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings


def create_test_database():
    """Create the test database if it doesn't exist."""
    # Extract connection details from main database URL
    db_url = settings.database_url
    
    # Parse the URL (postgresql+asyncpg://user:pass@host:port/dbname)
    parts = db_url.replace("postgresql+asyncpg://", "").split("@")
    user_pass = parts[0].split(":")
    user = user_pass[0]
    password = user_pass[1] if len(user_pass) > 1 else ""
    
    host_port_db = parts[1].split("/")
    host_port = host_port_db[0].split(":")
    host = host_port[0]
    port = int(host_port[1]) if len(host_port) > 1 else 5432
    
    test_db_name = "gallery_v2_test"
    
    print(f"Creating test database '{test_db_name}'...")
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (test_db_name,)
        )
        exists = cursor.fetchone()
        
        if exists:
            print(f"✓ Test database '{test_db_name}' already exists")
        else:
            # Create database
            cursor.execute(f'CREATE DATABASE "{test_db_name}"')
            print(f"✓ Created test database '{test_db_name}'")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"✗ Error creating test database: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(create_test_database())





