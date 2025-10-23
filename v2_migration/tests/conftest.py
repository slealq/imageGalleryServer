"""Pytest configuration and fixtures."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from httpx import AsyncClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import Base
from src.core.config import settings
from src.main import app
from src.core.database import get_db


# Test database URL
TEST_DATABASE_URL = settings.database_url.replace("/gallery_v2", "/gallery_v2_test")


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_db_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool,  # No connection pooling for tests
        echo=False
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session_maker = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client with database override."""
    
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_photoset_data() -> dict:
    """Sample photoset data for tests."""
    return {
        "name": "Test Photoset",
        "year": 2024,
        "source_url": "https://example.com/test",
        "extra_metadata": {"test": True}
    }


@pytest.fixture
def sample_image_data() -> dict:
    """Sample image data for tests."""
    return {
        "original_filename": "test_image.jpg",
        "width": 1920,
        "height": 1080,
        "file_size": 2458624,
        "mime_type": "image/jpeg",
        "extra_metadata": {}
    }


@pytest.fixture
def sample_caption_data() -> dict:
    """Sample caption data for tests."""
    return {
        "caption": "A beautiful test image",
        "generator_type": "test"
    }





