"""Integration tests for photoset API endpoints."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestPhotosetAPI:
    """Test photoset API endpoints."""
    
    async def test_create_photoset(self, client: AsyncClient, sample_photoset_data):
        """Test POST /api/v2/photosets."""
        response = await client.post(
            "/api/v2/photosets",
            json=sample_photoset_data
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert "id" in data
        assert data["name"] == sample_photoset_data["name"]
        assert data["year"] == sample_photoset_data["year"]
        assert data["image_count"] == 0
    
    async def test_get_photoset(self, client: AsyncClient, sample_photoset_data):
        """Test GET /api/v2/photosets/{id}."""
        # Create photoset
        create_response = await client.post(
            "/api/v2/photosets",
            json=sample_photoset_data
        )
        photoset_id = create_response.json()["id"]
        
        # Get photoset
        response = await client.get(f"/api/v2/photosets/{photoset_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == photoset_id
        assert data["name"] == sample_photoset_data["name"]
    
    async def test_get_nonexistent_photoset(self, client: AsyncClient):
        """Test getting a photoset that doesn't exist."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/api/v2/photosets/{fake_id}")
        
        assert response.status_code == 404





