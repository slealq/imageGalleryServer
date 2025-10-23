"""Integration tests for health check endpoints."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestHealthAPI:
    """Test health check endpoints."""
    
    async def test_health_check(self, client: AsyncClient):
        """Test GET /api/v2/health."""
        response = await client.get("/api/v2/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "database" in data
        assert "cache" in data
        assert data["database"] == "healthy"





