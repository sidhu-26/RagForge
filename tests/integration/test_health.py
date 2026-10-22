"""Integration tests for application endpoints."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from ragforge.core.config import Settings, get_settings
from ragforge.main import app


@pytest.fixture
def override_settings() -> Settings:
    """Provide a settings object configured for testing."""
    return Settings(
        ENVIRONMENT="testing",
        LOG_LEVEL="DEBUG",
    )


@pytest.fixture
async def async_client(override_settings: Settings) -> AsyncGenerator[AsyncClient, None]:
    """Provide an asynchronous HTTP test client bound to the ASGI application."""
    app.dependency_overrides[get_settings] = lambda: override_settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient) -> None:
    """Test GET / returns 200 and matches the RootServiceInfo schema."""
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "RAGForge API"
    assert data["version"] == "0.1.0"
    assert data["environment"] == "testing"
    assert data["docs_url"] == "/docs"
    assert data["api_v1_prefix"] == "/api/v1"


@pytest.mark.asyncio
async def test_health_live_endpoint(async_client: AsyncClient) -> None:
    """Test GET /api/v1/health/live returns alive status and timestamp."""
    response = await async_client.get("/api/v1/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_ready_endpoint(async_client: AsyncClient) -> None:
    """Test GET /api/v1/health/ready accurately reflects current dependency status."""
    response = await async_client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "timestamp" in data
    assert "dependencies" in data
    # Verify database is explicitly marked not_configured (no fake ready check)
    assert data["dependencies"]["database"]["status"] == "not_configured"


@pytest.mark.asyncio
async def test_openapi_json_available(async_client: AsyncClient) -> None:
    """Test OpenAPI JSON specification schema generation."""
    response = await async_client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "RAGForge API"
    assert "/api/v1/health/live" in schema["paths"]
    assert "/api/v1/health/ready" in schema["paths"]
