"""Integration tests for application endpoints."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ragforge.core.config import Settings, get_settings
from ragforge.db.session import get_db
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
async def test_health_ready_endpoint_healthy(async_client: AsyncClient) -> None:
    """Test GET /api/v1/health/ready returns 200 when database connectivity check succeeds."""
    mock_db = AsyncMock(spec=AsyncSession)
    mock_db.execute.return_value = None
    app.dependency_overrides[get_db] = lambda: mock_db

    response = await async_client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "timestamp" in data
    assert "dependencies" in data
    assert data["dependencies"]["database"]["status"] == "up"
    assert "healthy" in data["dependencies"]["database"]["details"].lower()
    mock_db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_ready_endpoint_database_down(async_client: AsyncClient) -> None:
    """Test GET /api/v1/health/ready returns 503 when database query fails."""
    mock_db = AsyncMock(spec=AsyncSession)
    mock_db.execute.side_effect = ConnectionRefusedError("Could not connect to PostgreSQL server")
    app.dependency_overrides[get_db] = lambda: mock_db

    response = await async_client.get("/api/v1/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert "timestamp" in data
    assert "dependencies" in data
    assert data["dependencies"]["database"]["status"] == "down"
    # Ensure sensitive connection details, credentials or hostnames are not leaked
    assert "Could not connect" not in data["dependencies"]["database"]["details"]
    assert "password" not in response.text


@pytest.mark.asyncio
async def test_health_liveness_remains_healthy_when_database_down(
    async_client: AsyncClient,
) -> None:
    """Verify GET /api/v1/health/live stays healthy even if database dependency is down."""
    mock_db = AsyncMock(spec=AsyncSession)
    mock_db.execute.side_effect = ConnectionRefusedError("Database unreachable")
    app.dependency_overrides[get_db] = lambda: mock_db

    response = await async_client.get("/api/v1/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


@pytest.mark.asyncio
async def test_openapi_json_available(async_client: AsyncClient) -> None:
    """Test OpenAPI JSON specification schema generation."""
    response = await async_client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "RAGForge API"
    assert "/api/v1/health/live" in schema["paths"]
    assert "/api/v1/health/ready" in schema["paths"]
