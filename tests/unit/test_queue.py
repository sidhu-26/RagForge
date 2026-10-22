"""Unit tests for ARQ Redis Queue integration and dual-write safety."""

import uuid
from unittest.mock import AsyncMock

import pytest

from ragforge.core.config import Settings
from ragforge.services.queue import QueueError, QueueService


@pytest.mark.asyncio
async def test_queue_service_enqueue_success() -> None:
    """Verify queue service successfully enqueues job with document ID."""
    mock_pool = AsyncMock()
    mock_job = AsyncMock()
    mock_job.job_id = "test_job_123"
    mock_pool.enqueue_job.return_value = mock_job

    queue = QueueService(redis_pool=mock_pool)
    doc_id = uuid.uuid4()
    job_id = await queue.enqueue_document_processing(doc_id)

    assert job_id == "test_job_123"
    mock_pool.enqueue_job.assert_called_once_with(
        "process_document_job",
        str(doc_id),
        _job_id=f"process_doc_{doc_id}",
    )


@pytest.mark.asyncio
async def test_queue_service_redis_unavailable_raises_queue_error() -> None:
    """Verify Redis connection failure raises QueueError and does not mask root cause."""
    mock_pool = AsyncMock()
    mock_pool.enqueue_job.side_effect = ConnectionError("Could not connect to Redis")

    queue = QueueService(redis_pool=mock_pool)
    doc_id = uuid.uuid4()

    with pytest.raises(QueueError) as exc_info:
        await queue.enqueue_document_processing(doc_id)

    assert "Could not connect to Redis" in str(exc_info.value)


@pytest.mark.asyncio
async def test_queue_health_check() -> None:
    """Verify health check returns True when Redis responds and False on error."""
    mock_pool = AsyncMock()
    mock_pool.ping.return_value = True
    queue = QueueService(redis_pool=mock_pool)
    assert await queue.is_healthy() is True

    mock_pool.ping.side_effect = Exception("Down")
    assert await queue.is_healthy() is False


@pytest.mark.asyncio
async def test_live_redis_enqueue_if_running() -> None:
    """Verify real Redis container enqueueing if Redis is reachable on localhost:6379."""
    settings = Settings(ENVIRONMENT="testing", REDIS_URL="redis://localhost:6379/0")
    queue = QueueService(settings=settings)
    if not await queue.is_healthy():
        pytest.skip("Local Redis container not reachable")

    doc_id = uuid.uuid4()
    job_id = await queue.enqueue_document_processing(doc_id)
    assert job_id is not None
    assert f"process_doc_{doc_id}" in job_id
