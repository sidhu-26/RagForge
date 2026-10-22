"""ARQ queue integration service.

Provides a clean interface for enqueuing background document processing jobs,
with health checks and error handling for queue outages.
"""

import logging
import uuid

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from ragforge.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class QueueError(Exception):
    """Raised when enqueuing a job fails or Redis is unavailable."""

    pass


class QueueService:
    """Service for dispatching background tasks to ARQ Redis worker queue."""

    def __init__(
        self, redis_pool: ArqRedis | None = None, settings: Settings | None = None
    ) -> None:
        self._pool = redis_pool
        self._settings = settings or get_settings()

    async def get_pool(self) -> ArqRedis:
        """Obtain or lazily instantiate ARQ Redis connection pool."""
        if self._pool is None:
            redis_url = self._settings.REDIS_URL.get_secret_value()
            redis_settings = RedisSettings.from_dsn(redis_url)
            self._pool = await create_pool(redis_settings)
        return self._pool

    async def enqueue_document_processing(self, document_id: uuid.UUID) -> str | None:
        """Enqueue document processing job by document UUID.

        Args:
            document_id: UUID of the document to parse and chunk.

        Returns:
            str | None: Job ID if enqueued, or None if duplicate.

        Raises:
            QueueError: If Redis is unreachable or enqueueing fails.
        """
        try:
            pool = await self.get_pool()
            # arq job_id can be tied to document_id for idempotency at the queue level
            job = await pool.enqueue_job(
                "process_document_job",
                str(document_id),
                _job_id=f"process_doc_{document_id}",
            )
            job_id = job.job_id if job else None
            logger.info(
                "Enqueued document processing job",
                extra={"document_id": str(document_id), "job_id": job_id},
            )
            return job_id
        except Exception as exc:
            logger.error(
                "Failed to enqueue document processing job to Redis",
                extra={"document_id": str(document_id), "error": str(exc)},
            )
            raise QueueError(f"Failed to enqueue document processing job: {exc}") from exc

    async def is_healthy(self) -> bool:
        """Check if Redis queue connection is alive."""
        try:
            pool = await self.get_pool()
            res = await pool.ping()
            return bool(res)
        except Exception:
            return False


_global_queue_service: QueueService | None = None


def get_queue_service() -> QueueService:
    """FastAPI dependency provider for QueueService."""
    global _global_queue_service
    if _global_queue_service is None:
        _global_queue_service = QueueService()
    return _global_queue_service
