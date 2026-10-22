"""ARQ background worker for document parsing, chunking, and persistence.

Coordinates:
1. Loading Document record via an isolated async SQLAlchemy session.
2. Status transition to 'processing'.
3. Reading file bytes through StorageEngine abstraction.
4. Parsing & extracting text via DocumentExtractor.
5. Chunking text via TextChunker.
6. Persisting chunks and marking status 'processed' atomically in one transaction.
7. Handling failures gracefully: rolling back chunks, setting status 'failed',
   storing machine error_code and sanitized error_message.
8. Idempotency: re-running or retrying replaces any prior chunks and avoids duplicate chunks.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from arq.connections import RedisSettings
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ragforge.core.config import Settings, get_settings
from ragforge.db.models import Document, DocumentChunk, DocumentStatus
from ragforge.services.chunker import ChunkingError, TextChunker
from ragforge.services.extractor import (
    DocumentBoundsExceededError,
    DocumentExtractor,
    EmptyDocumentError,
    ExtractionError,
    MalformedDocumentError,
)
from ragforge.services.storage import LocalFilesystemStorage, StorageEngine, StorageError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_worker_db_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Provide an isolated async session for a single background job."""
    async with session_factory() as session:
        yield session


async def process_document(
    document_id_str: str,
    db: AsyncSession,
    storage: StorageEngine,
    settings: Settings,
) -> None:
    """Core document processing pipeline execution.

    Args:
        document_id_str: String UUID of document.
        db: Scoped database session.
        storage: Storage engine to retrieve document payload.
        settings: Application settings.
    """
    start_time = time.perf_counter()
    try:
        doc_uuid = uuid.UUID(document_id_str)
    except ValueError:
        logger.error("Worker received invalid document UUID", extra={"raw_id": document_id_str})
        return

    # 1. Fetch document from DB
    stmt = select(Document).where(Document.id == doc_uuid)
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()

    if document is None:
        logger.warning(
            "Document not found for processing",
            extra={"document_id": str(doc_uuid)},
        )
        return

    # Check eligibility
    # If already processed, exit cleanly (idempotent no-op)
    # If uploaded, queued, or failed (reprocess), proceed.
    if document.status == DocumentStatus.PROCESSED.value:
        logger.info(
            "Document is already in PROCESSED state; skipping processing",
            extra={"document_id": str(doc_uuid)},
        )
        return

    # 2. Mark document as PROCESSING
    logger.info(
        "Transitioning document status to PROCESSING",
        extra={"document_id": str(doc_uuid), "filename": document.filename},
    )
    document.status = DocumentStatus.PROCESSING.value
    document.error_code = None
    document.error_message = None
    await db.commit()
    await db.refresh(document)

    try:
        # 3. Read file bytes from storage abstraction
        storage_key = document.storage_key
        try:
            file_bytes = await storage.read_bytes(storage_key)
        except StorageError as exc:
            raise ExtractionError(
                f"Failed to read file from storage: {exc}", error_code="STORAGE_READ_ERROR"
            ) from exc

        # 4. Extract text pages
        extractor = DocumentExtractor(
            max_pages=settings.EXTRACTION_MAX_PAGES,
            max_text_size_bytes=settings.EXTRACTION_MAX_TEXT_SIZE_BYTES,
        )
        extracted_pages = extractor.extract(mime_type=document.mime_type, file_bytes=file_bytes)

        # 5. Chunk extracted text
        chunker = TextChunker(
            target_chunk_size=settings.CHUNK_TARGET_SIZE,
            overlap=settings.CHUNK_OVERLAP,
            min_chunk_size=settings.CHUNK_MIN_SIZE,
            max_chunks_per_document=settings.MAX_CHUNKS_PER_DOCUMENT,
        )
        chunks = chunker.chunk_document(extracted_pages)

        # 6. Persist chunks and update document status to PROCESSED atomically
        # Remove any existing chunks first to ensure idempotency on retries/reprocessing
        del_stmt = delete(DocumentChunk).where(DocumentChunk.document_id == doc_uuid)
        await db.execute(del_stmt)

        for chunk_data in chunks:
            chunk_row = DocumentChunk(
                document_id=doc_uuid,
                chunk_index=chunk_data.chunk_index,
                content=chunk_data.content,
                page_number=chunk_data.page_number,
                character_count=chunk_data.character_count,
                token_count_estimate=chunk_data.token_count_estimate,
            )
            db.add(chunk_row)

        document.status = DocumentStatus.PROCESSED.value
        document.error_code = None
        document.error_message = None

        await db.commit()

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Document successfully parsed and chunked",
            extra={
                "document_id": str(doc_uuid),
                "chunks_count": len(chunks),
                "duration_seconds": round(elapsed, 3),
                "status": DocumentStatus.PROCESSED.value,
            },
        )

    except (
        EmptyDocumentError,
        DocumentBoundsExceededError,
        MalformedDocumentError,
        ExtractionError,
        ChunkingError,
    ) as exc:
        # Controlled extraction or chunking error
        await db.rollback()
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "Document processing failed with known extraction/chunking error",
            extra={
                "document_id": str(doc_uuid),
                "error_code": exc.error_code,
                "error_message": exc.message,
                "duration_seconds": round(elapsed, 3),
            },
        )
        # Update document state to FAILED
        document.status = DocumentStatus.FAILED.value
        document.error_code = exc.error_code
        document.error_message = exc.message
        await db.commit()

    except Exception:
        # Unexpected fatal error
        await db.rollback()
        elapsed = time.perf_counter() - start_time
        logger.exception(
            "Unexpected error during document processing",
            extra={
                "document_id": str(doc_uuid),
                "duration_seconds": round(elapsed, 3),
            },
        )
        document.status = DocumentStatus.FAILED.value
        document.error_code = "INTERNAL_PROCESSING_ERROR"
        document.error_message = "An internal processing error occurred."
        await db.commit()


async def process_document_job(ctx: dict[str, Any], document_id: str) -> None:
    """Entry point invoked by ARQ worker when receiving a job."""
    session_factory: async_sessionmaker[AsyncSession] = ctx["session_factory"]
    storage: StorageEngine = ctx["storage"]
    settings: Settings = ctx["settings"]

    async with get_worker_db_session(session_factory) as session:
        await process_document(
            document_id_str=document_id,
            db=session,
            storage=storage,
            settings=settings,
        )


async def startup(ctx: dict[str, Any]) -> None:
    """Worker startup hook initializing async engine, storage, and settings."""
    settings = get_settings()
    engine = create_async_engine(
        settings.DATABASE_URL.get_secret_value(),
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=settings.DB_POOL_PRE_PING,
    )
    session_factory = async_sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    storage = LocalFilesystemStorage(settings.STORAGE_LOCAL_ROOT)

    ctx["engine"] = engine
    ctx["session_factory"] = session_factory
    ctx["storage"] = storage
    ctx["settings"] = settings
    logger.info("ARQ Document Worker initialized successfully.")


async def shutdown(ctx: dict[str, Any]) -> None:
    """Worker shutdown hook cleaning up database engine."""
    engine = ctx.get("engine")
    if engine:
        await engine.dispose()
    logger.info("ARQ Document Worker stopped.")


class WorkerSettings:
    """ARQ Worker configuration class."""

    functions = [process_document_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().REDIS_URL.get_secret_value())
    max_tries = get_settings().WORKER_MAX_TRIES
    job_timeout = get_settings().QUEUE_JOB_TIMEOUT_SECONDS
