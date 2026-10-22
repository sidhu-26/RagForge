"""Unit tests for ARQ background worker and chunk persistence pipeline."""

import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragforge.core.config import Settings
from ragforge.db.models import Document, DocumentChunk, DocumentStatus
from ragforge.db.session import get_session_factory
from ragforge.services.storage import LocalFilesystemStorage
from ragforge.worker import process_document


@pytest.fixture
def worker_storage(tmp_path: Path) -> LocalFilesystemStorage:
    """Fixture providing isolated storage for worker tests."""
    s_dir = tmp_path / "worker_storage"
    s_dir.mkdir(parents=True, exist_ok=True)
    return LocalFilesystemStorage(s_dir)


@pytest.fixture
def worker_settings(tmp_path: Path) -> Settings:
    """Fixture providing settings tuned for worker tests."""
    return Settings(
        ENVIRONMENT="testing",
        STORAGE_LOCAL_ROOT=str(tmp_path / "worker_storage"),
        CHUNK_TARGET_SIZE=100,
        CHUNK_OVERLAP=20,
        CHUNK_MIN_SIZE=10,
        MAX_CHUNKS_PER_DOCUMENT=100,
    )


async def get_test_db() -> AsyncSession:
    """Helper to acquire real DB session."""
    factory = get_session_factory()
    return factory()


@pytest.mark.asyncio
async def test_worker_successful_end_to_end(
    worker_storage: LocalFilesystemStorage,
    worker_settings: Settings,
) -> None:
    """Verify document goes through PROCESSING -> PROCESSED and persists chunks in order."""
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            # Check DB reachability
            pass
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable")

    doc_id = uuid.uuid4()
    storage_key = f"test/{doc_id}.txt"
    sample_text = (
        "Paragraph 1 contains significant introductory information for RAGForge indexing.\n\n"
        "Paragraph 2 provides architectural details on asynchronous worker pipelines.\n\n"
        "Paragraph 3 discusses vector retrieval, embeddings, and context synthesis."
    )
    from collections.abc import AsyncIterator

    async def str_gen() -> AsyncIterator[bytes]:
        yield sample_text.encode("utf-8")

    await worker_storage.save(storage_key, str_gen())

    # Create Document record in DB
    async with session_factory() as session:
        doc = Document(
            id=doc_id,
            filename="worker_test.txt",
            storage_key=storage_key,
            mime_type="text/plain",
            file_size_bytes=len(sample_text.encode("utf-8")),
            sha256_checksum="a" * 64,
            status=DocumentStatus.QUEUED.value,
        )
        session.add(doc)
        await session.commit()

    # Run worker processing
    async with session_factory() as session:
        await process_document(
            document_id_str=str(doc_id),
            db=session,
            storage=worker_storage,
            settings=worker_settings,
        )

    # Verify results in DB
    async with session_factory() as session:
        stmt = select(Document).where(Document.id == doc_id)
        res = await session.execute(stmt)
        updated_doc = res.scalar_one()

        assert updated_doc.status == DocumentStatus.PROCESSED.value
        assert updated_doc.error_code is None
        assert updated_doc.error_message is None

        # Check persisted chunks
        c_stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        c_res = await session.execute(c_stmt)
        chunks = list(c_res.scalars().all())

        assert len(chunks) > 0
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.document_id == doc_id
            assert len(c.content) > 0
            assert c.character_count == len(c.content)
            assert c.token_count_estimate > 0


@pytest.mark.asyncio
async def test_worker_idempotency_and_reprocessing(
    worker_storage: LocalFilesystemStorage,
    worker_settings: Settings,
) -> None:
    """Verify duplicate or re-processing replaces previous chunks and does not duplicate them."""
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            pass
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable")

    doc_id = uuid.uuid4()
    storage_key = f"test/{doc_id}.txt"
    sample_text = "Idempotency test payload text for checking duplicate chunk prevention."

    from collections.abc import AsyncIterator

    async def str_gen() -> AsyncIterator[bytes]:
        yield sample_text.encode("utf-8")

    await worker_storage.save(storage_key, str_gen())

    # Create Document record in DB
    async with session_factory() as session:
        doc = Document(
            id=doc_id,
            filename="idempotency_test.txt",
            storage_key=storage_key,
            mime_type="text/plain",
            file_size_bytes=len(sample_text.encode("utf-8")),
            sha256_checksum="b" * 64,
            status=DocumentStatus.QUEUED.value,
        )
        session.add(doc)
        await session.commit()

    # First run
    async with session_factory() as session:
        await process_document(
            document_id_str=str(doc_id),
            db=session,
            storage=worker_storage,
            settings=worker_settings,
        )

    # Count chunks after first run
    async with session_factory() as session:
        c_stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
        c_res = await session.execute(c_stmt)
        count_first = len(list(c_res.scalars().all()))
        assert count_first > 0

    # Simulate reprocessing (e.g. document requeued to queued status)
    async with session_factory() as session:
        stmt = select(Document).where(Document.id == doc_id)
        doc = (await session.execute(stmt)).scalar_one()
        doc.status = DocumentStatus.QUEUED.value
        await session.commit()

    # Second run
    async with session_factory() as session:
        await process_document(
            document_id_str=str(doc_id),
            db=session,
            storage=worker_storage,
            settings=worker_settings,
        )

    # Count chunks after second run: must NOT be doubled
    async with session_factory() as session:
        c_stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
        c_res = await session.execute(c_stmt)
        count_second = len(list(c_res.scalars().all()))
        assert count_second == count_first


@pytest.mark.asyncio
async def test_worker_extraction_failure_sets_failed_status(
    worker_storage: LocalFilesystemStorage,
    worker_settings: Settings,
) -> None:
    """Verify permanent extraction error marks document as failed with sanitized error message."""
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            pass
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable")

    doc_id = uuid.uuid4()
    storage_key = f"test/{doc_id}.txt"
    # Corrupt text: non-UTF-8 bytes
    from collections.abc import AsyncIterator

    async def str_gen() -> AsyncIterator[bytes]:
        yield b"\xff\xfe\x00\x00invalid bytes"

    await worker_storage.save(storage_key, str_gen())

    async with session_factory() as session:
        doc = Document(
            id=doc_id,
            filename="bad_file.txt",
            storage_key=storage_key,
            mime_type="text/plain",
            file_size_bytes=20,
            sha256_checksum="c" * 64,
            status=DocumentStatus.QUEUED.value,
        )
        session.add(doc)
        await session.commit()

    async with session_factory() as session:
        await process_document(
            document_id_str=str(doc_id),
            db=session,
            storage=worker_storage,
            settings=worker_settings,
        )

    async with session_factory() as session:
        stmt = select(Document).where(Document.id == doc_id)
        doc = (await session.execute(stmt)).scalar_one()
        assert doc.status == DocumentStatus.FAILED.value
        assert doc.error_code == "MALFORMED_DOCUMENT"
        assert doc.error_message is not None

        # Verify no chunks were persisted
        c_stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
        c_res = await session.execute(c_stmt)
        assert len(list(c_res.scalars().all())) == 0


@pytest.mark.asyncio
async def test_worker_missing_document_handled_safely(
    worker_storage: LocalFilesystemStorage,
    worker_settings: Settings,
) -> None:
    """Verify missing document UUID does not raise uncaught exceptions."""
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            pass
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable")

    random_id = str(uuid.uuid4())
    async with session_factory() as session:
        # Should cleanly return without crashing
        await process_document(
            document_id_str=random_id,
            db=session,
            storage=worker_storage,
            settings=worker_settings,
        )
