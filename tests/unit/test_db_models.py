"""Unit tests for Document SQLAlchemy model and constraints."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from ragforge.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from ragforge.db.models import Document, DocumentStatus
from ragforge.db.session import get_session_factory


class SampleModel(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Sample concrete model for testing declarative mixins."""

    __tablename__ = "test_sample_models"


def test_base_metadata_discovery() -> None:
    """Verify that models inheriting from Base register their tables in metadata."""
    assert "documents" in Base.metadata.tables
    table = Base.metadata.tables["documents"]

    # Verify columns
    assert "id" in table.columns
    assert table.columns["id"].primary_key is True
    assert "filename" in table.columns
    assert "storage_key" in table.columns
    assert "mime_type" in table.columns
    assert "file_size_bytes" in table.columns
    assert "sha256_checksum" in table.columns
    assert "status" in table.columns
    assert "error_code" in table.columns
    assert "error_message" in table.columns
    assert "created_at" in table.columns
    assert "updated_at" in table.columns

    # Verify document_chunks table and columns
    assert "document_chunks" in Base.metadata.tables
    chunk_table = Base.metadata.tables["document_chunks"]
    assert "id" in chunk_table.columns
    assert chunk_table.columns["id"].primary_key is True
    assert "document_id" in chunk_table.columns
    assert "chunk_index" in chunk_table.columns
    assert "content" in chunk_table.columns
    assert "page_number" in chunk_table.columns
    assert "character_count" in chunk_table.columns
    assert "token_count_estimate" in chunk_table.columns


def test_document_model_instantiation() -> None:
    """Verify instantiating a Document model instance sets default status and attributes."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        filename="test.pdf",
        storage_key="docs/2026/10/22/test.pdf",
        mime_type="application/pdf",
        file_size_bytes=1024,
        sha256_checksum="a" * 64,
    )
    assert doc.id == doc_id
    assert doc.filename == "test.pdf"
    assert doc.status == DocumentStatus.UPLOADED.value
    assert doc.file_size_bytes == 1024
    assert doc.sha256_checksum == "a" * 64
    assert "Document" in repr(doc)


@pytest.mark.asyncio
async def test_document_database_constraints() -> None:
    """Verify database check constraints on real PostgreSQL instance if reachable."""
    factory = get_session_factory()
    try:
        async with factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable for live constraint check")

    async with factory() as session:
        # 1. Negative or zero file_size_bytes should fail constraint
        bad_doc = Document(
            id=uuid.uuid4(),
            filename="invalid_size.pdf",
            storage_key=f"test/{uuid.uuid4()}.pdf",
            mime_type="application/pdf",
            file_size_bytes=0,  # Fails ck_documents_positive_file_size
            sha256_checksum="b" * 64,
            status="uploaded",
        )
        session.add(bad_doc)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 2. Invalid status should fail constraint
        bad_status_doc = Document(
            id=uuid.uuid4(),
            filename="invalid_status.pdf",
            storage_key=f"test/{uuid.uuid4()}.pdf",
            mime_type="application/pdf",
            file_size_bytes=100,
            sha256_checksum="c" * 64,
            status="unsupported_status",  # Fails ck_documents_valid_status
        )
        session.add(bad_status_doc)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()
