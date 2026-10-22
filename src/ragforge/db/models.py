import uuid
from enum import StrEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ragforge.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DocumentStatus(StrEnum):
    """Processing lifecycle status of a document."""

    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Represents an uploaded document and its storage/processing metadata.

    File bytes are stored in the abstracted storage engine (e.g., local filesystem,
    Azure Blob Storage) under the immutable `storage_key`.
    """

    __tablename__ = "documents"

    # Original client filename (sanitized for display)
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Sanitized original file name for display and reference",
    )

    # Server-generated unique storage key (e.g. docs/yyyy/mm/dd/<uuid>.<ext>)
    storage_key: Mapped[str] = mapped_column(
        String(512),
        unique=True,
        nullable=False,
        index=True,
        doc="Internal server-generated unique storage key or path",
    )

    # Validated media MIME type
    mime_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="Validated MIME media type (e.g. application/pdf)",
    )

    # File size in bytes
    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        doc="File size in bytes",
    )

    # SHA-256 hex digest for data integrity
    sha256_checksum: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Hexadecimal SHA-256 digest of the stored file payload",
    )

    # Current processing status in the lifecycle
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=DocumentStatus.UPLOADED.value,
        server_default=DocumentStatus.UPLOADED.value,
        index=True,
        doc="Lifecycle status: uploaded, queued, processing, processed, failed",
    )

    def __init__(self, **kw: object) -> None:
        if "status" not in kw:
            kw["status"] = DocumentStatus.UPLOADED.value
        super().__init__(**kw)

    # Optional processing error code
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        default=None,
        doc="Sanitized machine-readable failure reason code if status is failed",
    )

    # Optional sanitized error message
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        doc="Sanitized user-facing failure description if status is failed",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('uploaded', 'queued', 'processing', 'processed', 'failed')",
            name="ck_documents_valid_status",
        ),
        CheckConstraint(
            "file_size_bytes > 0",
            name="ck_documents_positive_file_size",
        ),
        Index("ix_documents_status_created_at", "status", "created_at"),
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
        doc="Ordered text chunks associated with this document",
    )

    def __repr__(self) -> str:
        return (
            f"<Document(id={self.id}, filename='{self.filename}', "
            f"status='{self.status}', size={self.file_size_bytes})>"
        )


class DocumentChunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Represents a discrete semantic text chunk extracted from a Document.

    Chunks are strictly ordered by zero-based `chunk_index` within their parent document.
    """

    __tablename__ = "document_chunks"

    # Foreign key to parent document
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key to the parent document record",
    )

    # Zero-based sequential ordering index
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Zero-based ordering index within the document",
    )

    # Non-empty textual content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Extracted chunk text content",
    )

    # 1-based page number (nullable for documents like DOCX and TXT)
    page_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="1-based source page number (primarily for PDFs)",
    )

    # Non-negative character count
    character_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Total character count in this chunk",
    )

    # Approximate token count estimate (character-derived heuristic)
    token_count_estimate: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Estimated token count (heuristic: ~4 chars per token)",
    )

    # Relationship back to document
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )

    # Relationship to embeddings
    embeddings: Mapped[list["ChunkEmbedding"]] = relationship(
        "ChunkEmbedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
        doc="Dense vector embeddings computed for this chunk",
    )

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_doc_chunk_idx"),
        CheckConstraint("chunk_index >= 0", name="ck_document_chunks_non_negative_idx"),
        CheckConstraint("character_count > 0", name="ck_document_chunks_positive_char_count"),
        CheckConstraint("token_count_estimate >= 0", name="ck_document_chunks_non_negative_tokens"),
        CheckConstraint(
            "page_number IS NULL OR page_number > 0", name="ck_document_chunks_positive_page"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id}, document_id={self.document_id}, "
            f"index={self.chunk_index}, chars={self.character_count})>"
        )


class ChunkEmbedding(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Dense vector representation of a DocumentChunk generated by a specific embedding model.

    Enforces uniqueness on (chunk_id, model_name) to guarantee idempotent vector persistence.
    """

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key referencing the parent DocumentChunk",
    )

    provider: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Embedding provider identifier (e.g. 'local', 'azure', 'fake')",
    )

    model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        doc="Model identifier used to produce this embedding (e.g. 'all-MiniLM-L6-v2')",
    )

    dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Length of the embedding vector",
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(384),
        nullable=False,
        doc="Dense vector embedding stored with pgvector",
    )

    chunk: Mapped["DocumentChunk"] = relationship(
        "DocumentChunk",
        back_populates="embeddings",
    )

    __table_args__ = (
        UniqueConstraint("chunk_id", "model_name", name="uq_chunk_embeddings_chunk_model"),
        CheckConstraint("dimension > 0", name="ck_chunk_embeddings_positive_dim"),
    )

    def __repr__(self) -> str:
        return (
            f"<ChunkEmbedding(id={self.id}, chunk_id={self.chunk_id}, "
            f"model='{self.model_name}', dim={self.dimension})>"
        )
