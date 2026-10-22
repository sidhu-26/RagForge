"""SQLAlchemy models for document management."""

from enum import StrEnum

from sqlalchemy import BigInteger, CheckConstraint, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

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

    def __repr__(self) -> str:
        return (
            f"<Document(id={self.id}, filename='{self.filename}', "
            f"status='{self.status}', size={self.file_size_bytes})>"
        )
