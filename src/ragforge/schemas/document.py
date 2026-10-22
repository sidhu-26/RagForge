"""Document schemas for API request validation and response serialization."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ragforge.db.models import DocumentStatus


class DocumentResponse(BaseModel):
    """Document metadata response representation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(
        ...,
        description="Unique UUIDv4 document identifier",
        examples=["9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"],
    )
    filename: str = Field(
        ...,
        description="Sanitized original document filename",
        examples=["quarterly_report.pdf"],
    )
    storage_key: str = Field(
        ...,
        description="Server-controlled internal storage key",
        examples=["docs/2026/10/22/9b1deb4d3b7d4bad9bdd2b0d7b3dcb6d.pdf"],
    )
    mime_type: str = Field(
        ...,
        description="Validated media MIME type",
        examples=["application/pdf"],
    )
    file_size_bytes: int = Field(
        ...,
        description="File size in bytes",
        examples=[1048576],
    )
    sha256_checksum: str = Field(
        ...,
        description="Hexadecimal SHA-256 digest of stored file contents",
        examples=["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
    )
    status: DocumentStatus = Field(
        ...,
        description="Processing status: uploaded, queued, processing, processed, failed",
        examples=[DocumentStatus.UPLOADED],
    )
    error_code: str | None = Field(
        default=None,
        description="Error code if document processing failed",
        examples=[None],
    )
    error_message: str | None = Field(
        default=None,
        description="Sanitized error description if document processing failed",
        examples=[None],
    )
    created_at: datetime = Field(
        ...,
        description="UTC creation timestamp",
    )
    updated_at: datetime = Field(
        ...,
        description="UTC update timestamp",
    )


class DocumentListResponse(BaseModel):
    """Paginated collection of documents."""

    items: list[DocumentResponse] = Field(
        ...,
        description="Page of document records",
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of documents matching query",
        examples=[42],
    )
    page: int = Field(
        ...,
        ge=1,
        description="Current page number (1-indexed)",
        examples=[1],
    )
    page_size: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of items per page",
        examples=[20],
    )
    total_pages: int = Field(
        ...,
        ge=0,
        description="Total number of pages available",
        examples=[3],
    )


class DocumentDeleteResponse(BaseModel):
    """Response returned upon successful document deletion."""

    id: uuid.UUID = Field(
        ...,
        description="UUID of deleted document",
    )
    deleted: bool = Field(
        default=True,
        description="Flag indicating deletion was successful",
    )
    message: str = Field(
        default="Document and associated storage removed successfully",
        description="Informational deletion summary",
    )
