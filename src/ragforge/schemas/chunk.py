"""Chunk schemas for API response serialization."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunkResponse(BaseModel):
    """Schema for a single document chunk."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(
        ...,
        description="Unique UUIDv4 identifier of the chunk",
        examples=["b53e8787-8d05-4c07-b3c9-04a4f8ad2b99"],
    )
    document_id: uuid.UUID = Field(
        ...,
        description="UUID of the parent document",
        examples=["9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"],
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based ordering index within the document",
        examples=[0],
    )
    content: str = Field(
        ...,
        description="Text content of the chunk",
        examples=["This is the extracted content of the first chunk."],
    )
    page_number: int | None = Field(
        default=None,
        description="1-based source page number (primarily for PDFs)",
        examples=[1],
    )
    character_count: int = Field(
        ...,
        gt=0,
        description="Character count of this chunk",
        examples=[52],
    )
    token_count_estimate: int = Field(
        ...,
        ge=0,
        description="Estimated token count (character-based heuristic)",
        examples=[13],
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the chunk was created",
    )


class DocumentChunkListResponse(BaseModel):
    """Paginated collection of chunks for a document."""

    document_id: uuid.UUID = Field(
        ...,
        description="UUID of parent document",
    )
    status: str = Field(
        ...,
        description="Current processing status of parent document",
        examples=["processed"],
    )
    items: list[DocumentChunkResponse] = Field(
        ...,
        description="Page of chunks ordered by chunk_index ascending",
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of chunks in this document",
        examples=[15],
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
        description="Number of chunks per page",
        examples=[20],
    )
    total_pages: int = Field(
        ...,
        ge=0,
        description="Total number of pages available",
        examples=[1],
    )
