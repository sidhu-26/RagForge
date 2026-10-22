"""Schemas for semantic retrieval and nearest-neighbor vector search."""

import uuid

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    """Schema for querying the semantic search engine."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Search query text to embed and match against indexed document chunks",
        examples=["What are the architectural requirements for async workers?"],
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of ranked nearest-neighbor chunks to return",
        examples=[5],
    )
    document_id: uuid.UUID | None = Field(
        default=None,
        description="Optional document UUID filter to restrict search to a single document",
        examples=[None],
    )


class SearchResultItem(BaseModel):
    """Ranked document chunk match with similarity distance metadata."""

    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID = Field(
        ...,
        description="UUID of the matched document chunk",
    )
    document_id: uuid.UUID = Field(
        ...,
        description="UUID of the parent document",
    )
    filename: str = Field(
        ...,
        description="Sanitized filename of the parent document",
    )
    chunk_index: int = Field(
        ...,
        description="Zero-based ordering index within the document",
    )
    content: str = Field(
        ...,
        description="Extracted text content of the matched chunk",
    )
    page_number: int | None = Field(
        default=None,
        description="1-based PDF source page number if applicable",
    )
    similarity_score: float = Field(
        ...,
        description="Cosine similarity score (1.0 = identical direction, range [-1, 1])",
    )
    cosine_distance: float = Field(
        ...,
        description="Cosine distance metric (1 - cosine_similarity, range [0, 2])",
    )


class SearchResponse(BaseModel):
    """Response containing ranked semantic search results and query execution metadata."""

    query: str = Field(
        ...,
        description="Normalized query text used for embedding search",
    )
    model_name: str = Field(
        ...,
        description="Embedding model used for matching",
    )
    dimension: int = Field(
        ...,
        description="Vector dimension used for nearest-neighbor calculation",
    )
    total_results: int = Field(
        ...,
        ge=0,
        description="Number of matching results returned",
    )
    items: list[SearchResultItem] = Field(
        ...,
        description="Ranked list of relevant source chunks ordered by similarity descending",
    )
