"""Semantic retrieval service powered by pgvector nearest-neighbor search.

Takes query strings, generates dense vector representations with the configured
EmbeddingProvider, executes pgvector cosine similarity search in PostgreSQL,
and formats results with stable tie-breaking.
"""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragforge.core.config import Settings
from ragforge.db.models import ChunkEmbedding, Document, DocumentChunk
from ragforge.schemas.search import SearchResponse, SearchResultItem
from ragforge.services.embedding_provider import EmbeddingError, EmbeddingProvider

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when query embedding or vector search fails."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class RetrievalService:
    """Service orchestrating query embedding and nearest-neighbor search with pgvector."""

    def __init__(
        self,
        db: AsyncSession,
        embedding_provider: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self.db = db
        self.provider = embedding_provider
        self.settings = settings

    async def search(
        self,
        query: str,
        limit: int = 5,
        document_id: uuid.UUID | None = None,
    ) -> SearchResponse:
        """Execute semantic nearest-neighbor search.

        Args:
            query: Non-empty search query string.
            limit: Maximum ranked chunks to retrieve (1-50).
            document_id: Optional filter for a specific document.

        Returns:
            SearchResponse: Ranked collection of source chunks with similarity scores.
        """
        clean_query = query.strip()
        if not clean_query:
            raise RetrievalError("Query string must not be empty.", status_code=400)

        # Enforce limits
        max_limit = self.settings.RETRIEVAL_MAX_LIMIT
        limit = max(1, min(limit, max_limit))

        # 1. Generate query embedding
        try:
            query_vector = await self.provider.embed_query(clean_query)
        except EmbeddingError as exc:
            logger.error("Failed to generate query embedding", exc_info=True)
            raise RetrievalError(f"Failed to embed query: {exc.message}", status_code=502) from exc

        # 2. Build pgvector nearest-neighbor query using cosine distance (<=> operator)
        # Cosine distance = 1 - cosine_similarity
        # Lower distance means higher similarity.
        cosine_distance_col = ChunkEmbedding.embedding.cosine_distance(query_vector).label(
            "distance"
        )

        stmt = (
            select(
                DocumentChunk.id.label("chunk_id"),
                DocumentChunk.document_id,
                Document.filename,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                DocumentChunk.page_number,
                cosine_distance_col,
            )
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == DocumentChunk.id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(ChunkEmbedding.model_name == self.provider.model_name)
        )

        if document_id is not None:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        # Order by cosine distance ascending (closest first), with stable tie-breaking on chunk_id
        stmt = stmt.order_by(cosine_distance_col.asc(), DocumentChunk.id.asc()).limit(limit)

        result = await self.db.execute(stmt)
        rows = result.all()

        items: list[SearchResultItem] = []
        for row in rows:
            dist = float(row.distance)
            # Cosine similarity = 1 - distance
            similarity = round(1.0 - dist, 5)
            items.append(
                SearchResultItem(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    filename=row.filename,
                    chunk_index=row.chunk_index,
                    content=row.content,
                    page_number=row.page_number,
                    similarity_score=similarity,
                    cosine_distance=round(dist, 5),
                )
            )

        logger.info(
            "Semantic search executed",
            extra={
                "model": self.provider.model_name,
                "results_count": len(items),
                "filtered_doc_id": str(document_id) if document_id else None,
            },
        )

        return SearchResponse(
            query=clean_query,
            model_name=self.provider.model_name,
            dimension=self.provider.dimension,
            total_results=len(items),
            items=items,
        )
