"""Semantic search and retrieval API endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ragforge.core.config import Settings, get_settings
from ragforge.db.session import get_db
from ragforge.schemas.search import SearchRequest, SearchResponse
from ragforge.services.embedding_provider import (
    EmbeddingProvider,
    create_embedding_provider,
)
from ragforge.services.retrieval_service import RetrievalError, RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])

_global_embedding_provider: EmbeddingProvider | None = None


def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingProvider:
    """Dependency provider for the shared EmbeddingProvider."""
    global _global_embedding_provider
    if _global_embedding_provider is None:
        _global_embedding_provider = create_embedding_provider(settings)
    return _global_embedding_provider


def get_retrieval_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RetrievalService:
    """Dependency provider for RetrievalService."""
    return RetrievalService(db=db, embedding_provider=provider, settings=settings)


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic vector search",
    description=(
        "Embed the search query and perform nearest-neighbor cosine similarity search "
        "against indexed document chunks using pgvector. Returns ranked source chunks with "
        "similarity metadata and source attribution."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Ranked source chunks retrieved successfully.",
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Empty query or invalid search parameter.",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "Failed to connect to upstream embedding provider.",
        },
    },
)
async def semantic_search(
    request: SearchRequest,
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> SearchResponse:
    """Execute semantic retrieval for a user query."""
    try:
        return await retrieval_service.search(
            query=request.query,
            limit=request.limit,
            document_id=request.document_id,
        )
    except RetrievalError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during semantic search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during semantic search.",
        ) from exc
