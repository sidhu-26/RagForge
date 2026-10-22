"""API endpoints for SLM-powered grounded answer generation with citations."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ragforge.api.routes.search import get_retrieval_service
from ragforge.core.config import Settings, get_settings
from ragforge.schemas.ask import AskRequest, AskResponse
from ragforge.services.ask_service import AskService, AskServiceError
from ragforge.services.generation_provider import (
    GenerationProvider,
    create_generation_provider,
)
from ragforge.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ask", tags=["Question Answering"])

_global_generation_provider: GenerationProvider | None = None


def get_generation_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> GenerationProvider:
    """Dependency provider for the shared GenerationProvider."""
    global _global_generation_provider
    if _global_generation_provider is None:
        _global_generation_provider = create_generation_provider(settings)
    return _global_generation_provider


def get_ask_service(
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    generation_provider: Annotated[GenerationProvider, Depends(get_generation_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AskService:
    """Dependency provider for AskService."""
    return AskService(
        retrieval_service=retrieval_service,
        generation_provider=generation_provider,
        settings=settings,
    )


@router.post(
    "",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate grounded answer with verifiable citations",
    description=(
        "Accepts a user question, retrieves relevant document chunks via pgvector, "
        "assembles bounded evidence context, prompts the configured language model (SLM), "
        "validates all cited references against the actual retrieved sources, and returns "
        "the grounded answer with source attribution and latency telemetry."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Grounded answer with citations generated successfully.",
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid question, empty input, or bounds exceeded.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal error during retrieval or generation.",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "Upstream embedding or language model provider error.",
        },
    },
)
async def ask_question(
    request: AskRequest,
    ask_service: Annotated[AskService, Depends(get_ask_service)],
) -> AskResponse:
    """Execute grounded question answering pipeline."""
    try:
        return await ask_service.answer_question(request)
    except AskServiceError as err:
        logger.warning(
            "Question answering rejected or failed",
            extra={"status_code": err.status_code, "error": err.message},
        )
        raise HTTPException(status_code=err.status_code, detail=err.message) from err
    except Exception as exc:
        logger.error("Unexpected error in /ask route", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during answer generation.",
        ) from exc
