"""Question answering and grounded RAG answer orchestration service.

Orchestrates the end-to-end question answering pipeline:
1. Validates and normalizes user question.
2. Performs pgvector nearest-neighbor search via RetrievalService.
3. Evaluates evidence quality and applies configurable abstention policy.
4. Assembles bounded evidence context with source tokens.
5. Invokes GenerationProvider to generate grounded answer.
6. Validates and sanitizes citations against actual retrieved source chunks.
7. Emits structured response with source attribution and latency telemetry.
"""

import logging
import time

from ragforge.core.config import Settings
from ragforge.schemas.ask import (
    AskRequest,
    AskResponse,
    CitationSourceItem,
    GenerationTelemetry,
)
from ragforge.services.citation_validator import validate_and_sanitize_citations
from ragforge.services.generation_provider import (
    GenerationError,
    GenerationProvider,
)
from ragforge.services.prompt_builder import build_evidence_and_prompts
from ragforge.services.retrieval_service import RetrievalError, RetrievalService

logger = logging.getLogger(__name__)

ABSTENTION_MESSAGE_NO_EVIDENCE = (
    "The indexed documents did not provide sufficient evidence to answer the question."
)
ABSTENTION_MESSAGE_LOW_SIMILARITY = (
    "The retrieved passages have insufficient relevance to reliably answer the question."
)


class AskServiceError(Exception):
    """Base exception for AskService orchestration errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AskService:
    """Orchestrates semantic retrieval, evidence budgeting, and citation validation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_provider: GenerationProvider,
        settings: Settings,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generation_provider = generation_provider
        self.settings = settings

    async def answer_question(self, request: AskRequest) -> AskResponse:
        """Execute full RAG generation pipeline.

        Args:
            request: Validated user question, limit, and optional document filter.

        Returns:
            AskResponse: Answer text, validated citations, source attribution, and telemetry.
        """
        clean_question = request.question.strip()
        if not clean_question:
            raise AskServiceError("Question string must not be empty.", status_code=400)

        if len(clean_question) > self.settings.ASK_MAX_QUESTION_CHARS:
            max_chars = self.settings.ASK_MAX_QUESTION_CHARS
            raise AskServiceError(
                f"Question length exceeds maximum allowed limit of {max_chars} characters.",
                status_code=400,
            )

        # 1. Retrieve Candidate Chunks via RetrievalService
        t_retrieval_start = time.perf_counter()
        try:
            search_resp = await self.retrieval_service.search(
                query=clean_question,
                limit=request.limit,
                document_id=request.document_id,
            )
        except RetrievalError as ret_err:
            logger.error("Retrieval failed during question answering", exc_info=True)
            raise AskServiceError(
                f"Retrieval error: {ret_err.message}",
                status_code=ret_err.status_code,
            ) from ret_err

        retrieval_latency_ms = (time.perf_counter() - t_retrieval_start) * 1000.0
        retrieved_items = search_resp.items

        # 2. Evidence Quality & Abstention Check
        # Check A: No chunks found
        if not retrieved_items:
            logger.info("Abstaining from answer generation: zero chunks retrieved")
            return AskResponse(
                question=clean_question,
                answer=ABSTENTION_MESSAGE_NO_EVIDENCE,
                abstained=True,
                sources=[],
                citations=[],
                telemetry=GenerationTelemetry(
                    provider=self.generation_provider.provider_name,
                    model_name=self.generation_provider.model_name,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=0.0,
                    retrieved_chunk_count=0,
                    used_chunk_count=0,
                ),
            )

        # Check B: Top chunk similarity below configured threshold
        top_score = retrieved_items[0].similarity_score
        min_threshold = self.settings.ASK_MIN_SIMILARITY_THRESHOLD
        if min_threshold > 0.0 and top_score < min_threshold:
            logger.info(
                "Abstaining from answer generation: top similarity score below threshold",
                extra={"top_score": top_score, "threshold": min_threshold},
            )
            # Map candidate items to source representations for visibility
            candidate_sources = [
                CitationSourceItem(
                    source_id=f"[source_{i}]",
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    filename=item.filename,
                    chunk_index=item.chunk_index,
                    page_number=item.page_number,
                    content=item.content,
                    similarity_score=item.similarity_score,
                )
                for i, item in enumerate(retrieved_items, start=1)
            ]
            return AskResponse(
                question=clean_question,
                answer=ABSTENTION_MESSAGE_LOW_SIMILARITY,
                abstained=True,
                sources=candidate_sources,
                citations=[],
                telemetry=GenerationTelemetry(
                    provider=self.generation_provider.provider_name,
                    model_name=self.generation_provider.model_name,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=0.0,
                    retrieved_chunk_count=len(retrieved_items),
                    used_chunk_count=0,
                ),
            )

        # 3. Assemble Prompt & Evidence Context
        prompt_context = build_evidence_and_prompts(
            question=clean_question,
            retrieved_items=retrieved_items,
            max_evidence_chars=self.settings.ASK_MAX_EVIDENCE_TOTAL_CHARS,
        )

        # 4. Invoke Generation Provider
        try:
            gen_output = await self.generation_provider.generate(
                system_prompt=prompt_context.system_prompt,
                user_prompt=prompt_context.user_prompt,
                temperature=self.settings.GENERATION_TEMPERATURE,
                max_tokens=self.settings.GENERATION_MAX_OUTPUT_TOKENS,
            )
        except GenerationError as gen_err:
            logger.error("Generation provider error during question answering", exc_info=True)
            raise AskServiceError(
                f"Generation error: {gen_err.message}",
                status_code=502,
            ) from gen_err

        # 5. Validate & Sanitize Citations
        citation_result = validate_and_sanitize_citations(
            raw_answer=gen_output.text,
            valid_sources=prompt_context.sources,
        )

        # Check if model self-abstained
        is_self_abstaining = (
            "does not contain sufficient information" in citation_result.cleaned_answer.lower()
            or "not enough information" in citation_result.cleaned_answer.lower()
        )

        logger.info(
            "Answer generation completed",
            extra={
                "provider": self.generation_provider.provider_name,
                "model": self.generation_provider.model_name,
                "retrieved_chunks": len(retrieved_items),
                "used_chunks": prompt_context.used_chunk_count,
                "valid_citations": len(citation_result.validated_citations),
                "dropped_citations": len(citation_result.invalid_source_ids),
                "abstained": is_self_abstaining,
            },
        )

        return AskResponse(
            question=clean_question,
            answer=citation_result.cleaned_answer,
            abstained=is_self_abstaining,
            sources=prompt_context.sources,
            citations=citation_result.validated_citations,
            telemetry=GenerationTelemetry(
                provider=self.generation_provider.provider_name,
                model_name=self.generation_provider.model_name,
                retrieval_latency_ms=round(retrieval_latency_ms, 2),
                generation_latency_ms=round(gen_output.latency_ms, 2),
                retrieved_chunk_count=len(retrieved_items),
                used_chunk_count=prompt_context.used_chunk_count,
                prompt_tokens=gen_output.prompt_tokens,
                completion_tokens=gen_output.completion_tokens,
                total_tokens=gen_output.total_tokens,
            ),
        )
