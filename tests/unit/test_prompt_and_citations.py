"""Unit tests for prompt construction and citation validation."""

import uuid

from ragforge.schemas.ask import CitationSourceItem
from ragforge.schemas.search import SearchResultItem
from ragforge.services.citation_validator import validate_and_sanitize_citations
from ragforge.services.prompt_builder import build_evidence_and_prompts


def test_build_evidence_and_prompts_budget_and_formatting() -> None:
    """Verify evidence builder assigns [source_N] tokens and respects max character budget."""
    doc_id = uuid.uuid4()
    item1 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        filename="arch.txt",
        chunk_index=0,
        content="PostgreSQL async engine uses connection pooling.",
        page_number=1,
        similarity_score=0.9,
        cosine_distance=0.1,
    )
    item2 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        filename="arch.txt",
        chunk_index=1,
        content="ARQ worker consumes Redis streams.",
        page_number=2,
        similarity_score=0.8,
        cosine_distance=0.2,
    )

    ctx = build_evidence_and_prompts(
        question="How does pooling work?",
        retrieved_items=[item1, item2],
        max_evidence_chars=5000,
    )

    assert ctx.used_chunk_count == 2
    assert len(ctx.sources) == 2
    assert ctx.sources[0].source_id == "[source_1]"
    assert ctx.sources[1].source_id == "[source_2]"
    assert "[source_1] (Document: arch.txt (Page 1))" in ctx.user_prompt
    assert "[source_2] (Document: arch.txt (Page 2))" in ctx.user_prompt
    assert "QUESTION: How does pooling work?" in ctx.user_prompt
    assert "SECURITY & UNTRUSTED DATA" in ctx.system_prompt


def test_build_evidence_truncates_over_budget() -> None:
    """Verify evidence builder stops including chunks when character budget is exceeded."""
    doc_id = uuid.uuid4()
    long_content = "A" * 600
    item1 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        filename="doc1.txt",
        chunk_index=0,
        content=long_content,
        page_number=None,
        similarity_score=0.9,
        cosine_distance=0.1,
    )
    item2 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        filename="doc2.txt",
        chunk_index=0,
        content=long_content,
        page_number=None,
        similarity_score=0.8,
        cosine_distance=0.2,
    )

    # Budget allows only 1 chunk (~650 chars with header)
    ctx = build_evidence_and_prompts(
        question="Test?",
        retrieved_items=[item1, item2],
        max_evidence_chars=750,
    )
    assert ctx.used_chunk_count == 1
    assert len(ctx.sources) == 1
    assert ctx.sources[0].filename == "doc1.txt"


def test_validate_and_sanitize_citations_valid_and_hallucinated() -> None:
    """Verify citations matching retrieved sources are kept while invalid ones are stripped."""
    c1_id = uuid.uuid4()
    c2_id = uuid.uuid4()
    sources = [
        CitationSourceItem(
            source_id="[source_1]",
            chunk_id=c1_id,
            document_id=uuid.uuid4(),
            filename="postgres.txt",
            chunk_index=0,
            page_number=3,
            content="Text 1",
            similarity_score=0.85,
        ),
        CitationSourceItem(
            source_id="[source_2]",
            chunk_id=c2_id,
            document_id=uuid.uuid4(),
            filename="redis.txt",
            chunk_index=1,
            page_number=None,
            content="Text 2",
            similarity_score=0.75,
        ),
    ]

    # Model answer contains valid [source_1], valid [source_2], and hallucinated [source_99]
    raw_answer = (
        "PostgreSQL uses connection pools [source_1]. "
        "Redis runs in-memory queues [source_2] and Kafka handles events [source_99]."
    )

    result = validate_and_sanitize_citations(raw_answer=raw_answer, valid_sources=sources)

    assert result.has_valid_citations is True
    assert len(result.validated_citations) == 2
    assert result.validated_citations[0].source_id == "[source_1]"
    assert result.validated_citations[0].chunk_id == c1_id
    assert result.validated_citations[0].filename == "postgres.txt"
    assert result.validated_citations[0].page_number == 3

    assert result.validated_citations[1].source_id == "[source_2]"
    assert result.validated_citations[1].chunk_id == c2_id

    # Hallucinated citation [source_99] must be stripped from cleaned text
    assert "[source_99]" not in result.cleaned_answer
    assert "[source_99]" in result.invalid_source_ids
    assert "[source_1]" in result.cleaned_answer
    assert "[source_2]" in result.cleaned_answer


def test_validate_and_sanitize_citations_no_citations() -> None:
    """Verify answer without citations produces empty list without error."""
    raw_answer = "The provided context does not contain sufficient information to answer."
    result = validate_and_sanitize_citations(raw_answer=raw_answer, valid_sources=[])
    assert result.has_valid_citations is False
    assert len(result.validated_citations) == 0
    assert result.cleaned_answer == raw_answer
