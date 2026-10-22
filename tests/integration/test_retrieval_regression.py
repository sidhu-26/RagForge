"""Regression tests for semantic retrieval provider isolation and Ask abstention.

Covers:
1. Cross-provider embedding isolation (preventing fake vs local embedding corruption).
2. Unprocessed document retrieval and Ask abstention (Check A: zero chunks).
3. Grounded retrieval with relevant query vs low similarity abstention (Check B).
"""

import uuid

import pytest

from ragforge.core.config import Settings
from ragforge.db.models import ChunkEmbedding, Document, DocumentChunk, DocumentStatus
from ragforge.db.session import get_session_factory
from ragforge.schemas.ask import AskRequest
from ragforge.services.ask_service import (
    ABSTENTION_MESSAGE_LOW_SIMILARITY,
    ABSTENTION_MESSAGE_NO_EVIDENCE,
    AskService,
)
from ragforge.services.embedding_provider import FakeEmbeddingProvider
from ragforge.services.generation_provider import FakeGenerationProvider
from ragforge.services.retrieval_service import RetrievalService


class MockLocalProvider(FakeEmbeddingProvider):
    """Mock simulating a local provider with provider_name='local'."""

    @property
    def provider_name(self) -> str:
        return "local"


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(model_name="all-MiniLM-L6-v2", dimension=384)


@pytest.fixture
def local_mock_provider() -> MockLocalProvider:
    return MockLocalProvider(model_name="all-MiniLM-L6-v2", dimension=384)


@pytest.fixture
def regression_settings() -> Settings:
    return Settings(
        ENVIRONMENT="testing",
        EMBEDDING_PROVIDER="fake",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        EMBEDDING_DIMENSION=384,
        GENERATION_PROVIDER="fake",
        GENERATION_MODEL="fake-slm",
        ASK_MIN_SIMILARITY_THRESHOLD=0.50,
        ASK_MAX_QUESTION_CHARS=500,
        RETRIEVAL_DEFAULT_LIMIT=5,
        RETRIEVAL_MAX_LIMIT=10,
    )


@pytest.mark.asyncio
async def test_cross_provider_embedding_isolation(
    fake_provider: FakeEmbeddingProvider,
    local_mock_provider: MockLocalProvider,
    regression_settings: Settings,
) -> None:
    """Regression test: Ensure RetrievalService filters by provider in addition to model_name.

    Prevents vectors from incompatible embedding spaces (e.g. SHA-256 fake vs neural local)
    from being evaluated against each other, even when both share model_name='all-MiniLM-L6-v2'.
    """
    session_factory = get_session_factory()
    doc_id = uuid.uuid4()
    unique_tag = uuid.uuid4().hex[:8]

    text_fake = f"Chunk indexed using fake embedding provider {unique_tag}"
    text_local = f"Chunk indexed using local sentence transformer provider {unique_tag}"

    vec_fake = await fake_provider.embed_query(text_fake)
    vec_local = await local_mock_provider.embed_query(text_local)

    async with session_factory() as session:
        doc = Document(
            id=doc_id,
            filename=f"provider_isolation_{unique_tag}.txt",
            storage_key=f"test/{doc_id}.txt",
            mime_type="text/plain",
            file_size_bytes=100,
            sha256_checksum="f" * 64,
            status=DocumentStatus.PROCESSED.value,
        )
        session.add(doc)
        await session.flush()

        chunk_fake = DocumentChunk(
            document_id=doc_id,
            chunk_index=0,
            content=text_fake,
            page_number=1,
            character_count=len(text_fake),
            token_count_estimate=10,
        )
        chunk_local = DocumentChunk(
            document_id=doc_id,
            chunk_index=1,
            content=text_local,
            page_number=1,
            character_count=len(text_local),
            token_count_estimate=10,
        )
        session.add_all([chunk_fake, chunk_local])
        await session.flush()

        emb_fake = ChunkEmbedding(
            chunk_id=chunk_fake.id,
            provider="fake",
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            embedding=vec_fake,
        )
        emb_local = ChunkEmbedding(
            chunk_id=chunk_local.id,
            provider="local",
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            embedding=vec_local,
        )
        session.add_all([emb_fake, emb_local])
        await session.commit()

    async with session_factory() as session:
        # 1. Search with fake provider
        retrieval_fake = RetrievalService(
            db=session,
            embedding_provider=fake_provider,
            settings=regression_settings,
        )
        resp_fake = await retrieval_fake.search(query=text_fake, limit=10, document_id=doc_id)
        assert resp_fake.total_results == 1
        assert resp_fake.items[0].chunk_id == chunk_fake.id
        assert resp_fake.items[0].content == text_fake

        # 2. Search with local provider
        retrieval_local = RetrievalService(
            db=session,
            embedding_provider=local_mock_provider,
            settings=regression_settings,
        )
        resp_local = await retrieval_local.search(query=text_local, limit=10, document_id=doc_id)
        assert resp_local.total_results == 1
        assert resp_local.items[0].chunk_id == chunk_local.id
        assert resp_local.items[0].content == text_local


@pytest.mark.asyncio
async def test_unprocessed_document_retrieval_and_ask_abstention(
    fake_provider: FakeEmbeddingProvider,
    regression_settings: Settings,
) -> None:
    """Regression test: An uploaded unprocessed document yields 0 chunks and Ask abstains."""
    session_factory = get_session_factory()
    doc_id = uuid.uuid4()
    unique_tag = uuid.uuid4().hex[:8]

    async with session_factory() as session:
        # Document stranded in 'uploaded' status (worker didn't process it)
        doc = Document(
            id=doc_id,
            filename=f"unprocessed_{unique_tag}.txt",
            storage_key=f"test/{doc_id}.txt",
            mime_type="text/plain",
            file_size_bytes=100,
            sha256_checksum="u" * 64,
            status=DocumentStatus.UPLOADED.value,
        )
        session.add(doc)
        await session.commit()

    async with session_factory() as session:
        retrieval = RetrievalService(
            db=session,
            embedding_provider=fake_provider,
            settings=regression_settings,
        )
        search_resp = await retrieval.search(
            query="Any question about unprocessed doc",
            document_id=doc_id,
        )
        assert search_resp.total_results == 0
        assert len(search_resp.items) == 0

        # AskService should cleanly abstain with ABSTENTION_MESSAGE_NO_EVIDENCE
        fake_gen = FakeGenerationProvider(model_name="fake-slm")
        ask_service = AskService(
            retrieval_service=retrieval,
            generation_provider=fake_gen,
            settings=regression_settings,
        )
        ask_resp = await ask_service.answer_question(
            AskRequest(
                question="What is inside this document?",
                document_id=doc_id,
            )
        )
        assert ask_resp.abstained is True
        assert ask_resp.answer == ABSTENTION_MESSAGE_NO_EVIDENCE
        assert ask_resp.telemetry.retrieved_chunk_count == 0
        assert ask_resp.telemetry.generation_latency_ms == 0.0


@pytest.mark.asyncio
async def test_relevant_vs_irrelevant_query_behavior(
    fake_provider: FakeEmbeddingProvider,
    regression_settings: Settings,
) -> None:
    """Regression test: Grounded answer for relevant query, abstention for irrelevant."""
    session_factory = get_session_factory()
    doc_id = uuid.uuid4()
    unique_tag = uuid.uuid4().hex[:8]
    doc_content = (
        f"PostgreSQL pgvector extension provides fast vector similarity search {unique_tag}."
    )

    vec = await fake_provider.embed_query(doc_content)

    async with session_factory() as session:
        doc = Document(
            id=doc_id,
            filename=f"pgvector_doc_{unique_tag}.txt",
            storage_key=f"test/{doc_id}.txt",
            mime_type="text/plain",
            file_size_bytes=len(doc_content.encode("utf-8")),
            sha256_checksum="p" * 64,
            status=DocumentStatus.PROCESSED.value,
        )
        session.add(doc)
        await session.flush()

        chunk = DocumentChunk(
            document_id=doc_id,
            chunk_index=0,
            content=doc_content,
            page_number=1,
            character_count=len(doc_content),
            token_count_estimate=12,
        )
        session.add(chunk)
        await session.flush()

        emb = ChunkEmbedding(
            chunk_id=chunk.id,
            provider="fake",
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            embedding=vec,
        )
        session.add(emb)
        await session.commit()

    async with session_factory() as session:
        retrieval = RetrievalService(
            db=session,
            embedding_provider=fake_provider,
            settings=regression_settings,
        )
        fake_gen = FakeGenerationProvider(model_name="fake-slm")
        ask_service = AskService(
            retrieval_service=retrieval,
            generation_provider=fake_gen,
            settings=regression_settings,
        )

        # 1. Relevant query -> success
        rel_resp = await ask_service.answer_question(
            AskRequest(
                question=doc_content,
                document_id=doc_id,
            )
        )
        assert rel_resp.abstained is False
        assert rel_resp.telemetry.retrieved_chunk_count >= 1
        assert len(rel_resp.citations) >= 1

        # 2. Irrelevant query -> low similarity abstention (Check B)
        # We query with completely orthogonal text
        irrel_resp = await ask_service.answer_question(
            AskRequest(
                question="How do you bake sourdough bread with whole wheat flour in a dutch oven?",
                document_id=doc_id,
            )
        )
        assert irrel_resp.abstained is True
        assert irrel_resp.answer == ABSTENTION_MESSAGE_LOW_SIMILARITY
        assert irrel_resp.telemetry.generation_latency_ms == 0.0
