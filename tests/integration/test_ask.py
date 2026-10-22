"""Integration tests for POST /api/v1/ask grounded question answering."""

import uuid
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from ragforge.api.routes.ask import get_generation_provider
from ragforge.api.routes.search import get_embedding_provider
from ragforge.core.config import Settings, get_settings
from ragforge.db.models import ChunkEmbedding, Document, DocumentChunk, DocumentStatus
from ragforge.db.session import get_session_factory
from ragforge.main import app
from ragforge.schemas.ask import AskRequest
from ragforge.services.ask_service import (
    ABSTENTION_MESSAGE_LOW_SIMILARITY,
    ABSTENTION_MESSAGE_NO_EVIDENCE,
    AskService,
)
from ragforge.services.embedding_provider import FakeEmbeddingProvider
from ragforge.services.generation_provider import FakeGenerationProvider
from ragforge.services.retrieval_service import RetrievalService


@pytest.fixture
def fake_embedding_prov() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(model_name="all-MiniLM-L6-v2", dimension=384)


@pytest.fixture
def fake_gen_prov() -> FakeGenerationProvider:
    return FakeGenerationProvider(model_name="fake-slm")


@pytest.fixture
def ask_settings() -> Settings:
    return Settings(
        ENVIRONMENT="testing",
        EMBEDDING_PROVIDER="fake",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        EMBEDDING_DIMENSION=384,
        GENERATION_PROVIDER="fake",
        GENERATION_MODEL="fake-slm",
        ASK_MIN_SIMILARITY_THRESHOLD=0.30,
        ASK_MAX_QUESTION_CHARS=500,
    )


@pytest.fixture
async def ask_client(
    fake_embedding_prov: FakeEmbeddingProvider,
    fake_gen_prov: FakeGenerationProvider,
    ask_settings: Settings,
) -> AsyncGenerator[AsyncClient, None]:
    app.dependency_overrides[get_settings] = lambda: ask_settings
    app.dependency_overrides[get_embedding_provider] = lambda: fake_embedding_prov
    app.dependency_overrides[get_generation_provider] = lambda: fake_gen_prov

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ask_service_successful_grounded_answer(
    fake_embedding_prov: FakeEmbeddingProvider,
    fake_gen_prov: FakeGenerationProvider,
    ask_settings: Settings,
) -> None:
    """Verify AskService retrieves candidate chunk, generates answer, and verifies citation."""
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            pass
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable")

    doc_id = uuid.uuid4()
    unique_tag = uuid.uuid4().hex[:8]
    fact_text = f"PostgreSQL pgvector supports IVFFlat and HNSW indexing structures {unique_tag}."

    # Compute fake embedding
    vec = await fake_embedding_prov.embed_query(fact_text)

    async with session_factory() as session:
        doc = Document(
            id=doc_id,
            filename=f"pgvector_guide_{unique_tag}.txt",
            storage_key=f"test/{doc_id}.txt",
            mime_type="text/plain",
            file_size_bytes=len(fact_text.encode("utf-8")),
            sha256_checksum="d" * 64,
            status=DocumentStatus.PROCESSED.value,
        )
        session.add(doc)
        await session.flush()

        chunk = DocumentChunk(
            document_id=doc_id,
            chunk_index=0,
            content=fact_text,
            page_number=4,
            character_count=len(fact_text),
            token_count_estimate=15,
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
            embedding_provider=fake_embedding_prov,
            settings=ask_settings,
        )
        service = AskService(
            retrieval_service=retrieval,
            generation_provider=fake_gen_prov,
            settings=ask_settings,
        )

        resp = await service.answer_question(
            AskRequest(
                question=fact_text,
                limit=5,
                document_id=doc_id,
            )
        )

        assert resp.abstained is False
        assert len(resp.sources) >= 1
        assert resp.sources[0].chunk_id == chunk.id
        assert resp.sources[0].page_number == 4
        assert resp.sources[0].source_id == "[source_1]"

        assert len(resp.citations) == 1
        assert resp.citations[0].source_id == "[source_1]"
        assert resp.citations[0].chunk_id == chunk.id
        assert resp.citations[0].filename == f"pgvector_guide_{unique_tag}.txt"
        assert resp.citations[0].page_number == 4

        assert resp.telemetry.retrieval_latency_ms >= 0.0
        assert resp.telemetry.generation_latency_ms >= 0.0
        assert resp.telemetry.used_chunk_count >= 1


@pytest.mark.asyncio
async def test_ask_service_abstains_when_zero_evidence(
    fake_embedding_prov: FakeEmbeddingProvider,
    fake_gen_prov: FakeGenerationProvider,
    ask_settings: Settings,
) -> None:
    """Verify system returns clean abstention when document filter yields no chunks."""
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            pass
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable")

    nonexistent_doc_id = uuid.uuid4()
    async with session_factory() as session:
        retrieval = RetrievalService(
            db=session,
            embedding_provider=fake_embedding_prov,
            settings=ask_settings,
        )
        service = AskService(
            retrieval_service=retrieval,
            generation_provider=fake_gen_prov,
            settings=ask_settings,
        )

        resp = await service.answer_question(
            AskRequest(
                question="What is the meaning of life?",
                limit=5,
                document_id=nonexistent_doc_id,
            )
        )

        assert resp.abstained is True
        assert resp.answer == ABSTENTION_MESSAGE_NO_EVIDENCE
        assert len(resp.sources) == 0
        assert len(resp.citations) == 0
        assert resp.telemetry.generation_latency_ms == 0.0


@pytest.mark.asyncio
async def test_ask_service_abstains_on_low_similarity(
    fake_embedding_prov: FakeEmbeddingProvider,
    fake_gen_prov: FakeGenerationProvider,
) -> None:
    """Verify system abstains if top chunk score is below configured threshold."""
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            pass
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable")

    strict_settings = Settings(
        ENVIRONMENT="testing",
        EMBEDDING_PROVIDER="fake",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        EMBEDDING_DIMENSION=384,
        GENERATION_PROVIDER="fake",
        GENERATION_MODEL="fake-slm",
        ASK_MIN_SIMILARITY_THRESHOLD=0.9999,  # Impossibly high threshold forces abstention
    )

    doc_id = uuid.uuid4()
    async with session_factory() as session:
        doc = Document(
            id=doc_id,
            filename="temp.txt",
            storage_key=f"test/{doc_id}.txt",
            mime_type="text/plain",
            file_size_bytes=50,
            sha256_checksum="e" * 64,
            status=DocumentStatus.PROCESSED.value,
        )
        session.add(doc)
        await session.flush()
        c = DocumentChunk(
            document_id=doc_id,
            chunk_index=0,
            content="Unrelated text.",
            character_count=15,
            token_count_estimate=5,
        )
        session.add(c)
        await session.flush()
        unrelated_vec = await fake_embedding_prov.embed_query("Unrelated text.")
        e = ChunkEmbedding(
            chunk_id=c.id,
            provider="fake",
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            embedding=unrelated_vec,
        )
        session.add(e)
        await session.commit()

    async with session_factory() as session:
        retrieval = RetrievalService(
            db=session,
            embedding_provider=fake_embedding_prov,
            settings=strict_settings,
        )
        service = AskService(
            retrieval_service=retrieval,
            generation_provider=fake_gen_prov,
            settings=strict_settings,
        )
        resp = await service.answer_question(
            AskRequest(question="A question?", limit=5, document_id=doc_id)
        )
        assert resp.abstained is True
        assert resp.answer == ABSTENTION_MESSAGE_LOW_SIMILARITY


@pytest.mark.asyncio
async def test_ask_endpoint_http_validation(ask_client: AsyncClient) -> None:
    """Verify HTTP route input validation and 422 on empty question."""
    resp = await ask_client.post("/api/v1/ask", json={"question": ""})
    assert resp.status_code == 422

    resp_valid = await ask_client.post(
        "/api/v1/ask",
        json={"question": "What is connection pooling?", "limit": 3},
    )
    assert resp_valid.status_code == 200
    data = resp_valid.json()
    assert "question" in data
    assert "answer" in data
    assert "telemetry" in data
    assert "sources" in data
    assert "citations" in data
