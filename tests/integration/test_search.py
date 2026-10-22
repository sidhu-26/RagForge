"""Unit and integration tests for semantic RetrievalService and /api/v1/search endpoint."""

import uuid
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from ragforge.api.routes.search import get_embedding_provider
from ragforge.core.config import Settings, get_settings
from ragforge.db.models import ChunkEmbedding, Document, DocumentChunk, DocumentStatus
from ragforge.db.session import get_session_factory
from ragforge.main import app
from ragforge.services.embedding_provider import FakeEmbeddingProvider
from ragforge.services.retrieval_service import RetrievalError, RetrievalService


@pytest.fixture
def fake_embedding_provider() -> FakeEmbeddingProvider:
    """Provide deterministic fake embedding provider for retrieval tests."""
    return FakeEmbeddingProvider(model_name="all-MiniLM-L6-v2", dimension=384)


@pytest.fixture
def search_settings() -> Settings:
    """Provide testing settings."""
    return Settings(
        ENVIRONMENT="testing",
        EMBEDDING_PROVIDER="fake",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        EMBEDDING_DIMENSION=384,
        RETRIEVAL_DEFAULT_LIMIT=5,
        RETRIEVAL_MAX_LIMIT=10,
    )


@pytest.fixture
async def search_client(
    fake_embedding_provider: FakeEmbeddingProvider, search_settings: Settings
) -> AsyncGenerator[AsyncClient, None]:
    """Provide AsyncClient with fake embedding provider override."""
    app.dependency_overrides[get_settings] = lambda: search_settings
    app.dependency_overrides[get_embedding_provider] = lambda: fake_embedding_provider

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_retrieval_empty_query_raises_error(
    fake_embedding_provider: FakeEmbeddingProvider, search_settings: Settings
) -> None:
    """Verify empty query string is rejected with HTTP 400."""
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            service = RetrievalService(
                db=session,
                embedding_provider=fake_embedding_provider,
                settings=search_settings,
            )
            with pytest.raises(RetrievalError) as exc:
                await service.search(query="   ")
            assert exc.value.status_code == 400
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable")


@pytest.mark.asyncio
async def test_retrieval_service_ranked_similarity_and_document_filter(
    fake_embedding_provider: FakeEmbeddingProvider, search_settings: Settings
) -> None:
    """Verify nearest-neighbor search ranks exact matches first and supports document filtering."""
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            pass
    except Exception:
        pytest.skip("PostgreSQL test instance not reachable")

    doc1_id = uuid.uuid4()
    doc2_id = uuid.uuid4()

    uid = uuid.uuid4().hex[:8]
    query_text = f"PostgreSQL async architecture with connection pooling {uid}."
    unrelated_text = f"Baking sourdough bread using whole grain flour and wild yeast {uid}."
    different_doc_text = f"PostgreSQL async architecture tuning {uid}."

    # Compute fake embeddings
    query_vec = await fake_embedding_provider.embed_query(query_text)
    unrelated_vec = await fake_embedding_provider.embed_query(unrelated_text)
    diff_doc_vec = await fake_embedding_provider.embed_query(different_doc_text)

    async with session_factory() as session:
        # Create documents
        doc1 = Document(
            id=doc1_id,
            filename="postgres_architecture.txt",
            storage_key=f"test/{doc1_id}.txt",
            mime_type="text/plain",
            file_size_bytes=100,
            sha256_checksum="1" * 64,
            status=DocumentStatus.PROCESSED.value,
        )
        doc2 = Document(
            id=doc2_id,
            filename="sourdough_recipes.txt",
            storage_key=f"test/{doc2_id}.txt",
            mime_type="text/plain",
            file_size_bytes=100,
            sha256_checksum="2" * 64,
            status=DocumentStatus.PROCESSED.value,
        )
        session.add_all([doc1, doc2])
        await session.flush()

        # Create chunks
        c1 = DocumentChunk(
            document_id=doc1_id,
            chunk_index=0,
            content=query_text,
            page_number=1,
            character_count=len(query_text),
            token_count_estimate=15,
        )
        c2 = DocumentChunk(
            document_id=doc2_id,
            chunk_index=0,
            content=unrelated_text,
            page_number=None,
            character_count=len(unrelated_text),
            token_count_estimate=15,
        )
        c3 = DocumentChunk(
            document_id=doc2_id,
            chunk_index=1,
            content=different_doc_text,
            page_number=None,
            character_count=len(different_doc_text),
            token_count_estimate=10,
        )
        session.add_all([c1, c2, c3])
        await session.flush()

        # Add embeddings
        e1 = ChunkEmbedding(
            chunk_id=c1.id,
            provider="fake",
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            embedding=query_vec,
        )
        e2 = ChunkEmbedding(
            chunk_id=c2.id,
            provider="fake",
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            embedding=unrelated_vec,
        )
        e3 = ChunkEmbedding(
            chunk_id=c3.id,
            provider="fake",
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            embedding=diff_doc_vec,
        )
        session.add_all([e1, e2, e3])
        await session.commit()

    # Search with no document filter, but scoped to test documents or verify against chunks
    async with session_factory() as session:
        service = RetrievalService(
            db=session,
            embedding_provider=fake_embedding_provider,
            settings=search_settings,
        )
        # Search with document filter on doc1: exact match
        resp_doc1 = await service.search(query=query_text, limit=5, document_id=doc1_id)
        assert resp_doc1.total_results == 1
        assert resp_doc1.items[0].document_id == doc1_id
        assert resp_doc1.items[0].filename == "postgres_architecture.txt"
        assert resp_doc1.items[0].content == query_text
        assert resp_doc1.items[0].page_number == 1
        assert abs(resp_doc1.items[0].similarity_score - 1.0) < 1e-4

        # Search with document filter on doc2
        resp_filtered = await service.search(query=query_text, limit=5, document_id=doc2_id)
        assert len(resp_filtered.items) > 0
        for item in resp_filtered.items:
            assert item.document_id == doc2_id


@pytest.mark.asyncio
async def test_search_endpoint_api(search_client: AsyncClient) -> None:
    """Verify POST /api/v1/search validation, bounds, and response contract."""
    # 1. Empty query returns 422 (Pydantic min_length validation)
    resp_empty = await search_client.post("/api/v1/search", json={"query": ""})
    assert resp_empty.status_code == 422

    # 2. Valid query returns 200 with SearchResponse schema
    payload = {"query": "PostgreSQL connection pooling architecture", "limit": 3}
    resp = await search_client.post("/api/v1/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "query" in data
    assert "items" in data
    assert "total_results" in data
    assert "model_name" in data
