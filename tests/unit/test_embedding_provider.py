"""Unit tests for embedding providers (Fake, Local, and Azure)."""

import pytest

from ragforge.core.config import Settings
from ragforge.services.embedding_provider import (
    AzureOpenAIEmbeddingProvider,
    EmbeddingConfigurationError,
    EmbeddingError,
    FakeEmbeddingProvider,
    LocalSentenceTransformerProvider,
    create_embedding_provider,
)


def test_fake_embedding_provider_shape_and_norm() -> None:
    """Verify FakeEmbeddingProvider produces unit-normalized vectors with expected dimension."""
    provider = FakeEmbeddingProvider(model_name="test-fake", dimension=128)
    assert provider.provider_name == "fake"
    assert provider.model_name == "test-fake"
    assert provider.dimension == 128


@pytest.mark.asyncio
async def test_fake_embedding_embed_query_and_texts() -> None:
    """Verify deterministic embedding generation and batch handling."""
    provider = FakeEmbeddingProvider(dimension=64)

    # Empty query raises error
    with pytest.raises(EmbeddingError):
        await provider.embed_query("   ")

    vec1 = await provider.embed_query("Hello RAGForge")
    assert len(vec1) == 64
    # Unit norm check
    norm1 = sum(x * x for x in vec1) ** 0.5
    assert abs(norm1 - 1.0) < 1e-4

    # Identical query produces identical vector (deterministic)
    vec1_repeat = await provider.embed_query("Hello RAGForge")
    assert vec1 == vec1_repeat

    # Batch embedding
    batch_vecs = await provider.embed_texts(["Chunk 1", "Chunk 2"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 64
    assert len(batch_vecs[1]) == 64
    assert batch_vecs[0] != batch_vecs[1]


def test_azure_provider_missing_config_raises_error() -> None:
    """Verify Azure provider fails immediately with clear configuration error when unconfigured."""
    with pytest.raises(EmbeddingConfigurationError) as exc_1:
        AzureOpenAIEmbeddingProvider(
            endpoint=None,
            api_key="secret",
            deployment_name="embed-dep",
        )
    assert "ENDPOINT" in str(exc_1.value)

    with pytest.raises(EmbeddingConfigurationError) as exc_2:
        AzureOpenAIEmbeddingProvider(
            endpoint="https://example.openai.azure.com",
            api_key=None,
            deployment_name="embed-dep",
        )
    assert "API_KEY" in str(exc_2.value)

    with pytest.raises(EmbeddingConfigurationError) as exc_3:
        AzureOpenAIEmbeddingProvider(
            endpoint="https://example.openai.azure.com",
            api_key="secret",
            deployment_name=None,
        )
    assert "DEPLOYMENT_NAME" in str(exc_3.value)


def test_create_embedding_provider_factory() -> None:
    """Verify create_embedding_provider instantiates proper subclass based on settings."""
    # 1. Fake
    s_fake = Settings(EMBEDDING_PROVIDER="fake", EMBEDDING_DIMENSION=128)
    p_fake = create_embedding_provider(s_fake)
    assert isinstance(p_fake, FakeEmbeddingProvider)
    assert p_fake.dimension == 128

    # 2. Local
    s_local = Settings(EMBEDDING_PROVIDER="local", EMBEDDING_MODEL="all-MiniLM-L6-v2")
    p_local = create_embedding_provider(s_local)
    assert isinstance(p_local, LocalSentenceTransformerProvider)
    assert p_local.model_name == "all-MiniLM-L6-v2"

    # 3. Azure missing credentials raises error
    s_azure_bad = Settings(EMBEDDING_PROVIDER="azure")
    with pytest.raises(EmbeddingConfigurationError):
        create_embedding_provider(s_azure_bad)


@pytest.mark.asyncio
async def test_local_sentence_transformer_provider_live_or_skip() -> None:
    """Test real local sentence-transformers inference if dependencies are present."""
    provider = LocalSentenceTransformerProvider(
        model_name="all-MiniLM-L6-v2",
        expected_dimension=384,
    )
    vec = await provider.embed_query("Testing RAGForge local embeddings.")
    assert len(vec) == 384
    # Unit length check
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-3

    batch = await provider.embed_texts(["First test sentence.", "Second test sentence."])
    assert len(batch) == 2
    assert len(batch[0]) == 384
    assert len(batch[1]) == 384
