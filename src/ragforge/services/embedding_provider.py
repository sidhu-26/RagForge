"""Embedding provider abstractions and implementations.

Provides typed interfaces and adapters for generating dense vector embeddings:
- LocalSentenceTransformerProvider: Offline inference using sentence-transformers.
- AzureOpenAIEmbeddingProvider: Remote embeddings via Azure OpenAI REST API.
- FakeEmbeddingProvider: Fast deterministic pseudo-vectors for unit testing.
"""

import abc
import asyncio
import hashlib
import logging
from collections.abc import Sequence

import httpx

from ragforge.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding generation failures."""

    def __init__(self, message: str, error_code: str = "EMBEDDING_FAILED") -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when an embedding provider is misconfigured or missing credentials."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="EMBEDDING_CONFIG_ERROR")


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when an upstream provider throttles embedding requests (HTTP 429)."""

    def __init__(self, message: str = "Upstream embedding rate limit exceeded.") -> None:
        super().__init__(message, error_code="EMBEDDING_RATE_LIMIT")


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when the model's output vector dimension differs from the configured dimension."""

    def __init__(self, expected: int, actual: int) -> None:
        msg = f"Embedding dimension mismatch: expected {expected}, but model produced {actual}."
        super().__init__(msg, error_code="EMBEDDING_DIMENSION_MISMATCH")


class EmbeddingProvider(abc.ABC):
    """Abstract interface for text embedding models."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Name of the embedding provider backend (e.g. 'local', 'azure', 'fake')."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Identifier of the specific embedding model (e.g. 'all-MiniLM-L6-v2')."""
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Expected output vector dimension."""
        pass

    @abc.abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generate an embedding vector for a single query string.

        Args:
            query: Non-empty search query string.

        Returns:
            list[float]: Normalized embedding vector of length `dimension`.
        """
        pass

    @abc.abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of text chunks.

        Args:
            texts: Sequence of text chunks.

        Returns:
            list[list[float]]: List of normalized embedding vectors.
        """
        pass


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic pseudo-embedding provider for unit tests and local mock verification.

    Generates reproducible unit-normalized vectors derived from the SHA-256 hash of the input text.
    Does NOT require PyTorch or HuggingFace model downloads.
    """

    def __init__(self, model_name: str = "fake-embed-v1", dimension: int = 384) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> list[float]:
        if not text:
            raise EmbeddingError("Cannot embed empty text string.")

        # Hash text to seed deterministic floats
        raw_hash = hashlib.sha256(text.encode("utf-8")).digest()
        vector: list[float] = []
        for i in range(self._dimension):
            # Pick cyclic bytes from hash
            byte_val = raw_hash[i % len(raw_hash)]
            val = ((byte_val / 255.0) * 2.0 - 1.0) * (1.0 + (i % 7) * 0.1)
            vector.append(val)

        # Normalize to unit length (L2 norm)
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            return [x / norm for x in vector]
        return vector

    async def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise EmbeddingError("Cannot embed empty query.")
        return self._generate_vector(query.strip())

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._generate_vector(t) for t in texts]


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """Local offline sentence-transformers embedding provider.

    Loads model lazily upon first use or explicit warm-up to prevent blocking module imports.
    Runs inference inside asyncio.to_thread to keep the event loop responsive.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        expected_dimension: int = 384,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._expected_dimension = expected_dimension
        self._batch_size = batch_size
        self._model: object | None = None
        self._lock = asyncio.Lock()

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._expected_dimension

    def _load_model(self) -> object:
        """Synchronously instantiate SentenceTransformer."""
        if self._model is None:
            logger.info(
                "Loading local sentence-transformers model",
                extra={"model": self._model_name},
            )
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                logger.exception("Failed to load sentence-transformers model")
                raise EmbeddingError(
                    f"Failed to load local embedding model '{self._model_name}': {exc}"
                ) from exc
        return self._model

    async def ensure_loaded(self) -> None:
        """Explicitly warm up and load the model."""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    await asyncio.to_thread(self._load_model)

    async def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise EmbeddingError("Cannot embed empty query string.")

        await self.ensure_loaded()

        def _encode() -> list[float]:
            model = self._load_model()
            # encode returns numpy array or list
            vector = model.encode(  # type: ignore[attr-defined]
                query,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [float(x) for x in vector]

        vec = await asyncio.to_thread(_encode)
        if len(vec) != self._expected_dimension:
            raise EmbeddingDimensionMismatchError(self._expected_dimension, len(vec))
        return vec

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        await self.ensure_loaded()

        def _encode_batch() -> list[list[float]]:
            model = self._load_model()
            embeddings = model.encode(  # type: ignore[attr-defined]
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [[float(x) for x in row] for row in embeddings]

        vectors = await asyncio.to_thread(_encode_batch)
        for vec in vectors:
            if len(vec) != self._expected_dimension:
                raise EmbeddingDimensionMismatchError(self._expected_dimension, len(vec))
        return vectors


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """Azure OpenAI embeddings adapter using direct REST API calls.

    Requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.
    """

    def __init__(
        self,
        endpoint: str | None,
        api_key: str | None,
        deployment_name: str | None,
        api_version: str = "2024-02-01",
        expected_dimension: int = 1536,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not endpoint or not endpoint.strip():
            raise EmbeddingConfigurationError(
                "Azure OpenAI embedding provider selected, but "
                "RAGFORGE_AZURE_OPENAI_ENDPOINT is not configured."
            )
        if not api_key or not api_key.strip():
            raise EmbeddingConfigurationError(
                "Azure OpenAI embedding provider selected, but "
                "RAGFORGE_AZURE_OPENAI_API_KEY is not configured."
            )
        if not deployment_name or not deployment_name.strip():
            raise EmbeddingConfigurationError(
                "Azure OpenAI embedding provider selected, but "
                "RAGFORGE_AZURE_OPENAI_DEPLOYMENT_NAME is not configured."
            )

        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._deployment = deployment_name
        self._api_version = api_version
        self._expected_dimension = expected_dimension
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "azure"

    @property
    def model_name(self) -> str:
        return self._deployment

    @property
    def dimension(self) -> int:
        return self._expected_dimension

    async def _call_api(self, input_data: str | list[str]) -> list[list[float]]:
        """Issue embedding request to Azure OpenAI endpoint with bounded retries."""
        url = (
            f"{self._endpoint}/openai/deployments/{self._deployment}/embeddings"
            f"?api-version={self._api_version}"
        )
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {"input": input_data}

        attempts = 0
        last_error: Exception | None = None

        while attempts <= self._max_retries:
            attempts += 1
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, headers=headers, json=payload)

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 2.0))
                    if attempts <= self._max_retries:
                        await asyncio.sleep(retry_after)
                        continue
                    raise EmbeddingRateLimitError()

                if resp.status_code != 200:
                    raise EmbeddingError(
                        f"Azure OpenAI API returned HTTP {resp.status_code}",
                        error_code="AZURE_API_ERROR",
                    )

                data = resp.json()
                sorted_items = sorted(data["data"], key=lambda item: item["index"])
                vectors: list[list[float]] = [item["embedding"] for item in sorted_items]

                for v in vectors:
                    if len(v) != self._expected_dimension:
                        raise EmbeddingDimensionMismatchError(self._expected_dimension, len(v))
                return vectors

            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempts <= self._max_retries:
                    await asyncio.sleep(0.5 * (2 ** (attempts - 1)))
                    continue
                break

        raise EmbeddingError(
            f"Failed to connect to Azure OpenAI embedding service: {last_error}",
            error_code="AZURE_CONNECTION_ERROR",
        )

    async def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise EmbeddingError("Cannot embed empty query string.")
        vectors = await self._call_api(query.strip())
        return vectors[0]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._call_api(list(texts))


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory creating configured embedding provider based on application settings."""
    provider_type = settings.EMBEDDING_PROVIDER

    if provider_type == "fake":
        return FakeEmbeddingProvider(
            model_name=settings.EMBEDDING_MODEL,
            dimension=settings.EMBEDDING_DIMENSION,
        )

    if provider_type == "azure":
        api_key_str = (
            settings.AZURE_OPENAI_API_KEY.get_secret_value()
            if settings.AZURE_OPENAI_API_KEY
            else None
        )
        return AzureOpenAIEmbeddingProvider(
            endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=api_key_str,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            expected_dimension=settings.EMBEDDING_DIMENSION,
            timeout_seconds=settings.EMBEDDING_REQUEST_TIMEOUT_SECONDS,
            max_retries=settings.EMBEDDING_MAX_RETRIES,
        )

    # Default: local sentence-transformers provider
    return LocalSentenceTransformerProvider(
        model_name=settings.EMBEDDING_MODEL,
        expected_dimension=settings.EMBEDDING_DIMENSION,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
