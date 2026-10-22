"""Centralized application settings using Pydantic Settings."""

from enum import StrEnum
from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentType(StrEnum):
    """Application runtime environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(StrEnum):
    """Supported logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingProviderType(StrEnum):
    """Supported embedding provider backends."""

    LOCAL = "local"
    AZURE = "azure"
    FAKE = "fake"


class GenerationProviderType(StrEnum):
    """Supported answer generation provider backends."""

    AZURE = "azure"
    FAKE = "fake"
    GEMINI = "gemini"


class Settings(BaseSettings):
    """Central configuration for RAGForge.

    Values are loaded with precedence:
    1. System environment variables (prefixed with RAGFORGE_)
    2. Local .env file
    3. Field defaults defined below
    """

    model_config = SettingsConfigDict(
        env_prefix="RAGFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application metadata
    PROJECT_NAME: str = "RAGForge API"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Portfolio-grade SLM-powered Retrieval-Augmented Generation platform"
    API_V1_PREFIX: str = "/api/v1"

    # Runtime & Logging
    ENVIRONMENT: EnvironmentType = EnvironmentType.DEVELOPMENT
    LOG_LEVEL: LogLevel = LogLevel.INFO
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)

    # Security & CORS
    # In development, empty list means same-origin only. Explicit origins required.
    CORS_ORIGINS: list[str] = Field(default_factory=list)
    SECRET_KEY: SecretStr = SecretStr("insecure_dev_secret_key_needs_override_in_production")

    # Database connection & pooling (PostgreSQL async engine via asyncpg)
    DATABASE_URL: SecretStr = SecretStr(
        "postgresql+asyncpg://ragforge_user:ragforge_password@localhost:5432/ragforge_db"
    )
    DB_POOL_SIZE: int = Field(default=5, ge=1, le=100, description="Connection pool size")
    DB_MAX_OVERFLOW: int = Field(
        default=10, ge=0, le=100, description="Max overflow connections beyond pool size"
    )
    DB_POOL_TIMEOUT: float = Field(
        default=30.0, ge=1.0, description="Seconds to wait before timing out acquiring connection"
    )
    DB_POOL_RECYCLE: int = Field(
        default=1800, ge=-1, description="Seconds before recycling connection (-1 for no recycle)"
    )
    DB_POOL_PRE_PING: bool = Field(
        default=True, description="Enable connection health-checks before checkout (SELECT 1 ping)"
    )
    DB_ECHO: bool = Field(
        default=False, description="Log all executed SQL statements (disabled by default)"
    )

    # Document Storage Configuration
    STORAGE_LOCAL_ROOT: str = Field(
        default="data/storage",
        description="Root directory for local filesystem document storage",
    )
    STORAGE_MAX_UPLOAD_SIZE_BYTES: int = Field(
        default=25 * 1024 * 1024,  # 25 MB
        ge=1024,
        le=500 * 1024 * 1024,
        description="Maximum upload file size allowed in bytes (default 25 MB)",
    )
    STORAGE_CHUNK_SIZE_BYTES: int = Field(
        default=64 * 1024,  # 64 KB
        ge=4096,
        le=10 * 1024 * 1024,
        description="Chunk size for streaming uploads to disk and hashing (default 64 KB)",
    )

    # Redis and ARQ Queue Configuration
    REDIS_URL: SecretStr = SecretStr("redis://localhost:6379/0")
    QUEUE_JOB_TIMEOUT_SECONDS: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Maximum execution timeout for worker jobs in seconds",
    )
    WORKER_MAX_TRIES: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for worker jobs",
    )

    # Text Extraction Limits
    EXTRACTION_MAX_PAGES: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum allowed pages in a document (e.g. PDF)",
    )
    EXTRACTION_MAX_TEXT_SIZE_BYTES: int = Field(
        default=10 * 1024 * 1024,  # 10 MB text
        ge=1024,
        le=100 * 1024 * 1024,
        description="Maximum extracted plain text size in bytes",
    )

    # Chunking Configuration
    CHUNK_TARGET_SIZE: int = Field(
        default=1000,
        ge=50,
        le=10000,
        description="Target chunk size in characters",
    )
    CHUNK_OVERLAP: int = Field(
        default=150,
        ge=0,
        le=2000,
        description="Overlap between adjacent chunks in characters",
    )
    CHUNK_MIN_SIZE: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Minimum useful chunk size in characters",
    )
    MAX_CHUNKS_PER_DOCUMENT: int = Field(
        default=2000,
        ge=1,
        le=20000,
        description="Maximum allowed chunks per document",
    )

    # Embedding Provider & Model Configuration
    EMBEDDING_PROVIDER: EmbeddingProviderType = EmbeddingProviderType.LOCAL
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "Model identifier (e.g. all-MiniLM-L6-v2 for local, text-embedding-3-small for Azure)"
        ),
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        ge=1,
        le=4096,
        description="Expected embedding vector dimension (384 for all-MiniLM-L6-v2)",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for generating embeddings",
    )
    EMBEDDING_MAX_INPUT_CHARS: int = Field(
        default=8192,
        ge=128,
        le=65536,
        description="Maximum input characters allowed per text chunk for embedding",
    )
    EMBEDDING_REQUEST_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout in seconds for remote embedding API calls",
    )
    EMBEDDING_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max retries for transient embedding provider errors",
    )

    # Azure OpenAI Embedding Configuration (Optional; required only if EMBEDDING_PROVIDER=azure)
    AZURE_OPENAI_ENDPOINT: str | None = Field(
        default=None,
        description="Azure OpenAI endpoint URL (e.g. https://<res>.openai.azure.com/)",
    )
    AZURE_OPENAI_API_KEY: SecretStr | None = Field(
        default=None,
        description="Azure OpenAI API key",
    )
    AZURE_OPENAI_API_VERSION: str = Field(
        default="2024-02-01",
        description="Azure OpenAI REST API version",
    )
    AZURE_OPENAI_DEPLOYMENT_NAME: str | None = Field(
        default=None,
        description="Azure OpenAI deployment name for embeddings",
    )

    # Semantic Retrieval Configuration
    RETRIEVAL_DEFAULT_LIMIT: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Default number of ranked results returned by semantic search",
    )
    RETRIEVAL_MAX_LIMIT: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of ranked results allowed per query",
    )

    # Answer Generation & SLM Configuration (Milestone 6)
    GENERATION_PROVIDER: GenerationProviderType = Field(
        default=GenerationProviderType.FAKE,
        description="Generation provider backend: azure | fake | gemini",
    )
    GENERATION_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Model identifier (e.g. gpt-4o-mini, phi-3-mini, gemini-2.5-flash)",
    )
    GENERATION_MAX_OUTPUT_TOKENS: int = Field(
        default=1024,
        ge=16,
        le=4096,
        description="Maximum tokens the model is permitted to generate",
    )
    GENERATION_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 for deterministic factual answers)",
    )
    GENERATION_REQUEST_TIMEOUT_SECONDS: float = Field(
        default=45.0,
        ge=1.0,
        le=300.0,
        description="Request timeout in seconds for generation API calls",
    )
    GENERATION_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max retries for transient generation errors",
    )

    # Azure OpenAI Chat/Generation Deployment
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: str | None = Field(
        default=None,
        description="Azure OpenAI deployment name for chat/completions (e.g. gpt-4o-mini)",
    )

    # Google Gemini Generation Configuration (Required only if GENERATION_PROVIDER=gemini)
    GEMINI_API_KEY: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key for the Developer API",
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Google Gemini model identifier (e.g. gemini-2.5-flash)",
    )

    # Prompt & Evidence Context Bounds
    ASK_MAX_QUESTION_CHARS: int = Field(
        default=1000,
        ge=1,
        le=5000,
        description="Maximum allowed character length for a user question",
    )
    ASK_DEFAULT_RETRIEVAL_LIMIT: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of chunks retrieved for question answering",
    )
    ASK_MAX_EVIDENCE_TOTAL_CHARS: int = Field(
        default=12000,
        ge=500,
        le=50000,
        description="Maximum allowed character budget for concatenated retrieved evidence context",
    )
    ASK_MIN_SIMILARITY_THRESHOLD: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity score required for top chunk before abstaining "
            "(0.0 means never abstain on low similarity)"
        ),
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: SecretStr) -> SecretStr:
        """Validate that the database connection string uses asyncpg driver."""
        url_str = v.get_secret_value()
        if not url_str.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must start with 'postgresql+asyncpg://' for async PostgreSQL support"
            )
        return v

    @property
    def is_production(self) -> bool:
        """Helper to determine whether the application runs in production."""
        return self.ENVIRONMENT == EnvironmentType.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Helper to determine whether the application runs in testing."""
        return self.ENVIRONMENT == EnvironmentType.TESTING


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached application settings instance.

    Uses lru_cache so environment variables are parsed once at startup,
    which can be cleanly overridden in test fixtures using app.dependency_overrides.
    """
    return Settings()


SettingsDep = Annotated[Settings, Field()]
