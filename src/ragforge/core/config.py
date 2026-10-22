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
