"""Unit tests for configuration loading and validation."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ragforge.core.config import EnvironmentType, LogLevel, Settings


def test_default_settings() -> None:
    """Verify that default settings instantiate with expected values."""
    settings = Settings()
    assert settings.PROJECT_NAME == "RAGForge API"
    assert settings.VERSION == "0.1.0"
    assert settings.ENVIRONMENT == EnvironmentType.DEVELOPMENT
    assert settings.LOG_LEVEL == LogLevel.INFO
    assert settings.PORT == 8000
    assert settings.CORS_ORIGINS == []
    assert settings.is_production is False
    assert settings.is_testing is False


def test_settings_from_env_vars() -> None:
    """Verify that settings correctly parse prefixed environment variables."""
    env = {
        "RAGFORGE_ENVIRONMENT": "production",
        "RAGFORGE_LOG_LEVEL": "DEBUG",
        "RAGFORGE_PORT": "9000",
        "RAGFORGE_CORS_ORIGINS": '["https://ragforge.example.com"]',
    }
    with patch.dict(os.environ, env, clear=True):
        settings = Settings()
        assert settings.ENVIRONMENT == EnvironmentType.PRODUCTION
        assert settings.LOG_LEVEL == LogLevel.DEBUG
        assert settings.PORT == 9000
        assert settings.CORS_ORIGINS == ["https://ragforge.example.com"]
        assert settings.is_production is True


def test_invalid_environment_raises_validation_error() -> None:
    """Verify that an invalid environment value raises a Pydantic ValidationError."""
    with (
        patch.dict(os.environ, {"RAGFORGE_ENVIRONMENT": "invalid_env"}, clear=True),
        pytest.raises(ValidationError),
    ):
        Settings()


def test_invalid_port_range_raises_validation_error() -> None:
    """Verify that ports outside [1, 65535] fail validation."""
    with (
        patch.dict(os.environ, {"RAGFORGE_PORT": "70000"}, clear=True),
        pytest.raises(ValidationError),
    ):
        Settings()


def test_database_settings_defaults() -> None:
    """Verify default database settings and pool parameters."""
    settings = Settings()
    assert (
        settings.DATABASE_URL.get_secret_value()
        == "postgresql+asyncpg://ragforge_user:ragforge_password@localhost:5432/ragforge_db"
    )
    assert settings.DB_POOL_SIZE == 5
    assert settings.DB_MAX_OVERFLOW == 10
    assert settings.DB_POOL_TIMEOUT == 30.0
    assert settings.DB_POOL_RECYCLE == 1800
    assert settings.DB_POOL_PRE_PING is True
    assert settings.DB_ECHO is False


def test_database_url_validation_invalid_scheme() -> None:
    """Verify that non-asyncpg PostgreSQL scheme raises ValidationError."""
    with (
        patch.dict(
            os.environ,
            {"RAGFORGE_DATABASE_URL": "postgresql://user:pass@localhost:5432/ragforge_db"},
            clear=True,
        ),
        pytest.raises(ValidationError, match="postgresql\\+asyncpg://"),
    ):
        Settings()


def test_database_custom_pool_settings() -> None:
    """Verify custom database pooling settings parsed from env vars."""
    env = {
        "RAGFORGE_DATABASE_URL": "postgresql+asyncpg://app_user:secret@db.internal:5432/production_db",
        "RAGFORGE_DB_POOL_SIZE": "20",
        "RAGFORGE_DB_MAX_OVERFLOW": "30",
        "RAGFORGE_DB_POOL_TIMEOUT": "15.5",
        "RAGFORGE_DB_POOL_RECYCLE": "3600",
        "RAGFORGE_DB_POOL_PRE_PING": "false",
        "RAGFORGE_DB_ECHO": "true",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = Settings()
        assert (
            settings.DATABASE_URL.get_secret_value()
            == "postgresql+asyncpg://app_user:secret@db.internal:5432/production_db"
        )
        assert settings.DB_POOL_SIZE == 20
        assert settings.DB_MAX_OVERFLOW == 30
        assert settings.DB_POOL_TIMEOUT == 15.5
        assert settings.DB_POOL_RECYCLE == 3600
        assert settings.DB_POOL_PRE_PING is False
        assert settings.DB_ECHO is True


def test_storage_settings_defaults_and_s3_aliases() -> None:
    """Verify default storage settings and alias resolution for S3."""
    from ragforge.core.config import StorageBackendType

    # Verify default without env overrides or .env file values
    with patch.dict(os.environ, {}, clear=True):
        default_settings = Settings(_env_file=None)
        assert default_settings.STORAGE_BACKEND == StorageBackendType.LOCAL
        assert default_settings.S3_BUCKET == "sociofitt-community"
        assert default_settings.S3_REGION == "ap-south-1"
        assert default_settings.S3_PREFIX == "prod/staging/posts/572/"

    # Test alias resolution from standard AWS environment variables
    env = {
        "RAGFORGE_STORAGE_BACKEND": "s3",
        "AWS_STORAGE_BUCKET_NAME": "custom-community-bucket",
        "AWS_S3_REGION_NAME": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test-key-id",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
    }
    with patch.dict(os.environ, env, clear=False):
        s3_settings = Settings()
        assert s3_settings.STORAGE_BACKEND == StorageBackendType.S3
        assert s3_settings.S3_BUCKET == "custom-community-bucket"
        assert s3_settings.S3_REGION == "us-east-1"
        assert s3_settings.AWS_ACCESS_KEY_ID is not None
        assert s3_settings.AWS_ACCESS_KEY_ID.get_secret_value() == "test-key-id"
        assert s3_settings.AWS_SECRET_ACCESS_KEY is not None
        assert s3_settings.AWS_SECRET_ACCESS_KEY.get_secret_value() == "test-secret-key"
