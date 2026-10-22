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
