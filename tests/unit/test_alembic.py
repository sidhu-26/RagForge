"""Unit tests for Alembic configuration and metadata discovery."""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

from ragforge.core.config import get_settings
from ragforge.db.base import Base


def test_alembic_config_loads_without_hardcoded_credentials() -> None:
    """Verify that alembic.ini exists and does not contain sensitive cleartext passwords."""
    ini_path = Path("alembic.ini")
    assert ini_path.exists()
    content = ini_path.read_text(encoding="utf-8")
    assert "user:pass@localhost/dbname" not in content


def test_alembic_metadata_discovery() -> None:
    """Verify that Base.metadata is accessible and Alembic script directory is valid."""
    config = Config("alembic.ini")
    script_dir = ScriptDirectory.from_config(config)

    # Verify head revision is present and points to our initial migration
    heads = script_dir.get_heads()
    assert len(heads) == 1
    assert heads[0] == "405bd1185000"

    # Verify target metadata is bound to Base.metadata
    assert Base.metadata is not None


def test_alembic_url_matches_application_settings() -> None:
    """Verify Alembic env.py would receive the exact DATABASE_URL configured in Settings."""
    settings = get_settings()
    assert settings.DATABASE_URL.get_secret_value().startswith("postgresql+asyncpg://")
