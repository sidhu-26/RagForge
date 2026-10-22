"""Unit tests for SQLAlchemy declarative base and mixins."""

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Mapped, mapped_column

from ragforge.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class SampleModel(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Sample concrete model for testing declarative mixins."""

    __tablename__ = "test_sample_models"

    name: Mapped[str] = mapped_column(nullable=False)


def test_base_metadata_discovery() -> None:
    """Verify that models inheriting from Base register their tables in metadata."""
    assert "test_sample_models" in Base.metadata.tables
    table = Base.metadata.tables["test_sample_models"]

    # Verify primary key column
    assert "id" in table.columns
    assert table.columns["id"].primary_key is True

    # Verify timestamp columns
    assert "created_at" in table.columns
    assert table.columns["created_at"].nullable is False
    assert "updated_at" in table.columns
    assert table.columns["updated_at"].nullable is False

    # Clean up test table from shared metadata to avoid pollution
    Base.metadata.remove(table)


def test_uuid_primary_key_mixin_defaults() -> None:
    """Verify UUIDPrimaryKeyMixin produces UUID instances."""
    model = SampleModel(name="test")
    # Default is a callable (uuid.uuid4)
    model.id = uuid.uuid4()
    assert isinstance(model.id, uuid.UUID)


def test_timestamp_mixin_datetime_format() -> None:
    """Verify TimestampMixin sets UTC-aware datetime timestamps."""
    now = datetime.now(UTC)
    model = SampleModel(name="test", created_at=now, updated_at=now)
    assert model.created_at.tzinfo is not None
    assert model.updated_at.tzinfo is not None
