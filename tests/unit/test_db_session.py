"""Unit tests for database session lifecycle, engine lifecycle, and error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ragforge.db.session import close_db_engine, get_db, get_db_engine, get_session_factory


@pytest.mark.asyncio
async def test_get_db_engine_singleton() -> None:
    """Verify that get_db_engine initializes once and returns the same AsyncEngine singleton."""
    await close_db_engine()
    engine1 = get_db_engine()
    engine2 = get_db_engine()
    assert isinstance(engine1, AsyncEngine)
    assert engine1 is engine2
    await close_db_engine()


@pytest.mark.asyncio
async def test_get_session_factory_singleton() -> None:
    """Verify get_session_factory returns consistent async_sessionmaker bound to the engine."""
    await close_db_engine()
    factory1 = get_session_factory()
    factory2 = get_session_factory()
    assert isinstance(factory1, async_sessionmaker)
    assert factory1 is factory2
    await close_db_engine()


@pytest.mark.asyncio
async def test_close_db_engine_disposes_cleanly() -> None:
    """Verify that close_db_engine disposes engine and resets singleton references."""
    await close_db_engine()
    engine = get_db_engine()
    assert engine is not None
    await close_db_engine()
    # Re-calling close when already None should be safe and idempotent
    await close_db_engine()


@pytest.mark.asyncio
async def test_get_db_session_lifecycle_success() -> None:
    """Verify request-scoped get_db creates a session, yields it, and closes it."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_factory = MagicMock(return_value=mock_session)

    with patch("ragforge.db.session.get_session_factory", return_value=mock_factory):
        generator = get_db()
        yielded_session = await anext(generator)
        assert yielded_session is mock_session
        assert not mock_session.close.called

        # Simulate completion of request
        with pytest.raises(StopAsyncIteration):
            await anext(generator)

        mock_session.close.assert_awaited_once()
        mock_session.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_get_db_session_lifecycle_rollback_on_error() -> None:
    """Verify get_db rolls back on unhandled exceptions and always closes the session."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_factory = MagicMock(return_value=mock_session)

    with patch("ragforge.db.session.get_session_factory", return_value=mock_factory):
        generator = get_db()
        yielded_session = await anext(generator)
        assert yielded_session is mock_session

        # Simulate an exception raised within the route context
        test_error = RuntimeError("Simulated route error")
        with pytest.raises(RuntimeError, match="Simulated route error"):
            await generator.athrow(test_error)

        mock_session.rollback.assert_awaited_once()
        mock_session.close.assert_awaited_once()
