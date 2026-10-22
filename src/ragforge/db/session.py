"""Database session management, engine lifecycle, and FastAPI dependency injection."""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ragforge.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_db_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return the singleton AsyncEngine instance, initializing it if necessary."""
    global _engine
    if _engine is None:
        cfg = settings or get_settings()
        # Protect credentials: log host/db without credentials
        logger.info(
            "Initializing PostgreSQL async database engine",
            extra={
                "pool_size": cfg.DB_POOL_SIZE,
                "max_overflow": cfg.DB_MAX_OVERFLOW,
                "pool_timeout": cfg.DB_POOL_TIMEOUT,
                "pool_recycle": cfg.DB_POOL_RECYCLE,
                "pool_pre_ping": cfg.DB_POOL_PRE_PING,
                "echo": cfg.DB_ECHO,
            },
        )
        _engine = create_async_engine(
            cfg.DATABASE_URL.get_secret_value(),
            pool_size=cfg.DB_POOL_SIZE,
            max_overflow=cfg.DB_MAX_OVERFLOW,
            pool_timeout=cfg.DB_POOL_TIMEOUT,
            pool_recycle=cfg.DB_POOL_RECYCLE,
            pool_pre_ping=cfg.DB_POOL_PRE_PING,
            echo=cfg.DB_ECHO,
        )
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Return the singleton async_sessionmaker factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_db_engine(settings)
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            autoflush=False,
            expire_on_commit=False,
        )
    return _session_factory


async def close_db_engine() -> None:
    """Dispose of the global AsyncEngine connection pool on application shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        logger.info("Disposing PostgreSQL async engine connection pool")
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI request-scoped dependency providing an AsyncSession.

    Yields:
        AsyncSession: Active asynchronous database session.

    Session Lifecycle & Scoping:
        - A new AsyncSession is created per request.
        - AsyncSession instances are NOT thread-safe and must never be shared across
          concurrent requests or tasks.
        - Changes are not auto-committed; caller or service layer owns commit semantics.
        - If an unhandled exception occurs in the route, rollback() is explicitly invoked.
        - In the finally block, the session is cleanly closed and returned to the pool.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
