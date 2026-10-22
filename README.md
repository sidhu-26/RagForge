# RAGForge

Portfolio-grade Small Language Model (SLM)-powered Retrieval-Augmented Generation (RAG) platform built with FastAPI, SQLAlchemy 2.x async, Alembic, and PostgreSQL.

---

## Architecture & Technology Stack

- **Runtime**: Python 3.12 managed via `uv`
- **Web Framework**: FastAPI with modular APIRouter and lifespan management
- **Database ORM**: SQLAlchemy 2.x Async (`AsyncEngine`, `async_sessionmaker`, `AsyncSession`)
- **Database Driver**: `asyncpg`
- **Schema Migrations**: Alembic (asynchronous migration pipeline)
- **Configuration**: Pydantic Settings with strict validation and `SecretStr` security
- **Code Quality & Testing**: Ruff, Mypy (`strict = true`), Pytest with `pytest-asyncio`

---

## Database Architecture & Session Lifecycle

### 1. Connection Scheme & Driver
SQLAlchemy asynchronous PostgreSQL connections require the `postgresql+asyncpg://` scheme. Connection strings are validated via Pydantic validators in `Settings`.

### 2. Async Engine & Connection Pool
The application maintains a single, long-lived `AsyncEngine` initialized lazily via `get_db_engine()`. The engine is configured with:
- `DB_POOL_SIZE`: Base persistent connection pool size (default: 5)
- `DB_MAX_OVERFLOW`: Maximum temporary overflow connections during bursts (default: 10)
- `DB_POOL_TIMEOUT`: Maximum seconds to wait when acquiring a connection (default: 30.0s)
- `DB_POOL_RECYCLE`: Periodic recycling interval to drop stale pool connections (default: 1800s)
- `DB_POOL_PRE_PING`: Health check probe (`SELECT 1`) emitted prior to connection checkout to prevent dead connection usage (default: true)
- `DB_ECHO`: Statement logging (disabled by default)

The engine pool is disposed of cleanly during application shutdown within the lifespan context in `ragforge/main.py`.

### 3. Request-Scoped Session Dependency (`get_db`)
FastAPI routes inject database sessions using dependency injection:
```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ragforge.db.session import get_db


@router.get("/example")
async def example_route(db: AsyncSession = Depends(get_db)): ...
```

#### Why `AsyncSession` Must NOT Be Shared Concurrently:
`AsyncSession` encapsulates an active transaction and a single underlying connection borrowed from the pool. It is stateful, mutable, and **not thread-safe or task-safe**. Sharing a single session across concurrent requests or running multiple concurrent queries across tasks on the same session causes race conditions, corrupted transaction states, and protocol errors in `asyncpg`. 

`get_db` ensures strict 1-to-1 request scoping:
- Creates an isolated `AsyncSession` per HTTP request.
- Automatically rolls back uncommitted changes if an unhandled exception bubbles out of the route.
- Ensures the session is always closed and returned to the pool in a `finally` block.

---

## Developer Commands

### Environment Setup
```bash
# 1. Copy environment template
cp .env.example .env

# 2. Sync dependencies using uv
make install
# or: uv sync --all-extras
```

### Starting PostgreSQL
```bash
# Start PostgreSQL via Docker Compose
make db-up
# or: docker compose up -d postgres
```

### Alembic Migrations
```bash
# Apply pending migrations
make db-migrate
# or: uv run alembic upgrade head

# Autogenerate a new migration revision after model changes
make db-revision m="create_documents_table"
# or: uv run alembic revision --autogenerate -m "create_documents_table"

# Rollback the last applied migration
make db-downgrade
# or: uv run alembic downgrade -1

# Generate offline migration SQL without connecting to a database
uv run alembic upgrade --sql base:head
```

### Running the API & Checks
```bash
# Start local development server with hot reload
make dev
# or: uv run uvicorn ragforge.main:app --reload --host 0.0.0.0 --port 8000

# Run all quality checks (lint, format check, mypy, pytest)
make check

# Run individual checks
make lint       # uv run ruff check .
make format     # uv run ruff format .
make typecheck  # uv run mypy src tests
make test       # uv run pytest -v
```

---

## Health & Readiness Probes

- **Liveness Probe** (`GET /api/v1/health/live`):
  - Returns HTTP 200 `{"status": "alive"}`.
  - Verifies the ASGI process is active. Intentionally isolated from external dependencies to prevent cascade restarts in Kubernetes/container orchestrators.
- **Readiness Probe** (`GET /api/v1/health/ready`):
  - Injects `AsyncSession` and issues a `SELECT 1` connectivity probe.
  - Returns HTTP 200 `{"status": "ready"}` when PostgreSQL is responsive.
  - Returns HTTP 503 `{"status": "not_ready"}` when PostgreSQL is unreachable.
  - Securely logs diagnostic errors server-side while masking credentials, internal hostnames, and stack traces from the public API response.
