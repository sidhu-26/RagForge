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

---

## Document Management & Storage (Milestone 3)

RAGForge features an abstracted file storage layer, streaming document upload pipelines, content signature validation, and PostgreSQL metadata tracking.

### 1. Document Lifecycle & Status Model
Documents follow an explicit finite state machine:
- `uploaded`: File payload stored on disk, checksum verified, metadata saved in PostgreSQL. (Does **not** mean parsed or indexed).
- `queued`: Enqueued for background processing (chunking/embedding).
- `processing`: Worker is actively parsing or generating embeddings.
- `processed`: Chunks and embeddings are indexed and queryable in vector storage.
- `failed`: Document processing failed; `error_code` and `error_message` store sanitized diagnostic details.

Database constraints (`ck_documents_valid_status` and `ck_documents_positive_file_size`) guarantee integrity.

### 2. Architecture & Technical Concepts

#### FastAPI `UploadFile` & Multipart Streaming
FastAPI's `UploadFile` wraps Starlette's `SpooledTemporaryFile`. Rather than reading entire uploaded documents into server memory (`await file.read()`), RAGForge streams data in bounded chunks (configurable via `STORAGE_CHUNK_SIZE_BYTES`, default 64 KB). This prevents Denial-of-Service (DoS) and Out-Of-Memory (OOM) failures under heavy concurrent uploads. The SHA-256 hex digest is calculated incrementally on the fly as chunks stream into storage.

#### Dependency Injection for Repositories & Storage
The `StorageEngine` abstract base class decouples business logic from physical storage providers:
```python
class StorageEngine(abc.ABC):
    async def save(self, storage_key: str, stream: AsyncIterator[bytes]) -> int: ...
    async def open_stream(self, storage_key: str, chunk_size: int) -> AsyncIterator[bytes]: ...
    async def read_bytes(self, storage_key: str) -> bytes: ...
    async def delete(self, storage_key: str) -> bool: ...
    async def exists(self, storage_key: str) -> bool: ...
    async def get_metadata(self, storage_key: str) -> StorageMetadata: ...
```
Using FastAPI dependencies (`get_storage_engine`), `LocalFilesystemStorage` can be replaced with `AzureBlobStorage` or `S3Storage` via configuration without altering router or business logic.

#### SQLAlchemy Transaction Boundaries & Dual-Write Consistency
Writing to physical storage and inserting database records creates a dual-write consistency challenge:
1. **File Stored First**: The file is streamed to disk under a server-generated unique path (`docs/YYYY/MM/DD/<uuid>.<ext>`).
2. **Metadata Persisted**: A PostgreSQL transaction inserts the document metadata record.
3. **Rollback & Compensation**: If the database transaction fails (constraint violation, connectivity drop, etc.), an exception handler immediately compensates by removing the newly stored file (`await storage.delete(...)`), preventing orphaned files in storage.
4. **Deletion Sequence**: When deleting a document, the database record is deleted first within an atomic transaction. Upon commit, the stored file payload is removed.

#### Why File Parsing is Separated from HTTP Request Handling
Parsing complex documents (extracting PDF text, OCR, parsing XML/DOCX hierarchies) is CPU-bound and unpredictable in duration. Performing parsing inside the synchronous HTTP request lifecycle ties up ASGI worker threads, starves connection pools, and exposes the API to client timeouts. In RAGForge:
- The HTTP endpoint only handles **secure receipt, validation, storage, and persistence**.
- Document parsing and embedding generation are executed asynchronously by background workers (Milestone 4).

### 3. Security Considerations
- **Storage Keys**: Files are stored using UUID-based keys outside publicly accessible directories (`STORAGE_LOCAL_ROOT`). Client filenames are never used as filesystem paths.
- **Path Traversal Protection**: Keys and filenames are validated with strict containment checks against the storage root directory.
- **Content Inspection**: Formats (PDF, DOCX, TXT) are verified via magic byte signatures and OOXML archive validation, never trusting the client `Content-Type` header alone.
- **Malware Notice**: *MIME and content-signature validation alone are not a complete malware defense.* Production deployments should incorporate dedicated antivirus/antimalware scanning (e.g. ClamAV or cloud threat protection).

### 4. API Usage & Curl Examples

#### Upload a Document (`POST /api/v1/documents`)
```bash
# Upload a PDF document
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Accept: application/json" \
  -F "file=@sample_report.pdf"

# Response (HTTP 201 Created):
# {
#   "id": "123e4567-e89b-12d3-a456-426614174000",
#   "filename": "sample_report.pdf",
#   "storage_key": "docs/2026/10/22/123e4567e89b12d3a456426614174000.pdf",
#   "mime_type": "application/pdf",
#   "file_size_bytes": 1048576,
#   "sha256_checksum": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
#   "status": "uploaded",
#   "error_code": null,
#   "error_message": null,
#   "created_at": "2026-10-22T10:30:00Z",
#   "updated_at": "2026-10-22T10:30:00Z"
# }
```

#### List Documents (`GET /api/v1/documents`)
```bash
curl -X GET "http://localhost:8000/api/v1/documents?page=1&page_size=20" \
  -H "Accept: application/json"
```

#### Retrieve Document Metadata (`GET /api/v1/documents/{document_id}`)
```bash
curl -X GET "http://localhost:8000/api/v1/documents/123e4567-e89b-12d3-a456-426614174000" \
  -H "Accept: application/json"
```

#### Delete Document (`DELETE /api/v1/documents/{document_id}`)
```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/123e4567-e89b-12d3-a456-426614174000" \
  -H "Accept: application/json"
```

