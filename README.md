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

---

## Asynchronous Document Parsing, Chunking & Worker Pipeline (Milestone 4)

In Milestone 4, RAGForge integrates an asynchronous text processing pipeline powered by **Redis** and **ARQ**, featuring deterministic sentence/paragraph chunking and persistent chunks schema in PostgreSQL.

```mermaid
flowchart TD
    Client[Client / Web UI] -->|1. POST /api/v1/documents| API[FastAPI API Server]
    API -->|2. Stream Payload| Disk[Secure Storage data/storage]
    API -->|3. Register Metadata| DB[(PostgreSQL documents)]
    API -->|4. Enqueue Job| Redis[(Redis Queue)]
    API -->|5. Return 201 'queued'| Client
    
    Worker[ARQ Background Worker] -->|6. Pop Job| Redis
    Worker -->|7. Mark 'processing'| DB
    Worker -->|8. Read File Bytes| Disk
    Worker -->|9. Extract Text: PDF / DOCX / TXT| Extractor[DocumentExtractor]
    Worker -->|10. Semantic Chunking| Chunker[TextChunker]
    Worker -->|11. Persist Chunks & Mark 'processed'| DB
    
    Client -->|12. GET /api/v1/documents/{id}/chunks| API
    API -->|13. Read Paginated Chunks| DB
```

### 1. Starting PostgreSQL, Redis, API, and Worker

Start the components across separate terminal windows during local development:

```bash
# Terminal 1: Start PostgreSQL and Redis containers
make services-up
# or: docker compose up -d postgres redis

# Terminal 2: Run Alembic migrations
make db-migrate
# or: uv run alembic upgrade head

# Terminal 3: Start the FastAPI API server
make dev
# or: uv run uvicorn ragforge.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 4: Start the ARQ background worker
make worker
# or: uv run arq ragforge.worker.WorkerSettings
```

### 2. Upload, Requeue, and Chunk API Usage

#### Upload Document (PDF, DOCX, or TXT)
```bash
# Upload a PDF
curl -X POST "http://localhost:8000/api/v1/documents" \
  -F "file=@sample_report.pdf"

# Response immediately returns HTTP 201 with status="queued":
# {
#   "id": "e8632d16-b302-464b-b862-4c3f950ee131",
#   "filename": "sample_report.pdf",
#   "status": "queued",
#   ...
# }
```

#### Check Document Status
```bash
curl -X GET "http://localhost:8000/api/v1/documents/e8632d16-b302-464b-b862-4c3f950ee131"
# When worker finishes, status transitions to "processed":
# {
#   "id": "e8632d16-b302-464b-b862-4c3f950ee131",
#   "status": "processed",
#   "error_code": null,
#   ...
# }
```

#### Retrieve Persisted Chunks
```bash
curl -X GET "http://localhost:8000/api/v1/documents/e8632d16-b302-464b-b862-4c3f950ee131/chunks?page=1&page_size=20"
# Response:
# {
#   "document_id": "e8632d16-b302-464b-b862-4c3f950ee131",
#   "status": "processed",
#   "items": [
#     {
#       "id": "b53e8787-8d05-4c07-b3c9-04a4f8ad2b99",
#       "document_id": "e8632d16-b302-464b-b862-4c3f950ee131",
#       "chunk_index": 0,
#       "content": "First chunk text...",
#       "page_number": 1,
#       "character_count": 950,
#       "token_count_estimate": 238,
#       "created_at": "2026-10-22T12:00:00Z"
#     }
#   ],
#   "total": 12,
#   "page": 1,
#   "page_size": 20,
#   "total_pages": 1
# }
```

#### Requeue a Document
If Redis was unavailable during upload or if an earlier run failed:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/e8632d16-b302-464b-b862-4c3f950ee131/requeue"
# HTTP 202 Accepted: transitions status back to "queued"
```

### 3. Troubleshooting Queue & Processing Failures

1. **Redis unreachable (`status: uploaded` after upload)**:
   - Check if Redis container is active: `docker compose ps`
   - Start Redis: `make redis-up`
   - Trigger processing without re-uploading: `POST /api/v1/documents/{id}/requeue`
2. **Document status is `failed`**:
   - Inspect `error_code` and `error_message` on the document record (`GET /api/v1/documents/{id}`).
   - Common codes:
     - `DOCUMENT_EMPTY`: Document contained no text (e.g. image-only PDF or empty file).
     - `MALFORMED_DOCUMENT`: Corrupted zip structure in DOCX, unreadable PDF stream, or non-UTF-8 text.
     - `BOUNDS_EXCEEDED`: Exceeded `EXTRACTION_MAX_PAGES` (500) or `EXTRACTION_MAX_TEXT_SIZE_BYTES` (10 MB).
3. **Worker not picking up jobs**:
   - Verify worker is running: `make worker`
   - Verify Redis connection string in `.env` matches `RAGFORGE_REDIS_URL`.

### 3. Embeddings & Semantic Retrieval (Milestone 5)

RAGForge features an embedding generation and nearest-neighbor retrieval pipeline powered by PostgreSQL and the `pgvector` extension.

#### End-to-End Pipeline
```
Upload → Redis/ARQ worker → Extract → Chunk → Generate Embeddings → Persist Vectors → Semantic Search (pgvector)
```

#### Starting pgvector-Enabled PostgreSQL
RAGForge uses `pgvector/pgvector:pg16` in `docker-compose.yml`. Existing volumes are preserved without data loss.
```bash
docker compose up -d postgres redis
# Apply Alembic vector migration
uv run alembic upgrade head
```

#### Configuring Embedding Providers
RAGForge supports two embedding backends configured via `.env`:

1. **Local Mode (`EMBEDDING_PROVIDER=local`, Default)**:
   - Uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~80MB footprint).
   - Fully offline; requires **0 cloud credentials** and **0 external API calls**.
   - Lazy model loading ensures imports and HTTP startup remain instantaneous.
   - Resource requirements: ~500MB RAM, operates efficiently on standard multi-core CPUs and Apple Silicon (MPS).

2. **Azure OpenAI Mode (`EMBEDDING_PROVIDER=azure`)**:
   - Optional adapter for Azure OpenAI-hosted embeddings (e.g. `text-embedding-3-small`, 1536 dimensions).
   - Configure in `.env`:
     ```env
     RAGFORGE_EMBEDDING_PROVIDER=azure
     RAGFORGE_EMBEDDING_MODEL=text-embedding-3-small
     RAGFORGE_EMBEDDING_DIMENSION=1536
     RAGFORGE_AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
     RAGFORGE_AZURE_OPENAI_API_KEY=your-api-key
     RAGFORGE_AZURE_OPENAI_DEPLOYMENT_NAME=text-embedding-3-small
     RAGFORGE_AZURE_OPENAI_API_VERSION=2024-02-01
     ```
   - **Verification**: Verify that the selected deployment is provisioned and accessible under your Azure subscription before switching. RAGForge validates settings and fails fast if credentials are missing or invalid.

#### Semantic Search API (`POST /api/v1/search`)
Query indexed documents using pgvector cosine similarity search:

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does PostgreSQL connection pooling manage backend connections?",
    "limit": 5
  }'
```

Response:
```json
{
  "query": "How does PostgreSQL connection pooling manage backend connections?",
  "model_name": "all-MiniLM-L6-v2",
  "dimension": 384,
  "total_results": 2,
  "items": [
    {
      "chunk_id": "c138d4ef-1891-4475-b6d3-bc8efdca0784",
      "document_id": "e8632d16-b302-464b-b862-4c3f950ee131",
      "filename": "database_architecture.txt",
      "chunk_index": 0,
      "content": "PostgreSQL connection pooling with PgBouncer manages backend server connections...",
      "page_number": 1,
      "similarity_score": 0.8124,
      "cosine_distance": 0.1876
    }
  ]
}
```

Optional document filtering:
```json
{
  "query": "PostgreSQL connection pooling",
  "limit": 5,
  "document_id": "e8632d16-b302-464b-b862-4c3f950ee131"
}
```

#### Recovering Missing Embeddings & Reprocessing
If embeddings were not computed (e.g. provider downtime or transient failure), re-trigger the document pipeline using the requeue endpoint without re-uploading:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/{document_id}/requeue"
```
The worker will idempotently detect existing chunks and generate missing embeddings without creating duplicate records.

#### Running Retrieval Evaluation Benchmark
Measure Recall@5, latency, and result counts against a labeled benchmark dataset:
```bash
uv run python scripts/evaluate_retrieval.py
```

### 4. Grounded Question Answering with Citations (Milestone 6)

RAGForge features an SLM answer generation pipeline that retrieves relevant context chunks, constructs bounded evidence prompts, queries the configured language model, and validates all cited references against the actual retrieved sources.

#### End-to-End Question Answering Flow
```
User question
  → Validate request
  → Retrieve candidate chunks (RetrievalService / pgvector)
  → Quality & Abstention check (similarity threshold & evidence presence)
  → Assemble bounded prompt with [source_N] tokens
  → Invoke SLM (GenerationProvider: Fake or Azure OpenAI)
  → Validate and sanitize citations
  → Return grounded answer, citations, source attribution, and latency telemetry
```

#### Configuring Generation Providers
Configure the model provider in `.env`:

1. **Deterministic Mode (`GENERATION_PROVIDER=fake`, Default)**:
   - For fast, offline local development and unit/integration testing.
   - Requires **0 cloud credentials** and spends **0 cloud credits**.
   - Generates deterministic responses referencing retrieved `[source_N]` tokens.

2. **Azure OpenAI Mode (`GENERATION_PROVIDER=azure`)**:
   - Production adapter for Azure OpenAI Chat/Completions (e.g. `gpt-4o-mini`).
   - Configure in `.env`:
     ```env
     RAGFORGE_GENERATION_PROVIDER=azure
     RAGFORGE_GENERATION_MODEL=gpt-4o-mini
     RAGFORGE_AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
     RAGFORGE_AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
     RAGFORGE_AZURE_OPENAI_API_VERSION=2024-02-01
     RAGFORGE_AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
     ```
   - **Verification**: Ensure the deployment exists in your Azure subscription before switching. RAGForge fails fast if credentials are not configured.

#### Question Answering API (`POST /api/v1/ask`)
Ask a grounded question across all indexed documents or scope to a specific document:

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does connection pooling reduce database overhead in PostgreSQL?",
    "limit": 5
  }'
```

Response with verified citations and source metadata:
```json
{
  "question": "How does connection pooling reduce database overhead in PostgreSQL?",
  "answer": "Connection pools reuse active sockets to eliminate the high overhead of establishing new TCP sessions [source_1]. PgBouncer manages backend server connections efficiently in transaction pooling mode [source_2].",
  "abstained": false,
  "sources": [
    {
      "source_id": "[source_1]",
      "chunk_id": "c138d4ef-1891-4475-b6d3-bc8efdca0784",
      "document_id": "e8632d16-b302-464b-b862-4c3f950ee131",
      "filename": "db_connection_pooling.txt",
      "chunk_index": 1,
      "page_number": 1,
      "content": "Connection pools reuse active sockets to eliminate the high overhead of establishing new TCP sessions.",
      "similarity_score": 0.8124
    }
  ],
  "citations": [
    {
      "source_id": "[source_1]",
      "chunk_id": "c138d4ef-1891-4475-b6d3-bc8efdca0784",
      "filename": "db_connection_pooling.txt",
      "page_number": 1
    }
  ],
  "telemetry": {
    "provider": "fake",
    "model_name": "gpt-4o-mini",
    "retrieval_latency_ms": 11.74,
    "generation_latency_ms": 4.54,
    "retrieved_chunk_count": 3,
    "used_chunk_count": 3,
    "prompt_tokens": 185,
    "completion_tokens": 42,
    "total_tokens": 227
  }
}
```

#### Abstention Policies
- **No Evidence**: If no chunks are indexed or matching the filter, the system abstains immediately without invoking the model:
  `"The indexed documents did not provide sufficient evidence to answer the question."`
- **Low Relevance**: If the top retrieved chunk's similarity score is below `RAGFORGE_ASK_MIN_SIMILARITY_THRESHOLD` (default 0.40), the system abstains:
  `"The retrieved passages have insufficient relevance to reliably answer the question."`

#### Running Grounded QA Benchmark Evaluation
Run the question answering benchmark measuring grounded citation rate, abstention accuracy, and latencies:
```bash
uv run python scripts/evaluate_qa.py
```

### 5. Current Limitations & Milestone Boundaries (Milestone 6)
- **No Hybrid Search / Reranking**: Retrieval relies strictly on dense vector cosine similarity; BM25 keyword fusion and cross-encoder rerankers will follow in future milestones.
- **No Conversational Memory**: The `/api/v1/ask` endpoint is single-turn stateless question answering; multi-turn chat memory is deferred to conversational milestones.
- **No OCR**: Image-only PDFs and scanned files require optical character recognition.
- **No Public Authentication**: The API is private and intended for controlled development and testing environments.




