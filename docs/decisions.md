# Architecture Decision Records (ADRs)

## ADR-001: Python 3.12 Runtime and `uv` Package Manager

### Context
RAGForge requires reproducible dependency management, fast builds, and high performance for asynchronous I/O and vector search orchestration.

### Decision
Target Python 3.12 exclusively. Use `uv` by Astral as the primary virtual environment and dependency management engine, paired with standard `pyproject.toml` (PEP 621 / PEP 517).

### Consequences
- **Pros**: Sub-second package installations in CI/CD and Docker builds; native `StrEnum` and `datetime.UTC` adoption; cross-platform lockfile generation (`uv.lock`).
- **Cons**: Requires environments to have `uv` installed or fall back to standard `pip` using the generated `pyproject.toml`.

---

## ADR-002: FastAPI Application Framework & Lifespan Architecture

### Context
Modern microservices need high-throughput asynchronous execution, automatic OpenAPI contract generation, and native integration with Pydantic schemas.

### Decision
Adopt FastAPI >= 0.115.0 utilizing the modern `lifespan` context manager rather than deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` decorators.

### Consequences
- **Pros**: Clean startup and shutdown resource management within a standard asynchronous context manager (`@asynccontextmanager`); unified exception bubbling.
- **Cons**: Requires standard ASGI lifecycle compliance across testing fixtures.

---

## ADR-003: Pydantic v2 & Pydantic Settings for Configuration

### Context
Configuration errors in production should cause an immediate failure at startup (fail-fast), while preventing inadvertent exposure of credentials in logs.

### Decision
Use `pydantic-settings` with `BaseSettings` and typed fields. Mark passwords and secrets as `SecretStr`. Use `StrEnum` for environment names and log levels.

### Consequences
- **Pros**: Strong type validation with informative `ValidationError` traces; secrets are masked (`**********`) when models are converted to string representation; clean override capability via environment variables.
- **Cons**: Minor overhead of schema parsing, mitigated by caching `get_settings()` with `@lru_cache(maxsize=1)`.

---

## ADR-004: Strict Separation Between Liveness and Readiness

### Context
Container orchestrators (Kubernetes, Docker Swarm, Nomad) use probe endpoints to govern container lifecycles and traffic routing. Conflating liveness with readiness or checking downstream databases in liveness probes can cause catastrophic restart cascades.

### Decision
1. `GET /api/v1/health/live`: Tests strictly whether the Python ASGI worker is responsive and can process requests. It performs **no external I/O**.
2. `GET /api/v1/health/ready`: Assesses subsystem readiness for live user traffic. In Milestone 1, before database connectivity is fully implemented, the database subsystem status is explicitly reported as `not_configured` without failing the probe or falsely reporting that the DB is available.

### Consequences
- **Pros**: Orchestrators will not restart healthy workers when the database experiences brief transient blips.
- **Cons**: Full end-to-end readiness checks await Milestone 2 database connection pooling.

---

## ADR-005: Zero Secret Ingestion & Conservative CORS Policy

### Context
Web security demands defense-in-depth: preventing cross-origin request forgery (CSRF), leaking authorization credentials, or publishing debug stack traces.

### Decision
- Default `CORS_ORIGINS` to an empty list (same-origin only).
- Catch-all `Exception` handler suppresses internal traceback details in client responses.
- Structured logger sanitizes records to strip authorization and secret keys.
- Multi-stage Docker container runs under an unprivileged `ragforge` user (UID 10001).

### Consequences
- **Pros**: High baseline security posture meeting enterprise compliance standards.
- **Cons**: Developers connecting local frontends must explicitly define `RAGFORGE_CORS_ORIGINS=["http://localhost:3000"]` in their `.env`.

---

## ADR-006: Asynchronous Document Processing Pipeline & Deterministic Chunking

### Context
Document parsing and chunking are compute- and memory-intensive operations. Performing PDF parsing, text extraction, and semantic chunking inside the synchronous HTTP upload request introduces latency spikes, risks request timeouts, and starves the ASGI server event loop. Furthermore, retrieval systems require deterministic chunk boundaries, accurate source attribution, and atomic state handling.

### Decisions
1. **Asynchronous Background Processing with ARQ and Redis**:
   - The HTTP upload route strictly validates file signatures, streams payloads to secure storage, registers initial metadata in PostgreSQL, and enqueues a background job via Redis.
   - Heavy parsing (pypdf, python-docx, text decoding) and chunking occur in a dedicated ARQ background worker process.
2. **Standalone Deterministic Chunking Service**:
   - Implemented `TextChunker` as a standalone, deterministic service decoupled from FastAPI and SQLAlchemy.
   - Employs natural semantic split boundaries: paragraphs first (`\n\n`), then sentences (`. `, `? `, `! `), then words.
   - For PDFs, chunking preserves page boundaries so each chunk retains accurate source page numbers.
3. **Dual-Write Trade-Off**:
   - Committing document metadata in PostgreSQL and enqueuing jobs in Redis are distinct network operations without distributed 2PC (two-phase commit).
   - If Redis is temporarily unreachable after a database commit, the document remains safely in the `uploaded` state (rather than failing the upload or deleting stored files).
   - A dedicated `POST /api/v1/documents/{document_id}/requeue` endpoint allows manual or automated retries without re-uploading file bytes. A full transactional outbox pattern was deferred as an unnecessary complexity for this prototype milestone.
4. **Character-Based Chunking Heuristic**:
   - Chunk targets and overlaps are defined in characters (~1,000 target, ~150 overlap). Character counting is model-agnostic, deterministic, and avoids coupling document ingestion to a specific tokenizer before the embedding model is chosen in later milestones.
   - Estimated token counts (`(len + 3) // 4`) are tracked and clearly documented as heuristic estimates.
5. **Rich Chunk Metadata for Downstream Retrieval & Citations**:
   - Chunks persist zero-based sequential `chunk_index`, source `page_number`, `character_count`, and approximate tokens to power precise citation attribution and context assembly in later retrieval milestones.

### Consequences
- **Pros**: HTTP uploads return sub-second responses; worker failures cannot crash the API; idempotent job processing allows safe retries and reprocessing; clear chunk provenance.
- **Cons**: Redis service dependency required for processing; temporary dual-write risk if queue is offline (mitigated via requeue endpoint).
