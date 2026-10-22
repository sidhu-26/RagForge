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

---

## ADR-007: Embedding Generation Architecture, pgvector Storage, and Semantic Retrieval

### Context
Milestone 5 introduces semantic retrieval over persisted document chunks. We need dense vector representations generated from text, persistent indexing with nearest-neighbor search, and strict model compatibility without blocking HTTP request handlers, coupling chunk storage to vector formats, or incurring cloud costs during local development and testing.

### Decisions
1. **Dedicated `chunk_embeddings` Table**:
   - Rather than embedding vectors directly as an array or vector column on `document_chunks`, embeddings are stored in a dedicated `chunk_embeddings` table with a foreign key to `document_chunks(id)`.
   - **Rationale**: Keeps the chunk content as the source of truth, allows re-embedding with different models or dimensions without mutating or re-parsing original chunk text, and enforces `UniqueConstraint("chunk_id", "model_name")`.
2. **PostgreSQL with `pgvector` Extension**:
   - Replaced base PostgreSQL image in `docker-compose.yml` with `pgvector/pgvector:pg16`.
   - Enabled `CREATE EXTENSION IF NOT EXISTS vector;` in an Alembic migration (`c98097ea7a6c`).
   - Retained the existing PostgreSQL data volume without destructive resets.
3. **Cosine Distance (`<=>`) Metric**:
   - Selected cosine distance as the primary nearest-neighbor metric for dense text embeddings:
     $\text{distance} = 1 - \text{cosine\_similarity}$.
   - Normalized embedding vectors allow ranking where distance = 0.0 represents exact semantic alignment, converted to `similarity_score = round(1.0 - distance, 5)`.
4. **Typed Provider Abstraction & Model Compatibility**:
   - Built `EmbeddingProvider` abstract base class with:
     - `LocalSentenceTransformerProvider` (`all-MiniLM-L6-v2`, 384 dimensions) using lazy loading and `asyncio.to_thread` for CPU/MPS inference.
     - `AzureOpenAIEmbeddingProvider` with bounded retries, fail-fast credential validation, and no automatic cloud resource provisioning.
     - `FakeEmbeddingProvider` for fast, deterministic unit test fixtures.
   - Enforced strict vector dimension checks: attempts to compare vectors across mismatched model dimensions or names fail explicitly with `EmbeddingDimensionMismatchError`.
5. **Worker Integration & Idempotency**:
   - Extended the ARQ background worker pipeline: chunks are embedded in bounded batches (`EMBEDDING_BATCH_SIZE=32`) immediately following extraction and chunking within the same job transaction.
   - Idempotency guarantees: existing embeddings for the target model are verified before generation, preventing duplicate rows or unnecessary model inference.
6. **Dedicated Semantic Retrieval Service (`POST /api/v1/search`)**:
   - Decoupled search logic into `RetrievalService` with query validation, query embedding, pgvector SQL generation, stable tie-breaking on `chunk_id`, and rich source attribution (`filename`, `chunk_index`, `page_number`).
   - Does not perform cross-database Python vector comparisons; all distance calculations execute natively in PostgreSQL.

### Consequences
- **Pros**: Fully functional offline local embeddings with 0 cloud cost; clean separation between source text and vector representations; fast SQL nearest-neighbor search; resilient background processing.
- **Cons**: Switching embedding models or dimensions requires running an embedding backfill across existing chunks.

---

## ADR-008: SLM-Powered Grounded Generation, Evidence Budgeting, and Citation Verification

### Context
Milestone 6 introduces question answering over retrieved document chunks. The system requires factual answer generation grounded strictly in retrieved passages, verifiable source citations, clear abstention policies when evidence is absent or irrelevant, and defense against prompt injection in untrusted user-uploaded documents, without cloud cost locks or blocking worker pipelines.

### Decisions
1. **Decoupled `GenerationProvider` Abstraction**:
   - Implemented abstract base class `GenerationProvider` with `generate(system_prompt, user_prompt, temperature, max_tokens) -> GenerationOutput`.
   - Built `FakeGenerationProvider` with deterministic test responses and source referencing.
   - Built `AzureOpenAIGenerationProvider` targeting Azure OpenAI Chat/Completions REST endpoint with exponential backoff on HTTP 429 and fail-fast configuration validation.
   - Zero automatic provisioning or cloud spending: Azure mode requires explicit user configuration via environment variables.
2. **Untrusted Evidence Isolation & Context Budgeting**:
   - Prompt construction is strictly modularized in `prompt_builder.py`.
   - Distinct system instructions emphasize factual adherence, source attribution, and abstention instructions.
   - Document chunks are explicitly demarcated as untrusted data (`--- [source_N] (Document: filename) ---`), and the model is commanded to ignore instructions, directives, or role-override attempts embedded inside document text.
   - Hard limits on question length (`ASK_MAX_QUESTION_CHARS=1000`) and concatenated evidence context (`ASK_MAX_EVIDENCE_TOTAL_CHARS=12000`) prevent context window exhaustion and prompt flooding.
3. **Application-Assigned Citation Tokens & Verification**:
   - Rather than asking the model to invent filenames or page numbers, the application assigns stable citation tokens (`[source_1]`, `[source_2]`, etc.) to each chunk in the prompt.
   - The citation validator (`citation_validator.py`) matches citation tokens against the active request's actual retrieved chunks. Hallucinated or nonexistent citation tokens are stripped from the response text, and only verified chunk citations are emitted.
4. **Configurable Evidence Quality & Abstention Policy**:
   - If retrieval yields zero chunks, the generation model is **not invoked**; the system immediately returns an abstention message (`ABSTENTION_MESSAGE_NO_EVIDENCE`).
   - If the top candidate chunk's similarity score is below `ASK_MIN_SIMILARITY_THRESHOLD` (default 0.40), the system abstains (`ABSTENTION_MESSAGE_LOW_SIMILARITY`), preventing hallucinations on out-of-domain questions.
5. **Request Path vs. Background Worker Separation**:
   - Kept question answering (`POST /api/v1/ask`) within the HTTP request path using async non-blocking I/O.
   - Did not place generation tasks inside the ARQ document worker. Document ingestion and query answer generation remain decoupled architectures.

### Consequences
- **Pros**: Clean separation between retrieval, prompt formatting, LLM invocation, and citation validation; complete immunity to cloud provider dependencies during test runs; grounded citations; protection against prompt injections in document content.
- **Cons**: Dense retrieval similarity scores are uncalibrated heuristics; threshold selection requires empirical tuning for specific document domains.


