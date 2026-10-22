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
