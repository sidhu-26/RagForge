# ==============================================================================
# Multi-stage Dockerfile for RAGForge FastAPI Backend
# Follows security best-practices:
# - Minimal attack surface with non-root user
# - Multi-stage build to isolate build tools from runtime image
# - No hardcoded secrets or environment credentials in layers
# ==============================================================================

# Stage 1: Build stage
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

# Install uv for fast, deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.12.20 /uv /bin/uv

# Copy dependency specifications
COPY pyproject.toml uv.lock README.md ./

# Create virtual environment and install production dependencies
RUN uv venv /opt/venv --python 3.12 && \
    uv pip install --no-cache -r pyproject.toml

# Stage 2: Production runtime stage
FROM python:3.12-slim-bookworm AS runtime

# Set security and runtime environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Create a non-root group and user with specific UID/GID
RUN groupadd -g 10001 ragforge && \
    useradd -u 10001 -g ragforge -s /bin/sh -M -d /app ragforge

# Copy installed virtual environment from builder stage
COPY --from=builder --chown=ragforge:ragforge /opt/venv /opt/venv

# Copy application source code
COPY --chown=ragforge:ragforge src/ /app/src/

# Switch to unprivileged runtime user
USER ragforge

# Document exposed HTTP port
EXPOSE 8000

# Run with production Uvicorn ASGI server
CMD ["uvicorn", "ragforge.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--proxy-headers", "--forwarded-allow-ips", "*"]
