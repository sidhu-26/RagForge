"""Root API v1 router combining all resource routes."""

from fastapi import APIRouter

from ragforge.api.routes import ask, documents, health, search

api_v1_router = APIRouter()

# Mount health probes under /api/v1/health
api_v1_router.include_router(health.router)

# Mount document management under /api/v1/documents
api_v1_router.include_router(documents.router)

# Mount semantic retrieval under /api/v1/search
api_v1_router.include_router(search.router)

# Mount question answering under /api/v1/ask
api_v1_router.include_router(ask.router)
