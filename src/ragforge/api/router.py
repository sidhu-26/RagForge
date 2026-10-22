"""Root API v1 router combining all resource routes."""

from fastapi import APIRouter

from ragforge.api.routes import health

api_v1_router = APIRouter()

# Mount health probes under /api/v1/health
api_v1_router.include_router(health.router)
