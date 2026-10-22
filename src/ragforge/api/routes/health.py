"""Health and readiness probe endpoints."""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ragforge.core.config import Settings, get_settings
from ragforge.db.session import get_db
from ragforge.schemas.health import DependencyStatus, LivenessResponse, ReadinessResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health Probes"])


@router.get(
    "/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Process liveness check",
    description=(
        "Returns HTTP 200 if the process is up and able to respond to requests. "
        "Does not check external dependencies to avoid cascade restarts in orchestration systems."
    ),
)
async def get_liveness() -> LivenessResponse:
    """Liveness probe for Kubernetes / container health checks.

    This endpoint verifies only that the ASGI application worker is running and responsive.
    It intentionally does not evaluate external downstream services.
    """
    return LivenessResponse(
        status="alive",
        timestamp=datetime.now(UTC),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "Database or critical dependency is down.",
        }
    },
    summary="Application readiness check",
    description=(
        "Returns HTTP 200 when all core dependencies (e.g. PostgreSQL) are reachable and healthy. "
        "Returns HTTP 503 if any required dependency fails connectivity check."
    ),
)
async def get_readiness(
    response: Response,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """Readiness probe for load balancers and orchestrators.

    Executes a lightweight SELECT 1 check against PostgreSQL via the injected AsyncSession.
    Guarantees credentials or raw connection errors are not exposed in client responses.
    """
    dependencies: dict[str, DependencyStatus] = {}
    is_ready = True

    try:
        await db.execute(text("SELECT 1"))
        dependencies["database"] = DependencyStatus(
            status="up",
            details="Database connection healthy",
        )
    except Exception:
        # Log internal diagnostic details securely server-side without leaking in response
        logger.warning("Database readiness probe failed: unable to execute SELECT 1", exc_info=True)
        is_ready = False
        dependencies["database"] = DependencyStatus(
            status="down",
            details="Database connectivity unavailable",
        )

    overall_status = "ready" if is_ready else "not_ready"
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status=overall_status,
        timestamp=datetime.now(UTC),
        dependencies=dependencies,
    )
