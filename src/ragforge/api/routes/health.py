"""Health and readiness probe endpoints."""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, status

from ragforge.core.config import Settings, get_settings
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
    summary="Application readiness check",
    description=(
        "Returns HTTP 200 when the service is ready to accept external user traffic. "
        "In Milestone 1, database connectivity verification is not yet active, so it is "
        "transparently marked as not_configured rather than falsely reported as available."
    ),
)
async def get_readiness(
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """Readiness probe for load balancers and orchestrators.

    Checks application subsystem dependencies.
    Per Milestone 1 constraints, database connectivity checks are not falsely asserted
    before database connectivity integration is implemented.
    """
    dependencies: dict[str, DependencyStatus] = {
        "database": DependencyStatus(
            status="not_configured",
            details="Database connectivity health check pending Milestone 2 integration",
        )
    }

    # If any required dependency is 'down', status would be 'not_ready'.
    # Currently all active dependencies are ready.
    overall_status = "ready"

    return ReadinessResponse(
        status=overall_status,
        timestamp=datetime.now(UTC),
        dependencies=dependencies,
    )
