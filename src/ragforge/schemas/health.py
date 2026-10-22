"""Health and service status schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LivenessResponse(BaseModel):
    """Schema returned by GET /health/live."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["alive"] = Field(
        default="alive",
        description="Process liveness state. Indicates the worker is up and accepting events.",
    )
    timestamp: datetime = Field(
        description="UTC timestamp of the liveness check.",
    )


class DependencyStatus(BaseModel):
    """Health check status for an individual subsystem dependency."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["up", "down", "not_configured", "degraded"] = Field(
        description="Status of the dependency.",
    )
    details: str | None = Field(
        default=None,
        description="Additional context regarding dependency state.",
    )


class ReadinessResponse(BaseModel):
    """Schema returned by GET /health/ready.

    Reflects whether the application can safely handle user traffic.
    Readiness does not falsely report external dependencies before they are configured and verified.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "not_ready"] = Field(
        description="Overall readiness state.",
    )
    timestamp: datetime = Field(
        description="UTC timestamp of the readiness check.",
    )
    dependencies: dict[str, DependencyStatus] = Field(
        default_factory=dict,
        description="Map of dependency names to their current evaluated status.",
    )


class RootServiceInfo(BaseModel):
    """Schema returned by GET / (Root entrypoint)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Service name.")
    version: str = Field(description="Service semantic version.")
    description: str = Field(description="Brief description of the service.")
    environment: str = Field(description="Current running environment.")
    docs_url: str = Field(description="URL for OpenAPI Swagger interactive docs.")
    redoc_url: str = Field(description="URL for ReDoc documentation.")
    api_v1_prefix: str = Field(description="Base path for API version 1.")
