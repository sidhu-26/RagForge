"""Main entrypoint for the RAGForge FastAPI application.

Sets up lifespan events, middleware, exception handling, and routing.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ragforge.api.router import api_v1_router
from ragforge.core.config import Settings, get_settings
from ragforge.core.logging import setup_logging
from ragforge.schemas.health import RootServiceInfo

logger = logging.getLogger("ragforge.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Executed once before accepting requests and once on shutdown.
    Replaces deprecated on_event('startup') and on_event('shutdown').
    """
    settings = get_settings()
    setup_logging(settings)
    logger.info(
        "Starting RAGForge backend",
        extra={"environment": settings.ENVIRONMENT.value, "version": settings.VERSION},
    )
    try:
        yield
    finally:
        logger.info("Shutting down RAGForge backend")
        from ragforge.db.session import close_db_engine

        await close_db_engine()


def create_application() -> FastAPI:
    """FastAPI application factory.

    Instantiates the application, registers middleware, global exception handlers,
    and mounts modular routers.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS: Strict by default. Only enable specified origins if provided.
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=["Authorization", "Content-Type", "Accept"],
        )

    # Global unhandled exception handler: prevents internal details/stack traces from leaking
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled server exception occurred",
            extra={"path": request.url.path, "method": request.method},
        )
        # Always return generic message without internal stack traces in public responses
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected server error occurred. Please contact the administrator.",
            },
        )

    # Root endpoint for service discovery and entrypoint information
    @app.get(
        "/",
        response_model=RootServiceInfo,
        status_code=status.HTTP_200_OK,
        tags=["System"],
        summary="Service metadata entrypoint",
    )
    async def root_entrypoint(
        app_settings: Settings = Depends(get_settings),
    ) -> RootServiceInfo:
        """Root API endpoint providing service metadata and discovery links."""
        return RootServiceInfo(
            name=app_settings.PROJECT_NAME,
            version=app_settings.VERSION,
            description=app_settings.DESCRIPTION,
            environment=app_settings.ENVIRONMENT.value,
            docs_url="/docs",
            redoc_url="/redoc",
            api_v1_prefix=app_settings.API_V1_PREFIX,
        )

    # Mount API v1 router
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_application()
