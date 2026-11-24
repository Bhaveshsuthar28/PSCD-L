"""TRINETRA PSCD&L FastAPI application entrypoint and factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from backend.app.api.deps import AppContainer
from backend.app.api.routes import (
    analytics,
    events,
    evidence,
    health,
    processing,
    settings as settings_routes,
    system,
    videos,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import TrinetraBackendError
from backend.app.core.logging import configure_logging, get_logger, request_id_ctx
from backend.app.database.connection import DatabaseManager
from backend.app.integrations.trinetra_engine import ManagedInferenceEngine
from backend.app.schemas.common import ErrorDetail, ErrorResponse
from backend.app.services.analytics_service import AnalyticsService
from backend.app.services.event_service import EventService
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.export_service import ExportService
from backend.app.services.processing_service import ProcessingService
from backend.app.services.storage_service import StorageService
from backend.app.services.system_service import SystemService
from backend.app.services.video_service import VideoService
from backend.app.utils.serialization import sanitize_for_json
from backend.app.workers.job_manager import JobManager
from backend.app.workers.progress_manager import ProgressManager


def create_app(
    settings: Settings | None = None,
    *,
    engine_factory: Callable[[Path, str | None], Any] | None = None,
    start_workers: bool = True,
) -> FastAPI:
    """Construct and configure the integrated TRINETRA FastAPI application."""
    cfg = settings or get_settings()
    cfg.ensure_directories()
    configure_logging(
        cfg.resolved_log_dir,
        level=cfg.log_level,
        max_bytes=cfg.log_max_bytes,
        backup_count=cfg.log_backup_count,
    )
    logger = get_logger("app")

    db_manager = DatabaseManager(cfg.database_url)
    storage = StorageService(cfg)
    managed_engine = ManagedInferenceEngine(cfg, engine_factory=engine_factory)
    progress_manager = ProgressManager(history_limit=cfg.progress_history_limit)
    job_manager = JobManager(
        settings=cfg,
        db_manager=db_manager,
        storage=storage,
        managed_engine=managed_engine,
        progress_manager=progress_manager,
    )

    container = AppContainer(
        settings=cfg,
        db_manager=db_manager,
        storage=storage,
        managed_engine=managed_engine,
        progress_manager=progress_manager,
        job_manager=job_manager,
        video_service=VideoService(cfg, db_manager, storage),
        processing_service=ProcessingService(cfg, db_manager, storage, job_manager),
        event_service=EventService(db_manager, api_prefix=cfg.api_prefix),
        evidence_service=EvidenceService(db_manager, storage, api_prefix=cfg.api_prefix),
        analytics_service=AnalyticsService(db_manager),
        export_service=ExportService(db_manager, storage, api_prefix=cfg.api_prefix),
        system_service=SystemService(
            cfg, db_manager, storage, managed_engine, job_manager
        ),
    )

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        logger.info("Starting TRINETRA backend (%s)", cfg.app_env, extra={"operation": "app_startup"})
        db_manager.initialize()
        if start_workers:
            job_manager.start()
        try:
            yield
        finally:
            logger.info("Shutting down TRINETRA backend", extra={"operation": "app_shutdown"})
            if start_workers:
                job_manager.shutdown()
            db_manager.dispose()

    application = FastAPI(
        title=cfg.app_name,
        description=(
            "Persistent Scene Change Detection and Localization (PSCD&L) backend API. "
            "Detects persistent visual scene changes, tracks candidate regions, evaluates "
            "temporal persistence criteria, and manages auditable event evidence."
        ),
        version=cfg.api_version,
        lifespan=lifespan,
    )
    application.state.container = container

    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        token = request_id_ctx.set(req_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            logger.info(
                "%s %s -> %s",
                request.method,
                request.url.path,
                response.status_code,
                extra={"operation": "http_request"},
            )
            return response
        finally:
            request_id_ctx.reset(token)

    def _error_response(
        request: Request,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        body = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code=code,
                message=message,
                details=sanitize_for_json(details),
                request_id=req_id,
            ),
        )
        return JSONResponse(status_code=status_code, content=body.model_dump())

    @application.exception_handler(TrinetraBackendError)
    async def backend_error_handler(request: Request, exc: TrinetraBackendError):
        return _error_response(
            request,
            exc.status_code,
            exc.error_code,
            exc.message,
            exc.details,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return _error_response(
            request,
            422,
            "REQUEST_VALIDATION_ERROR",
            "Request validation failed.",
            exc.errors(),
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        return _error_response(
            request,
            exc.status_code,
            f"HTTP_{exc.status_code}",
            str(exc.detail),
        )

    @application.exception_handler(SQLAlchemyError)
    async def db_error_handler(request: Request, exc: SQLAlchemyError):
        logger.exception(
            "Database error during request: %s",
            exc,
            extra={"operation": "http_request", "error_code": "DATABASE_ERROR"},
        )
        return _error_response(
            request,
            500,
            "DATABASE_ERROR",
            "A database error occurred while processing the request.",
        )

    @application.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception(
            "Unhandled server error: %s",
            exc,
            extra={"operation": "http_request", "error_code": "INTERNAL_SERVER_ERROR"},
        )
        return _error_response(
            request,
            500,
            "INTERNAL_SERVER_ERROR",
            "An unexpected internal server error occurred.",
        )

    @application.get("/", tags=["root"])
    def root():
        return {
            "project": "TRINETRA PSCD&L",
            "version": cfg.api_version,
            "status": "running",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
        }

    application.include_router(health.router)
    application.include_router(videos.router, prefix=cfg.api_prefix)
    application.include_router(processing.router, prefix=cfg.api_prefix)
    application.include_router(events.router, prefix=cfg.api_prefix)
    application.include_router(evidence.router, prefix=cfg.api_prefix)
    application.include_router(analytics.router, prefix=cfg.api_prefix)
    application.include_router(settings_routes.router, prefix=cfg.api_prefix)
    application.include_router(system.router, prefix=cfg.api_prefix)

    return application


app = create_app()
