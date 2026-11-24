"""FastAPI dependency accessors bound to application state."""

from __future__ import annotations

from dataclasses import dataclass
from fastapi import Request

from backend.app.core.config import Settings
from backend.app.database.connection import DatabaseManager
from backend.app.integrations.trinetra_engine import ManagedInferenceEngine
from backend.app.services.analytics_service import AnalyticsService
from backend.app.services.event_service import EventService
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.export_service import ExportService
from backend.app.services.processing_service import ProcessingService
from backend.app.services.storage_service import StorageService
from backend.app.services.system_service import SystemService
from backend.app.services.video_service import VideoService
from backend.app.workers.job_manager import JobManager
from backend.app.workers.progress_manager import ProgressManager


@dataclass
class AppContainer:
    settings: Settings
    db_manager: DatabaseManager
    storage: StorageService
    managed_engine: ManagedInferenceEngine
    progress_manager: ProgressManager
    job_manager: JobManager
    video_service: VideoService
    processing_service: ProcessingService
    event_service: EventService
    evidence_service: EvidenceService
    analytics_service: AnalyticsService
    export_service: ExportService
    system_service: SystemService


def get_container(request: Request) -> AppContainer:
    return request.app.state.container
