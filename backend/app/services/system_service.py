"""System health, hardware resource inspection, model status, and settings service."""

from __future__ import annotations

from typing import Any

import psutil
import torch

from backend.app.core.config import Settings
from backend.app.core.exceptions import ValidationError
from backend.app.database.connection import DatabaseManager
from backend.app.database.repositories import JobRepository, SettingsRepository
from backend.app.integrations.trinetra_engine import ManagedInferenceEngine
from backend.app.services.storage_service import StorageService
from backend.app.utils.serialization import utc_now
from backend.app.workers.job_manager import JobManager


FORBIDDEN_SETTING_PREFIXES = ("secret", "password", "token", "api_key", "checkpoint")


class SystemService:
    def __init__(
        self,
        settings: Settings,
        db_manager: DatabaseManager,
        storage: StorageService,
        managed_engine: ManagedInferenceEngine,
        job_manager: JobManager,
    ) -> None:
        self.settings = settings
        self.db_manager = db_manager
        self.storage = storage
        self.managed_engine = managed_engine
        self.job_manager = job_manager

    def health(self) -> dict[str, Any]:
        db_ok = self.db_manager.check_connectivity()
        storage_ok = self.storage.check_writable()
        worker_ok = self.job_manager.is_running()
        checkpoint_ok = self.managed_engine.checkpoint_exists()
        model_init = self.managed_engine.is_initialized()
        cuda_ok = torch.cuda.is_available()

        overall = "healthy" if (db_ok and storage_ok and worker_ok) else "degraded"
        return {
            "status": overall,
            "api": "up",
            "database": "connected" if db_ok else "unavailable",
            "storage": "writable" if storage_ok else "unavailable",
            "worker": "running" if worker_ok else "stopped",
            "checkpoint_available": checkpoint_ok,
            "model_initialized": model_init,
            "cuda_available": cuda_ok,
            "timestamp": utc_now(),
        }

    def status(self) -> dict[str, Any]:
        db_ok = self.db_manager.check_connectivity()
        queued = 0
        active = 0
        if db_ok:
            with self.db_manager.session() as db:
                repo = JobRepository(db)
                _, queued = repo.list_paginated(offset=0, limit=1, status="QUEUED")
                active = repo.count_active()

        return {
            "project": "TRINETRA PSCD&L",
            "version": self.settings.api_version,
            "environment": self.settings.app_env,
            "database_connected": db_ok,
            "storage_writable": self.storage.check_writable(),
            "worker_running": self.job_manager.is_running(),
            "queued_jobs": queued,
            "active_jobs": active,
            "checkpoint_available": self.managed_engine.checkpoint_exists(),
            "model_initialized": self.managed_engine.is_initialized(),
            "cuda_available": torch.cuda.is_available(),
            "scientific_scope": (
                "TRINETRA detects persistent visual scene changes. It does not "
                "independently establish object identity, ownership, abandonment, or intent."
            ),
        }

    def resources(self) -> dict[str, Any]:
        cpu_pct: float | None = None
        ram_total: int | None = None
        ram_avail: int | None = None
        ram_pct: float | None = None
        try:
            cpu_pct = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            ram_total = int(vm.total)
            ram_avail = int(vm.available)
            ram_pct = float(vm.percent)
        except Exception:
            pass

        disk = self.storage.get_disk_usage()
        cuda_ok = torch.cuda.is_available()
        gpu_name: str | None = None
        gpu_alloc: int | None = None
        gpu_res: int | None = None
        gpu_total: int | None = None
        if cuda_ok:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_alloc = int(torch.cuda.memory_allocated(0))
                gpu_res = int(torch.cuda.memory_reserved(0))
                props = torch.cuda.get_device_properties(0)
                gpu_total = int(props.total_memory)
            except Exception:
                pass

        return {
            "cpu_percent": cpu_pct,
            "ram_total_bytes": ram_total,
            "ram_available_bytes": ram_avail,
            "ram_used_percent": ram_pct,
            "disk_total_bytes": disk["total_bytes"],
            "disk_used_bytes": disk["used_bytes"],
            "disk_free_bytes": disk["free_bytes"],
            "cuda_available": cuda_ok,
            "gpu_name": gpu_name,
            "gpu_memory_allocated_bytes": gpu_alloc,
            "gpu_memory_reserved_bytes": gpu_res,
            "gpu_memory_total_bytes": gpu_total,
        }

    def model_info(self) -> dict[str, Any]:
        return self.managed_engine.get_model_status()

    def list_settings(self) -> list:
        with self.db_manager.session() as db:
            return SettingsRepository(db).list_all()

    def update_settings(self, updates: dict[str, Any]) -> list:
        if not updates:
            raise ValidationError("No settings provided for update.")
        for key in updates:
            lowered = str(key).lower()
            if any(bad in lowered for bad in FORBIDDEN_SETTING_PREFIXES):
                raise ValidationError(
                    f"Setting key '{key}' cannot be modified or stored in system settings.",
                    error_code="FORBIDDEN_SETTING_KEY",
                )
        with self.db_manager.session() as db:
            repo = SettingsRepository(db)
            for key, value in updates.items():
                repo.upsert(str(key), value)
            return repo.list_all()
