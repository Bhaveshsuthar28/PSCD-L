"""Local storage management with path containment, atomic placement, and disk checks."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import uuid

from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    ForbiddenPathError,
    StorageExhaustedError,
)
from backend.app.core.logging import get_logger
from backend.app.core.security import resolve_within_root


logger = get_logger("storage")


class StorageService:
    """Manages uploads, job outputs, evidence, exports, and temporary files."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()

    @property
    def storage_root(self) -> Path:
        return self.settings.resolved_storage_root

    @property
    def upload_dir(self) -> Path:
        return self.settings.resolved_upload_dir

    @property
    def evidence_dir(self) -> Path:
        return self.settings.resolved_evidence_dir

    @property
    def export_dir(self) -> Path:
        return self.settings.resolved_export_dir

    @property
    def temp_dir(self) -> Path:
        return self.settings.resolved_temp_dir

    def check_writable(self) -> bool:
        try:
            self.settings.ensure_directories()
            probe = self.temp_dir / f".health_probe_{uuid.uuid4().hex}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except Exception as exc:
            logger.error("Storage health check failed: %s", exc, extra={"operation": "storage_health"})
            return False

    def get_disk_usage(self) -> dict[str, int]:
        self.settings.ensure_directories()
        usage = shutil.disk_usage(self.storage_root)
        return {
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
        }

    def ensure_free_space(self, required_bytes: int = 0) -> None:
        usage = self.get_disk_usage()
        free = usage["free_bytes"]
        threshold = self.settings.min_free_disk_bytes + max(0, required_bytes)
        if free < threshold:
            raise StorageExhaustedError(
                "Insufficient storage space available for upload/processing.",
                details={"free_bytes": free, "required_minimum_bytes": threshold},
            )

    def to_relative_path(self, path: Path) -> str:
        """Convert an absolute path inside storage_root into a POSIX relative storage key."""
        resolved = path.resolve()
        try:
            rel = resolved.relative_to(self.storage_root)
        except ValueError as exc:
            raise ForbiddenPathError("Path lies outside configured storage root.") from exc
        return rel.as_posix()

    def resolve_relative_path(self, relative_path: str) -> Path:
        """Safely resolve a stored relative path within storage_root."""
        return resolve_within_root(self.storage_root, relative_path)

    def resolve_upload_target(self, stored_filename: str) -> Path:
        return resolve_within_root(self.upload_dir, stored_filename)

    def create_job_output_dir(self, job_id: str) -> Path:
        job_dir = resolve_within_root(self.evidence_dir, f"job_{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def create_export_dir(self, job_id: str) -> Path:
        export_dir = resolve_within_root(self.export_dir, f"job_{job_id}")
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir

    def atomic_move(self, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        return destination

    def safe_unlink(self, path: Path) -> bool:
        try:
            if path.exists() and path.is_file():
                path.unlink()
            return True
        except OSError as exc:
            logger.error(
                "Failed to delete file: %s",
                exc,
                extra={"operation": "storage_delete", "error_code": "FILE_DELETE_FAILED"},
            )
            return False

    @staticmethod
    def compute_file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest().upper()
