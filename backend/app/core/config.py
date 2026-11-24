"""Validated application configuration for the TRINETRA backend."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_ROOT = BACKEND_ROOT / "storage"
DEFAULT_LOG_DIR = BACKEND_ROOT / "logs"
DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "outputs" / "training" / "checkpoints" / "trinetra-33-0.1482.ckpt"
)
EXPECTED_FROZEN_CHECKPOINT_SHA256 = (
    "790BC43661096E8F45B513E774C1BE1D"
    "6734D44093D3FD5AA5063A8DA6C90BF9"
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_prefix="TRINETRA_",
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application metadata
    app_name: str = "TRINETRA PSCD&L Backend"
    app_env: Literal["development", "testing", "staging", "production"] = "development"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    # Server
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # CORS
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # Database
    database_url: str = Field(
        default=f"sqlite:///{(DEFAULT_STORAGE_ROOT / 'trinetra.db').as_posix()}"
    )

    # Storage paths
    storage_root: Path = DEFAULT_STORAGE_ROOT
    upload_dir: Path | None = None
    evidence_dir: Path | None = None
    export_dir: Path | None = None
    temp_dir: Path | None = None
    log_dir: Path = DEFAULT_LOG_DIR

    # Upload & video validation limits
    max_upload_size_bytes: int = Field(default=500 * 1024 * 1024, ge=1024)
    allowed_video_extensions: list[str] = Field(
        default_factory=lambda: [".avi", ".mp4", ".mov", ".mkv"]
    )
    max_video_duration_seconds: float = Field(default=3600.0, gt=0.0)
    max_video_width: int = Field(default=3840, ge=16)
    max_video_height: int = Field(default=2160, ge=16)
    min_free_disk_bytes: int = Field(default=256 * 1024 * 1024, ge=0)
    upload_chunk_size_bytes: int = Field(default=1024 * 1024, ge=4096)

    # Processing & worker limits
    worker_count: int = Field(default=1, ge=1, le=4)
    gpu_job_concurrency: int = Field(default=1, ge=1, le=2)
    max_queue_size: int = Field(default=20, ge=1, le=500)
    min_sampling_interval_seconds: float = Field(default=0.1, gt=0.0)
    max_sampling_interval_seconds: float = Field(default=60.0, gt=0.0)
    progress_db_persist_interval_seconds: float = Field(default=1.0, ge=0.1)
    progress_db_persist_frame_batch: int = Field(default=3, ge=1)
    progress_history_limit: int = Field(default=200, ge=10, le=5000)
    sse_heartbeat_seconds: float = Field(default=5.0, ge=0.5, le=60.0)

    # Frozen AI model configuration (controlled exclusively by backend config)
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    expected_checkpoint_sha256: str = EXPECTED_FROZEN_CHECKPOINT_SHA256
    inference_device: str | None = None  # None -> auto-detect CUDA/CPU
    keep_model_loaded_across_jobs: bool = True

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_max_bytes: int = Field(default=10 * 1024 * 1024, ge=10240)
    log_backup_count: int = Field(default=5, ge=1, le=50)

    # Retention policy (in days; None means never auto-delete)
    retain_temp_hours: int = Field(default=24, ge=1)
    auto_delete_evidence: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value):
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in text.split(",") if item.strip()]
        return value

    @field_validator("allowed_video_extensions", mode="before")
    @classmethod
    def _parse_extensions(cls, value):
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                items = json.loads(text)
            else:
                items = text.split(",")
        else:
            items = list(value)
        normalized = []
        for item in items:
            ext = str(item).strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            normalized.append(ext)
        if not normalized:
            raise ValueError("allowed_video_extensions cannot be empty")
        return sorted(set(normalized))

    @field_validator("expected_checkpoint_sha256")
    @classmethod
    def _normalize_sha256(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if len(cleaned) != 64 or any(c not in "0123456789ABCDEF" for c in cleaned):
            raise ValueError("expected_checkpoint_sha256 must be a 64-char hex string")
        return cleaned

    @property
    def resolved_storage_root(self) -> Path:
        return self.storage_root.resolve()

    @property
    def resolved_upload_dir(self) -> Path:
        return (self.upload_dir or (self.storage_root / "uploads")).resolve()

    @property
    def resolved_evidence_dir(self) -> Path:
        return (self.evidence_dir or (self.storage_root / "evidence")).resolve()

    @property
    def resolved_export_dir(self) -> Path:
        return (self.export_dir or (self.storage_root / "exports")).resolve()

    @property
    def resolved_temp_dir(self) -> Path:
        return (self.temp_dir or (self.storage_root / "temporary")).resolve()

    @property
    def resolved_log_dir(self) -> Path:
        return self.log_dir.resolve()

    @property
    def resolved_checkpoint_path(self) -> Path:
        return self.checkpoint_path.resolve()

    def ensure_directories(self) -> None:
        """Create required runtime directories safely."""
        for directory in (
            self.resolved_storage_root,
            self.resolved_upload_dir,
            self.resolved_evidence_dir,
            self.resolved_export_dir,
            self.resolved_temp_dir,
            self.resolved_log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
