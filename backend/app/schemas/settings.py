"""Pydantic schemas for system health, resources, model info, and settings."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str
    api: str
    database: str
    storage: str
    worker: str
    checkpoint_available: bool
    model_initialized: bool
    cuda_available: bool
    timestamp: datetime


class SystemStatusResponse(BaseModel):
    project: str
    version: str
    environment: str
    database_connected: bool
    storage_writable: bool
    worker_running: bool
    queued_jobs: int
    active_jobs: int
    checkpoint_available: bool
    model_initialized: bool
    cuda_available: bool
    scientific_scope: str


class SystemResourcesResponse(BaseModel):
    cpu_percent: float | None
    ram_total_bytes: int | None
    ram_available_bytes: int | None
    ram_used_percent: float | None
    disk_total_bytes: int
    disk_used_bytes: int
    disk_free_bytes: int
    cuda_available: bool
    gpu_name: str | None
    gpu_memory_allocated_bytes: int | None
    gpu_memory_reserved_bytes: int | None
    gpu_memory_total_bytes: int | None


class SystemModelResponse(BaseModel):
    architecture: str
    backbone: str
    feature_dimensions: list[int]
    output_logits_dimensions: list[int]
    checkpoint_exists: bool
    expected_sha256: str
    checkpoint_sha256_verified: bool | None
    model_initialized: bool
    device: str | None
    load_time_seconds: float | None
    last_error: str | None


class SystemSettingItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    value_json: Any
    description: str | None
    updated_at: datetime


class SystemSettingsResponse(BaseModel):
    items: list[SystemSettingItem]


class SystemSettingUpdateRequest(BaseModel):
    settings: dict[str, Any]
