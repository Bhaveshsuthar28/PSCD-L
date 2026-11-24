"""Persisted non-secret system settings routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import AppContainer, get_container
from backend.app.schemas.settings import (
    SystemSettingsResponse,
    SystemSettingUpdateRequest,
)


router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SystemSettingsResponse)
def get_settings(container: AppContainer = Depends(get_container)):
    return {"items": container.system_service.list_settings()}


@router.put("", response_model=SystemSettingsResponse)
def update_settings(
    payload: SystemSettingUpdateRequest,
    container: AppContainer = Depends(get_container),
):
    return {"items": container.system_service.update_settings(payload.settings)}
