"""System status, resource usage, and model diagnostics routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import AppContainer, get_container
from backend.app.schemas.settings import (
    SystemModelResponse,
    SystemResourcesResponse,
    SystemStatusResponse,
)


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status", response_model=SystemStatusResponse)
def system_status(container: AppContainer = Depends(get_container)):
    return container.system_service.status()


@router.get("/resources", response_model=SystemResourcesResponse)
def system_resources(container: AppContainer = Depends(get_container)):
    return container.system_service.resources()


@router.get("/model", response_model=SystemModelResponse)
def system_model(container: AppContainer = Depends(get_container)):
    return container.system_service.model_info()
