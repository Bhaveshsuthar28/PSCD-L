"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import AppContainer, get_container
from backend.app.schemas.settings import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(container: AppContainer = Depends(get_container)):
    return container.system_service.health()


@router.get("/api/v1/health", response_model=HealthResponse, include_in_schema=False)
def health_v1_alias(container: AppContainer = Depends(get_container)):
    return container.system_service.health()
