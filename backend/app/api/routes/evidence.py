"""Evidence metadata and secure file download routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from backend.app.api.deps import AppContainer, get_container
from backend.app.schemas.evidence import (
    EvidenceListResponse,
    EvidencePaginatedResponse,
    EvidenceResponse,
)


router = APIRouter(tags=["evidence"])


@router.get("/evidence", response_model=EvidencePaginatedResponse)
def list_evidence(
    video_id: str | None = Query(default=None),
    job_id: str | None = Query(default=None),
    event_id: str | None = Query(default=None),
    evidence_type: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    container: AppContainer = Depends(get_container),
):
    return container.evidence_service.list_paginated(
        video_id=video_id,
        job_id=job_id,
        event_id=event_id,
        evidence_type=evidence_type,
        page=page,
        page_size=page_size,
    )


@router.get("/events/{event_id}/evidence", response_model=EvidenceListResponse)
def list_event_evidence(
    event_id: str,
    container: AppContainer = Depends(get_container),
):
    return container.evidence_service.list_for_event(event_id)


@router.get("/evidence/{evidence_id}", response_model=EvidenceResponse)
def get_evidence_metadata(
    evidence_id: str,
    container: AppContainer = Depends(get_container),
):
    return container.evidence_service.get_evidence(evidence_id)


@router.get("/evidence/{evidence_id}/download")
def download_evidence(
    evidence_id: str,
    container: AppContainer = Depends(get_container),
):
    item, file_path = container.evidence_service.resolve_evidence_file(evidence_id)
    return FileResponse(
        path=str(file_path),
        media_type=item.mime_type,
        filename=item.original_filename,
    )


@router.get("/events/{event_id}/evidence/{evidence_type}")
def get_event_evidence_by_type(
    event_id: str,
    evidence_type: str,
    container: AppContainer = Depends(get_container),
):
    item, file_path = container.evidence_service.resolve_event_evidence_by_type(
        event_id, evidence_type
    )
    return FileResponse(
        path=str(file_path),
        media_type=item.mime_type,
        filename=item.original_filename,
    )
