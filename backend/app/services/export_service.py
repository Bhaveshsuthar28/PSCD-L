"""Atomic JSON, CSV, and traversal-safe ZIP export generation service."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from backend.app.core.exceptions import ConflictError, ResourceNotFoundError, ValidationError
from backend.app.core.security import validate_uuid
from backend.app.database.connection import DatabaseManager
from backend.app.database.repositories import EventRepository, EvidenceRepository, JobRepository
from backend.app.services.storage_service import StorageService
from backend.app.utils.serialization import ensure_utc_iso, sanitize_for_json
from src.inference.event_store import CSV_FIELDS


EXPORT_FORMATS = {"json", "csv", "zip"}


class ExportService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        storage: StorageService,
        api_prefix: str = "/api/v1",
    ) -> None:
        self.db_manager = db_manager
        self.storage = storage
        self.api_prefix = api_prefix

    @staticmethod
    def _atomic_write_text(destination: Path, writer_fn) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=destination.parent,
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                writer_fn(tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, destination)
        except Exception:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_write_zip(destination: Path, writer_fn) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
            with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                writer_fn(zf)
            os.replace(tmp_path, destination)
        except Exception:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _safe_arcname(raw_name: str) -> str:
        posix = PurePosixPath(raw_name)
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"Unsafe archive entry path: {raw_name}")
        return posix.as_posix()

    def generate_export(
        self,
        job_id: str,
        fmt: str,
        *,
        overwrite: bool = False,
    ) -> tuple[Path, str, str]:
        canonical_id = validate_uuid(job_id, "job_id")
        normalized_fmt = (fmt or "").strip().lower()
        if normalized_fmt not in EXPORT_FORMATS:
            raise ValidationError(
                f"Unsupported export format '{fmt}'. Allowed: {sorted(EXPORT_FORMATS)}",
                error_code="UNSUPPORTED_EXPORT_FORMAT",
            )

        with self.db_manager.session() as db:
            job = JobRepository(db).get(canonical_id)
            if job is None:
                raise ResourceNotFoundError(
                    f"Processing job '{canonical_id}' not found.",
                    error_code="JOB_NOT_FOUND",
                )
            if job.status in {"QUEUED", "VALIDATING", "INITIALIZING", "PROCESSING", "FINALIZING"}:
                raise ConflictError(
                    "Export is available only after processing reaches a terminal state.",
                    error_code="JOB_NOT_FINISHED",
                )

            events = EventRepository(db).list_all_for_job(canonical_id)
            evidence_items = EvidenceRepository(db).list_for_job(canonical_id)

            export_dir = self.storage.create_export_dir(canonical_id)
            filename = f"trinetra_job_{canonical_id}.{normalized_fmt}"
            target_path = export_dir / filename

            if target_path.exists() and not overwrite:
                mime = self._mime_for_fmt(normalized_fmt)
                return target_path, filename, mime

            json_payload = self._build_json_payload(job, events, evidence_items)

            if normalized_fmt == "json":
                self._write_json_file(target_path, json_payload)
            elif normalized_fmt == "csv":
                self._write_csv_file(target_path, events, evidence_items)
            else:
                self._write_zip_file(
                    target_path,
                    job,
                    events,
                    evidence_items,
                    json_payload,
                )

            return target_path, filename, self._mime_for_fmt(normalized_fmt)

    @staticmethod
    def _mime_for_fmt(fmt: str) -> str:
        return {
            "json": "application/json",
            "csv": "text/csv; charset=utf-8",
            "zip": "application/zip",
        }[fmt]

    def _build_json_payload(self, job, events, evidence_items) -> dict:
        ev_by_event: dict[str, list[dict]] = {}
        for item in evidence_items:
            ref = {
                "evidence_id": item.id,
                "evidence_type": item.evidence_type,
                "filename": item.original_filename,
                "mime_type": item.mime_type,
                "file_size_bytes": item.file_size_bytes,
                "sha256": item.sha256,
                "download_url": f"{self.api_prefix}/evidence/{item.id}/download",
            }
            key = item.event_id or "__job__"
            ev_by_event.setdefault(key, []).append(ref)

        serialized_events = []
        for ev in events:
            serialized_events.append(
                {
                    "id": ev.id,
                    "job_id": ev.job_id,
                    "video_id": ev.video_id,
                    "track_id": ev.track_id,
                    "state": ev.event_state,
                    "first_seen_seconds": ev.first_seen_seconds,
                    "last_seen_seconds": ev.last_seen_seconds,
                    "duration_seconds": ev.duration_seconds,
                    "observations": ev.observations,
                    "missed_observations": ev.missed_observations,
                    "total_missed_observations": ev.total_missed_observations,
                    "detection_consistency": ev.detection_consistency,
                    "center_deviation_pixels": ev.center_deviation_pixels,
                    "bbox_stability": ev.bbox_stability,
                    "motion_samples": ev.motion_samples,
                    "motion_median_ratio": ev.motion_median_ratio,
                    "motion_p90_ratio": ev.motion_p90_ratio,
                    "settling_motion_samples": ev.settling_motion_samples,
                    "settling_motion_median_ratio": ev.settling_motion_median_ratio,
                    "settling_motion_p90_ratio": ev.settling_motion_p90_ratio,
                    "confidence": ev.confidence,
                    "confidence_kind": ev.confidence_type,
                    "boundary_candidate": ev.boundary_candidate,
                    "was_confirmed": ev.was_confirmed,
                    "confirmation_seconds": ev.confirmation_seconds,
                    "review_required": ev.review_required,
                    "post_confirmation_degraded": ev.post_confirmation_degraded,
                    "degradation_first_seen_seconds": ev.degradation_first_seen_seconds,
                    "post_confirmation_degradation_reasons": ev.post_confirmation_degradation_reasons,
                    "description": ev.description,
                    "state_history": ev.state_history,
                    "evidence_references": ev_by_event.get(ev.id, []),
                }
            )

        return sanitize_for_json(
            {
                "scientific_note": (
                    "TRINETRA detects persistent visual scene changes. It does not "
                    "independently establish object identity, ownership, abandonment, or intent. "
                    "Confidence is a heuristic score, not a calibrated probability."
                ),
                "job": {
                    "id": job.id,
                    "video_id": job.video_id,
                    "status": job.status,
                    "current_stage": job.current_stage,
                    "processed_frames": job.processed_frames,
                    "total_frames": job.total_frames,
                    "reference_seconds": job.reference_seconds,
                    "start_seconds": job.start_seconds,
                    "end_seconds": job.end_seconds,
                    "sampling_interval_seconds": job.sampling_interval_seconds,
                    "created_at": ensure_utc_iso(job.created_at),
                    "started_at": ensure_utc_iso(job.started_at),
                    "completed_at": ensure_utc_iso(job.completed_at),
                    "error_code": job.error_code,
                    "error_message": job.error_message,
                    "evidence_generation_status": job.evidence_generation_status,
                },
                "configuration": job.configuration_snapshot,
                "summary": job.summary_snapshot,
                "events": serialized_events,
                "job_evidence_references": ev_by_event.get("__job__", []),
            }
        )

    def _write_json_file(self, destination: Path, payload: dict) -> None:
        def _write(fh):
            json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
        self._atomic_write_text(destination, _write)

    def _write_csv_file(self, destination: Path, events, evidence_items) -> None:
        ev_by_event: dict[str, list] = {}
        for item in evidence_items:
            if item.event_id:
                ev_by_event.setdefault(item.event_id, []).append(item)

        fieldnames = ["event_id", "job_id", "video_id", *CSV_FIELDS, "description"]

        def _write(fh):
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for ev in events:
                items = ev_by_event.get(ev.id, [])
                annotated = next(
                    (i.original_filename for i in items if i.evidence_type == "annotated_frame"),
                    "",
                )
                row = sanitize_for_json(
                    {
                        "event_id": ev.id,
                        "job_id": ev.job_id,
                        "video_id": ev.video_id,
                        "track_id": ev.track_id,
                        "state": ev.event_state,
                        "first_seen_seconds": ev.first_seen_seconds,
                        "last_seen_seconds": ev.last_seen_seconds,
                        "duration_seconds": ev.duration_seconds,
                        "observations": ev.observations,
                        "missed_observations": ev.missed_observations,
                        "total_missed_observations": ev.total_missed_observations,
                        "detection_consistency": ev.detection_consistency,
                        "center_deviation_pixels": ev.center_deviation_pixels,
                        "bbox_stability": ev.bbox_stability,
                        "motion_samples": ev.motion_samples,
                        "motion_median_ratio": ev.motion_median_ratio,
                        "motion_p90_ratio": ev.motion_p90_ratio,
                        "settling_motion_samples": ev.settling_motion_samples,
                        "settling_motion_median_ratio": ev.settling_motion_median_ratio,
                        "settling_motion_p90_ratio": ev.settling_motion_p90_ratio,
                        "confidence": ev.confidence,
                        "confidence_kind": ev.confidence_type,
                        "boundary_candidate": ev.boundary_candidate,
                        "was_confirmed": ev.was_confirmed,
                        "confirmation_seconds": ev.confirmation_seconds,
                        "review_required": ev.review_required,
                        "post_confirmation_degraded": ev.post_confirmation_degraded,
                        "degradation_first_seen_seconds": ev.degradation_first_seen_seconds,
                        "post_confirmation_degradation_reasons": ev.post_confirmation_degradation_reasons,
                        "evidence_count": len(items),
                        "evidence_annotated_path": annotated,
                        "description": ev.description,
                    }
                )
                writer.writerow(row)

        self._atomic_write_text(destination, _write)

    def _write_zip_file(
        self,
        destination: Path,
        job,
        events,
        evidence_items,
        json_payload: dict,
    ) -> None:
        temp_json = destination.parent / f"._{job.id}_tmp.json"
        temp_csv = destination.parent / f"._{job.id}_tmp.csv"
        try:
            self._write_json_file(temp_json, json_payload)
            self._write_csv_file(temp_csv, events, evidence_items)

            def _write_zip(zf: zipfile.ZipFile):
                zf.write(temp_json, arcname="report.json")
                zf.write(temp_csv, arcname="events.csv")
                for item in evidence_items:
                    if item.job_id != job.id:
                        continue
                    try:
                        resolved = self.storage.resolve_relative_path(item.relative_storage_path)
                    except Exception:
                        continue
                    if not resolved.is_file():
                        continue
                    subfolder = item.event_id or "job_reports"
                    arcname = self._safe_arcname(
                        f"evidence/{subfolder}/{item.evidence_type}_{item.original_filename}"
                    )
                    zf.write(resolved, arcname=arcname)

            self._atomic_write_zip(destination, _write_zip)
        finally:
            temp_json.unlink(missing_ok=True)
            temp_csv.unlink(missing_ok=True)
