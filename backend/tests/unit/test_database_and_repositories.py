"""Unit tests for SQLAlchemy models, repositories, state transitions, and recovery."""

from __future__ import annotations

from pathlib import Path
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import Settings
from backend.app.core.exceptions import ConflictError
from backend.app.database.connection import DatabaseManager
from backend.app.database.models import Event, Evidence, ProcessingJob, Video
from backend.app.database.repositories import (
    EventRepository,
    EvidenceRepository,
    JobRepository,
    VideoRepository,
)
from backend.app.integrations.trinetra_engine import ManagedInferenceEngine
from backend.app.services.storage_service import StorageService
from backend.app.workers.job_manager import JobManager
from backend.app.workers.progress_manager import ProgressManager


def _make_db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "unit.db"
    mgr = DatabaseManager(f"sqlite:///{db_path.as_posix()}")
    mgr.initialize()
    return mgr


def _seed_video(db, sha_suffix: str = "1") -> Video:
    vid = Video(
        id=str(uuid.uuid4()),
        original_filename="test.avi",
        stored_filename=f"{uuid.uuid4()}.avi",
        relative_storage_path=f"uploads/{uuid.uuid4()}.avi",
        file_size_bytes=1234,
        sha256=(sha_suffix * 64)[:64],
        duration_seconds=20.0,
        fps=10.0,
        frame_count=200,
        width=160,
        height=120,
        codec="MJPG",
        format="avi",
    )
    return VideoRepository(db).create(vid)


def test_foreign_keys_and_unique_constraints(tmp_path: Path):
    mgr = _make_db(tmp_path)
    try:
        with pytest.raises(IntegrityError):
            with mgr.session() as db:
                JobRepository(db).create(
                    ProcessingJob(
                        id=str(uuid.uuid4()),
                        video_id=str(uuid.uuid4()),  # missing FK
                        output_directory="evidence/job_x",
                    )
                )

        with mgr.session() as db:
            v = _seed_video(db)
            job = JobRepository(db).create(
                ProcessingJob(
                    id=str(uuid.uuid4()),
                    video_id=v.id,
                    output_directory="evidence/job_1",
                )
            )
            ev = Event(
                id=str(uuid.uuid4()),
                job_id=job.id,
                video_id=v.id,
                track_id=1,
                event_state="CONFIRMED",
                first_seen_seconds=1.0,
                last_seen_seconds=12.0,
                duration_seconds=11.0,
                confidence=0.9,
                was_confirmed=True,
                confirmation_seconds=11.0,
            )
            db.add(ev)
            db.flush()

            item = EvidenceRepository(db).create(
                Evidence(
                    id=str(uuid.uuid4()),
                    event_id=ev.id,
                    job_id=job.id,
                    evidence_type="annotated_frame",
                    relative_storage_path="evidence/job_1/track_1/frame.jpg",
                    original_filename="frame.jpg",
                    mime_type="image/jpeg",
                    file_size_bytes=100,
                    sha256="B" * 64,
                )
            )
            assert item.id is not None
    finally:
        mgr.dispose()


def test_illegal_job_state_transitions_rejected(tmp_path: Path):
    mgr = _make_db(tmp_path)
    try:
        with mgr.session() as db:
            v = _seed_video(db)
            repo = JobRepository(db)
            job = repo.create(
                ProcessingJob(
                    id=str(uuid.uuid4()),
                    video_id=v.id,
                    output_directory="evidence/job_trans",
                )
            )
            repo.transition_status(job, "VALIDATING")
            repo.transition_status(job, "INITIALIZING")
            repo.transition_status(job, "PROCESSING")
            repo.transition_status(job, "FINALIZING")
            repo.transition_status(job, "COMPLETED")

            with pytest.raises(ConflictError):
                repo.transition_status(job, "PROCESSING")

            job2 = repo.create(
                ProcessingJob(
                    id=str(uuid.uuid4()),
                    video_id=v.id,
                    output_directory="evidence/job_cancel",
                )
            )
            repo.transition_status(job2, "VALIDATING")
            repo.transition_status(job2, "CANCEL_REQUESTED")
            with pytest.raises(ConflictError):
                repo.transition_status(job2, "COMPLETED")
    finally:
        mgr.dispose()


def test_crash_recovery_marks_interrupted_jobs_failed(tmp_path: Path):
    cfg = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'rec.db').as_posix()}",
        storage_root=tmp_path / "storage",
        log_dir=tmp_path / "logs",
    )
    mgr = DatabaseManager(cfg.database_url)
    mgr.initialize()
    try:
        with mgr.session() as db:
            v = _seed_video(db, "C")
            repo = JobRepository(db)
            j1 = repo.create(
                ProcessingJob(
                    id=str(uuid.uuid4()),
                    video_id=v.id,
                    status="PROCESSING",
                    current_stage="PROCESSING_FRAMES",
                    output_directory="evidence/j1",
                )
            )
            j1_id = j1.id

        storage = StorageService(cfg)
        jm = JobManager(
            cfg,
            mgr,
            storage,
            ManagedInferenceEngine(cfg),
            ProgressManager(),
        )
        recovered = jm.recover_interrupted_jobs()
        assert recovered == 1

        with mgr.session() as db:
            j_after = JobRepository(db).get(j1_id)
            assert j_after.status == "FAILED"
            assert j_after.error_code == "INTERRUPTED_BY_RESTART"
    finally:
        mgr.dispose()


def test_event_repository_filtering_and_pagination(tmp_path: Path):
    mgr = _make_db(tmp_path)
    try:
        with mgr.session() as db:
            v = _seed_video(db, "D")
            job = JobRepository(db).create(
                ProcessingJob(
                    id=str(uuid.uuid4()),
                    video_id=v.id,
                    output_directory="evidence/j_ev",
                )
            )
            ev_repo = EventRepository(db)
            ev_repo.create_many(
                [
                    Event(
                        id=str(uuid.uuid4()),
                        job_id=job.id,
                        video_id=v.id,
                        track_id=1,
                        event_state="COMPLETED",
                        first_seen_seconds=1.0,
                        last_seen_seconds=15.0,
                        duration_seconds=14.0,
                        confidence=0.95,
                        was_confirmed=True,
                        confirmation_seconds=11.0,
                        review_required=False,
                    ),
                    Event(
                        id=str(uuid.uuid4()),
                        job_id=job.id,
                        video_id=v.id,
                        track_id=2,
                        event_state="COMPLETED",
                        first_seen_seconds=2.0,
                        last_seen_seconds=14.0,
                        duration_seconds=12.0,
                        confidence=0.60,
                        was_confirmed=True,
                        confirmation_seconds=12.0,
                        review_required=True,
                        post_confirmation_degraded=True,
                    ),
                    Event(
                        id=str(uuid.uuid4()),
                        job_id=job.id,
                        video_id=v.id,
                        track_id=3,
                        event_state="COMPLETED",
                        first_seen_seconds=3.0,
                        last_seen_seconds=5.0,
                        duration_seconds=2.0,
                        confidence=0.30,
                        was_confirmed=False,
                        review_required=False,
                    ),
                ]
            )

            items, total = ev_repo.list_filtered(
                offset=0,
                limit=10,
                was_confirmed=True,
                review_required=False,
            )
            assert total == 1
            assert items[0].track_id == 1

            items2, total2 = ev_repo.list_filtered(
                offset=0, limit=2, min_confidence=0.5, sort_by="confidence", sort_order="desc"
            )
            assert total2 == 2
            assert [i.track_id for i in items2] == [1, 2]
    finally:
        mgr.dispose()
