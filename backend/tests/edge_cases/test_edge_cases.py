"""Edge-case tests for uploads, processing, database, events, evidence, streaming, and analytics."""

from __future__ import annotations

from pathlib import Path
import time
import uuid

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.database.models import Evidence
from backend.app.main import create_app
from backend.tests.conftest import MockTrinetraEngine, create_synthetic_avi


def test_upload_edge_cases(client, tmp_path: Path):
    # Empty 0-byte upload
    r = client.post("/api/v1/videos/upload", files={"file": ("empty.avi", b"", "video/x-msvideo")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "EMPTY_UPLOAD"

    # Wrong extension
    r = client.post("/api/v1/videos/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "UNSUPPORTED_VIDEO_EXTENSION"

    # Fake video extension (text disguised as .mp4)
    r = client.post("/api/v1/videos/upload", files={"file": ("fake.mp4", b"plain text payload 12345", "video/mp4")})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "INVALID_VIDEO_SIGNATURE"

    # Truncated AVI
    r = client.post("/api/v1/videos/upload", files={"file": ("short.avi", b"RIFF1234", "video/x-msvideo")})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VIDEO_TRUNCATED"

    # Corrupted AVI with RIFF/AVI header
    r = client.post("/api/v1/videos/upload", files={"file": ("corrupt.avi", b"RIFF\x00\x00\x00\x00AVI " + b"\xff" * 128, "video/x-msvideo")})
    assert r.status_code == 422

    # Filename traversal attempt
    avi = create_synthetic_avi(tmp_path / "good.avi", duration_seconds=2.0)
    with avi.open("rb") as fh:
        r = client.post("/api/v1/videos/upload", files={"file": ("../../evil.avi", fh, "video/x-msvideo")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_FILENAME"

    # Unicode + long filename succeeds and stays sanitized
    long_unicode = ("caméra_日本語_" * 20) + ".avi"
    with avi.open("rb") as fh:
        r = client.post("/api/v1/videos/upload", files={"file": (long_unicode, fh, "video/x-msvideo")})
    assert r.status_code == 201
    assert len(r.json()["original_filename"]) <= 200


def test_oversized_and_storage_full_upload(tmp_path: Path, mock_engine_factory):
    avi = create_synthetic_avi(tmp_path / "size.avi", duration_seconds=2.0)
    tiny_limit_settings = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'tiny.db').as_posix()}",
        storage_root=tmp_path / "storage_tiny",
        log_dir=tmp_path / "logs_tiny",
        max_upload_size_bytes=1024,
        min_free_disk_bytes=1024,
    )
    app = create_app(tiny_limit_settings, engine_factory=mock_engine_factory, start_workers=False)
    with TestClient(app) as c:
        with avi.open("rb") as fh:
            r = c.post("/api/v1/videos/upload", files={"file": ("big.avi", fh, "video/x-msvideo")})
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "UPLOAD_TOO_LARGE"

    full_disk_settings = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'full.db').as_posix()}",
        storage_root=tmp_path / "storage_full",
        log_dir=tmp_path / "logs_full",
        min_free_disk_bytes=10**18,
    )
    app2 = create_app(full_disk_settings, engine_factory=mock_engine_factory, start_workers=False)
    with TestClient(app2) as c2:
        with avi.open("rb") as fh:
            r = c2.post("/api/v1/videos/upload", files={"file": ("disk.avi", fh, "video/x-msvideo")})
        assert r.status_code == 507
        assert r.json()["error"]["code"] == "INSUFFICIENT_STORAGE"


def test_processing_validation_and_worker_edge_cases(client, tmp_path: Path):
    # Invalid UUID
    r = client.post("/api/v1/processing/jobs", json={"video_id": "bad-id"})
    assert r.status_code == 400

    # Missing video
    r = client.post("/api/v1/processing/jobs", json={"video_id": str(uuid.uuid4())})
    assert r.status_code == 404

    avi = create_synthetic_avi(tmp_path / "proc.avi", duration_seconds=5.0, fps=10.0)
    with avi.open("rb") as fh:
        vid = client.post("/api/v1/videos/upload", files={"file": ("proc.avi", fh, "video/x-msvideo")}).json()["id"]

    # Reference beyond duration
    r = client.post("/api/v1/processing/jobs", json={"video_id": vid, "reference_seconds": 9.0, "start_seconds": 0.0})
    assert r.status_code == 400

    # Start after end
    r = client.post("/api/v1/processing/jobs", json={"video_id": vid, "start_seconds": 4.0, "end_seconds": 2.0})
    assert r.status_code == 422

    # Negative timestamp
    r = client.post("/api/v1/processing/jobs", json={"video_id": vid, "reference_seconds": -1.0})
    assert r.status_code == 422

    # Zero / out-of-bounds sampling interval
    r = client.post("/api/v1/processing/jobs", json={"video_id": vid, "sampling_interval_seconds": 0.0})
    assert r.status_code == 422
    r = client.post("/api/v1/processing/jobs", json={"video_id": vid, "sampling_interval_seconds": 0.001})
    assert r.status_code == 400
    r = client.post("/api/v1/processing/jobs", json={"video_id": vid, "sampling_interval_seconds": 999.0})
    assert r.status_code == 400

    # Disallowed configuration key
    r = client.post(
        "/api/v1/processing/jobs",
        json={"video_id": vid, "configuration": {"checkpoint_path": "evil.ckpt"}},
    )
    assert r.status_code == 400


def test_model_failure_and_cancellation_and_zero_event_job(tmp_path: Path):
    # 1. Failing model initialization marks job FAILED cleanly
    def failing_factory(ckpt, dev):
        raise RuntimeError("Simulated missing checkpoint or CUDA error")

    settings = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'fail.db').as_posix()}",
        storage_root=tmp_path / "storage_fail",
        log_dir=tmp_path / "logs_fail",
        min_free_disk_bytes=1024,
    )
    app_fail = create_app(settings, engine_factory=failing_factory, start_workers=True)
    avi = create_synthetic_avi(tmp_path / "empty_scene.avi", duration_seconds=4.0, fps=10.0, static_box_from_second=None)

    with TestClient(app_fail) as c:
        with avi.open("rb") as fh:
            vid = c.post("/api/v1/videos/upload", files={"file": ("empty_scene.avi", fh, "video/x-msvideo")}).json()["id"]
        job_id = c.post("/api/v1/processing/jobs", json={"video_id": vid, "start_seconds": 0.0, "end_seconds": 3.0}).json()["id"]
        for _ in range(100):
            j = c.get(f"/api/v1/processing/jobs/{job_id}").json()
            if j["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.05)
        assert j["status"] == "FAILED"
        assert j["error_code"] == "MODEL_INITIALIZATION_FAILED"

    # 2. Cooperative cancellation during slow frame processing
    def slow_factory(ckpt, dev):
        eng = MockTrinetraEngine(ckpt, dev)
        eng.delay_per_frame = 0.2
        return eng

    settings2 = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'cancel.db').as_posix()}",
        storage_root=tmp_path / "storage_cancel",
        log_dir=tmp_path / "logs_cancel",
        min_free_disk_bytes=1024,
    )
    app_cancel = create_app(settings2, engine_factory=slow_factory, start_workers=True)
    avi_long = create_synthetic_avi(tmp_path / "long.avi", duration_seconds=10.0, fps=10.0, static_box_from_second=None)

    with TestClient(app_cancel) as c2:
        with avi_long.open("rb") as fh:
            vid2 = c2.post("/api/v1/videos/upload", files={"file": ("long.avi", fh, "video/x-msvideo")}).json()["id"]
        job2 = c2.post(
            "/api/v1/processing/jobs",
            json={"video_id": vid2, "start_seconds": 0.0, "end_seconds": 9.0, "sampling_interval_seconds": 0.5},
        ).json()["id"]
        time.sleep(0.15)
        cancel_res = c2.post(f"/api/v1/processing/jobs/{job2}/cancel")
        assert cancel_res.status_code == 200
        for _ in range(100):
            j2 = c2.get(f"/api/v1/processing/jobs/{job2}").json()
            if j2["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.05)
        assert j2["status"] == "CANCELLED"

        # 3. Zero-candidate job completes cleanly with 0 events
    settings3 = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'zero.db').as_posix()}",
        storage_root=tmp_path / "storage_zero",
        log_dir=tmp_path / "logs_zero",
        min_free_disk_bytes=1024,
    )
    app_zero = create_app(settings3, engine_factory=lambda c, d: MockTrinetraEngine(c, d), start_workers=True)
    with TestClient(app_zero) as c3:
        with avi.open("rb") as fh:
            vid3 = c3.post("/api/v1/videos/upload", files={"file": ("no_change.avi", fh, "video/x-msvideo")}).json()["id"]
        job3 = c3.post("/api/v1/processing/jobs", json={"video_id": vid3, "start_seconds": 0.0, "end_seconds": 3.0}).json()["id"]
        for _ in range(100):
            j3 = c3.get(f"/api/v1/processing/jobs/{job3}").json()
            if j3["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.05)
        assert j3["status"] == "COMPLETED"
        assert j3["confirmed_events_count"] == 0


def test_evidence_analytics_and_streaming_edge_cases(client):
    # Empty DB analytics
    assert client.get("/api/v1/analytics/overview").json()["total_uploaded_videos"] == 0
    assert client.get("/api/v1/analytics/timeline").json()["count"] == 0

    # Invalid analytics date range
    r_bad_date = client.get("/api/v1/analytics/events?start_date=2026-05-10&end_date=2026-05-01")
    assert r_bad_date.status_code == 400

    # Invalid pagination & filters
    assert client.get("/api/v1/events?page=0").status_code == 422
    assert client.get("/api/v1/events?sort_by=evil_sql").status_code == 400

    # Missing/invalid evidence & streaming IDs
    assert client.get("/api/v1/evidence/not-a-uuid/download").status_code == 400
    assert client.get(f"/api/v1/evidence/{uuid.uuid4()}/download").status_code == 404
    assert client.get(f"/api/v1/processing/jobs/{uuid.uuid4()}/stream").status_code == 404
    assert client.get("/api/v1/processing/jobs/invalid/stream").status_code == 400


def test_worker_unavailable_returns_503(tmp_path: Path, mock_engine_factory):
    avi = create_synthetic_avi(tmp_path / "w503.avi", duration_seconds=3.0)
    settings = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'w503.db').as_posix()}",
        storage_root=tmp_path / "storage_w503",
        log_dir=tmp_path / "logs_w503",
        min_free_disk_bytes=1024,
    )
    app = create_app(settings, engine_factory=mock_engine_factory, start_workers=False)
    with TestClient(app) as c:
        with avi.open("rb") as fh:
            vid = c.post("/api/v1/videos/upload", files={"file": ("w503.avi", fh, "video/x-msvideo")}).json()["id"]
        res = c.post("/api/v1/processing/jobs", json={"video_id": vid})
        assert res.status_code == 503
        assert res.json()["error"]["code"] == "WORKER_UNAVAILABLE"


def test_missing_evidence_file_on_disk_returns_404(client, sample_avi_path: Path):
    # Create job + wait, then remove an evidence file from disk and ensure 404 (never fabricated)
    with sample_avi_path.open("rb") as fh:
        vid = client.post("/api/v1/videos/upload", files={"file": ("ev_miss.avi", fh, "video/x-msvideo")}).json()["id"]
    job_id = client.post(
        "/api/v1/processing/jobs",
        json={
            "video_id": vid,
            "start_seconds": 1.0,
            "end_seconds": 13.5,
            "configuration": {
                "temporal_config": {
                    "min_persistence_seconds": 5.0,
                    "min_observations": 4,
                    "settling_min_seconds": 2.0,
                    "min_motion_samples": 2,
                    "settling_motion_window_samples": 3,
                }
            },
        },
    ).json()["id"]
    for _ in range(100):
        j = client.get(f"/api/v1/processing/jobs/{job_id}").json()
        if j["status"] == "COMPLETED":
            break
        time.sleep(0.05)
    events = client.get(f"/api/v1/processing/jobs/{job_id}/events?was_confirmed=true").json()["items"]
    ev_items = client.get(f"/api/v1/events/{events[0]['id']}/evidence").json()["items"]
    target_id = ev_items[0]["id"]

    container = client.app.state.container
    with container.db_manager.session() as db:
        row = db.get(Evidence, target_id)
        file_path = container.storage.resolve_relative_path(row.relative_storage_path)
        file_path.unlink()

    dl = client.get(f"/api/v1/evidence/{target_id}/download")
    assert dl.status_code == 404
    assert dl.json()["error"]["code"] == "EVIDENCE_FILE_MISSING"


def test_duplicate_job_queue_full_delete_conflict_and_cuda_oom(tmp_path: Path):
    import torch

    def slow_factory(ckpt, dev):
        eng = MockTrinetraEngine(ckpt, dev)
        eng.delay_per_frame = 0.25
        return eng

    settings = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'qfull.db').as_posix()}",
        storage_root=tmp_path / "storage_qfull",
        log_dir=tmp_path / "logs_qfull",
        min_free_disk_bytes=1024,
        max_queue_size=1,
    )
    avi = create_synthetic_avi(tmp_path / "qfull.avi", duration_seconds=6.0, fps=10.0, static_box_from_second=None)
    app = create_app(settings, engine_factory=slow_factory, start_workers=True)

    with TestClient(app) as c:
        with avi.open("rb") as fh:
            vid = c.post("/api/v1/videos/upload", files={"file": ("qfull.avi", fh, "video/x-msvideo")}).json()["id"]

        # Job 1 starts processing
        j1 = c.post(
            "/api/v1/processing/jobs",
            json={"video_id": vid, "start_seconds": 0.0, "end_seconds": 5.0, "sampling_interval_seconds": 0.5},
        )
        assert j1.status_code == 202
        j1_id = j1.json()["id"]

        # Duplicate identical job rejected with 409
        dup = c.post(
            "/api/v1/processing/jobs",
            json={"video_id": vid, "start_seconds": 0.0, "end_seconds": 5.0, "sampling_interval_seconds": 0.5},
        )
        assert dup.status_code == 409
        assert dup.json()["error"]["code"] == "DUPLICATE_JOB_SUBMISSION"

        # Active video deletion rejected with 409
        del_active = c.delete(f"/api/v1/videos/{vid}")
        assert del_active.status_code == 409
        assert del_active.json()["error"]["code"] == "VIDEO_IN_USE"

        # Fill bounded queue (capacity=1), then next distinct job hits 429
        time.sleep(0.1)
        j2 = c.post(
            "/api/v1/processing/jobs",
            json={"video_id": vid, "start_seconds": 0.5, "end_seconds": 5.0, "sampling_interval_seconds": 0.5},
        )
        assert j2.status_code == 202
        j3 = c.post(
            "/api/v1/processing/jobs",
            json={"video_id": vid, "start_seconds": 1.0, "end_seconds": 5.0, "sampling_interval_seconds": 0.5},
        )
        assert j3.status_code == 429
        assert j3.json()["error"]["code"] == "PROCESSING_QUEUE_FULL"

        c.post(f"/api/v1/processing/jobs/{j1_id}/cancel")
        c.post(f"/api/v1/processing/jobs/{j2.json()['id']}/cancel")

    # CUDA OOM simulation
    class OomEngine(MockTrinetraEngine):
        def predict(self, current_frame_bgr):
            raise torch.cuda.OutOfMemoryError("Simulated CUDA OOM")

    settings_oom = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'oom.db').as_posix()}",
        storage_root=tmp_path / "storage_oom",
        log_dir=tmp_path / "logs_oom",
        min_free_disk_bytes=1024,
    )
    app_oom = create_app(settings_oom, engine_factory=lambda c, d: OomEngine(c, d), start_workers=True)
    with TestClient(app_oom) as c_oom:
        with avi.open("rb") as fh:
            vid_oom = c_oom.post("/api/v1/videos/upload", files={"file": ("oom.avi", fh, "video/x-msvideo")}).json()["id"]
        job_oom = c_oom.post("/api/v1/processing/jobs", json={"video_id": vid_oom, "start_seconds": 0.0, "end_seconds": 4.0}).json()["id"]
        for _ in range(100):
            st = c_oom.get(f"/api/v1/processing/jobs/{job_oom}").json()
            if st["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.05)
        assert st["status"] == "FAILED"
        assert st["error_code"] == "CUDA_OUT_OF_MEMORY"


def test_evidence_write_failure_preserves_confirmed_event(tmp_path: Path, mock_engine_factory, monkeypatch):
    from src.inference.evidence import EvidenceWriter

    def broken_save(*args, **kwargs):
        raise OSError("Simulated disk write failure for evidence")

    monkeypatch.setattr(EvidenceWriter, "save", broken_save)

    settings = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(tmp_path / 'evfail.db').as_posix()}",
        storage_root=tmp_path / "storage_evfail",
        log_dir=tmp_path / "logs_evfail",
        min_free_disk_bytes=1024,
    )
    avi = create_synthetic_avi(tmp_path / "evfail.avi", duration_seconds=14.0, fps=10.0, static_box_from_second=1.0)
    app = create_app(settings, engine_factory=mock_engine_factory, start_workers=True)
    with TestClient(app) as c:
        with avi.open("rb") as fh:
            vid = c.post("/api/v1/videos/upload", files={"file": ("evfail.avi", fh, "video/x-msvideo")}).json()["id"]
        job_id = c.post(
            "/api/v1/processing/jobs",
            json={
                "video_id": vid,
                "start_seconds": 1.0,
                "end_seconds": 13.5,
                "configuration": {
                    "temporal_config": {
                        "min_persistence_seconds": 5.0,
                        "min_observations": 4,
                        "settling_min_seconds": 2.0,
                        "min_motion_samples": 2,
                        "settling_motion_window_samples": 3,
                    }
                },
            },
        ).json()["id"]
        for _ in range(100):
            j = c.get(f"/api/v1/processing/jobs/{job_id}").json()
            if j["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.05)
        assert j["status"] == "COMPLETED"
        assert j["confirmed_events_count"] >= 1
        assert j["evidence_generation_status"] == "PARTIAL_FAILED"
        assert "Simulated disk write failure" in (j["evidence_generation_error"] or "")

