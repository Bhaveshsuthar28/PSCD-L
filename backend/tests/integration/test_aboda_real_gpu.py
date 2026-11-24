"""Real end-to-end ABODA integration test using the frozen TRINETRA checkpoint on GPU."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest
import torch

from backend.app.core.config import REPO_ROOT, Settings
from backend.app.main import create_app


ABODA_VIDEO10 = REPO_ROOT / "Dataset" / "ABODA" / "video10.avi"
FROZEN_CHECKPOINT = (
    REPO_ROOT / "outputs" / "training" / "checkpoints" / "trinetra-33-0.1482.ckpt"
)


@pytest.mark.skipif(
    not (ABODA_VIDEO10.is_file() and FROZEN_CHECKPOINT.is_file()),
    reason="ABODA video10.avi or frozen TRINETRA checkpoint not present.",
)
def test_real_aboda_video10_end_to_end(tmp_path: Path):
    storage_root = tmp_path / "storage"
    log_dir = tmp_path / "logs"
    settings = Settings(
        app_env="testing",
        database_url=f"sqlite:///{(storage_root / 'aboda_real.db').as_posix()}",
        storage_root=storage_root,
        log_dir=log_dir,
        min_free_disk_bytes=1024 * 1024,
        worker_count=1,
        gpu_job_concurrency=1,
        keep_model_loaded_across_jobs=True,
    )

    app = create_app(settings, engine_factory=None, start_workers=True)
    with TestClient(app) as client:
        # 1. Verify model starts uninitialized on /health
        h = client.get("/health").json()
        assert h["status"] == "healthy"
        assert h["checkpoint_available"] is True
        assert h["model_initialized"] is False

        # 2. Upload real ABODA video10.avi
        with ABODA_VIDEO10.open("rb") as fh:
            up = client.post(
                "/api/v1/videos/upload",
                files={"file": ("video10.avi", fh, "video/x-msvideo")},
            )
        assert up.status_code == 201, up.text
        vid = up.json()["id"]
        assert up.json()["width"] == 720
        assert up.json()["height"] == 480

        # 3. Submit validated ABODA window (ref=30, start=48, end=65, interval=1.0)
        t_submit = time.perf_counter()
        job_res = client.post(
            "/api/v1/processing/jobs",
            json={
                "video_id": vid,
                "reference_seconds": 30.0,
                "start_seconds": 48.0,
                "end_seconds": 65.0,
                "sampling_interval_seconds": 1.0,
                "configuration": {},
            },
        )
        assert job_res.status_code == 202, job_res.text
        job_id = job_res.json()["id"]

        # 4. Poll until terminal completion
        deadline = time.monotonic() + 180.0
        final = None
        while time.monotonic() < deadline:
            r = client.get(f"/api/v1/processing/jobs/{job_id}")
            assert r.status_code == 200
            final = r.json()
            if final["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.5)

        total_wall = time.perf_counter() - t_submit
        assert final is not None
        assert final["status"] == "COMPLETED", final
        assert final["processed_frames"] == 17
        assert final["confirmed_events_count"] == 1
        assert final["review_required_events_count"] == 0
        assert final["evidence_generation_status"] == "COMPLETED"

        # 5. Inspect system model & resource metrics
        model_info = client.get("/api/v1/system/model").json()
        assert model_info["model_initialized"] is True
        assert model_info["checkpoint_sha256_verified"] is True
        res_info = client.get("/api/v1/system/resources").json()
        print(
            f"\n[ABODA REAL] wall={total_wall:.2f}s "
            f"model_load={model_info['load_time_seconds']}s "
            f"fps={final['processing_fps']} "
            f"cuda={torch.cuda.is_available()} "
            f"gpu_alloc_mb={(res_info['gpu_memory_allocated_bytes'] or 0) / (1024*1024):.1f}"
        )

        # 6. Verify confirmed event matches Task 2 ABODA reference (~58.025s confirmation)
        events = client.get(
            f"/api/v1/processing/jobs/{job_id}/events?was_confirmed=true"
        ).json()["items"]
        assert len(events) == 1
        ev = events[0]
        assert ev["track_id"] == 1
        assert ev["was_confirmed"] is True
        assert ev["review_required"] is False
        assert abs(ev["confirmation_seconds"] - 58.025) < 0.2
        assert ev["confidence"] >= 0.95

        # 7. Verify evidence retrieval and JSON/CSV exports
        ev_files = client.get(f"/api/v1/events/{ev['id']}/evidence").json()["items"]
        assert len(ev_files) == 4
        for item in ev_files:
            dl = client.get(f"/api/v1/evidence/{item['id']}/download")
            assert dl.status_code == 200
            assert len(dl.content) > 0

        exp_json = client.get(f"/api/v1/processing/jobs/{job_id}/export?format=json")
        assert exp_json.status_code == 200
        parsed = json.loads(exp_json.content.decode("utf-8"))
        assert len(parsed["events"]) == final["summary_snapshot"]["total_tracks"] if "summary_snapshot" in final else 2

        exp_csv = client.get(f"/api/v1/processing/jobs/{job_id}/export?format=csv")
        assert exp_csv.status_code == 200
        rows = list(csv.DictReader(io.StringIO(exp_csv.content.decode("utf-8"))))
        assert len(rows) == 2

        # 8. Verify analytics
        ov = client.get("/api/v1/analytics/overview").json()
        assert ov["completed_jobs"] == 1
        assert ov["non_review_confirmations"] == 1
