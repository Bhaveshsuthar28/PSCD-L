"""End-to-end integration tests with synthetic videos and mocked inference engine."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import time
import zipfile

from backend.tests.conftest import create_synthetic_avi


def _wait_for_terminal_job(client, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = client.get(f"/api/v1/processing/jobs/{job_id}")
        assert res.status_code == 200
        payload = res.json()
        if payload["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return payload
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


def test_full_video_processing_events_evidence_exports_analytics_and_cleanup(
    client,
    sample_avi_path: Path,
):
    # 1. Upload valid synthetic video
    with sample_avi_path.open("rb") as fh:
        up = client.post(
            "/api/v1/videos/upload",
            files={"file": ("synthetic_abandoned.avi", fh, "video/x-msvideo")},
        )
    assert up.status_code == 201
    video_id = up.json()["id"]

    # 2. Create processing job
    job_req = {
        "video_id": video_id,
        "reference_seconds": 0.0,
        "start_seconds": 1.0,
        "end_seconds": 13.5,
        "sampling_interval_seconds": 1.0,
        "configuration": {
            "temporal_config": {
                "min_persistence_seconds": 5.0,
                "min_observations": 4,
                "settling_min_seconds": 2.0,
                "min_motion_samples": 2,
                "settling_motion_window_samples": 3,
            }
        },
    }
    create_res = client.post("/api/v1/processing/jobs", json=job_req)
    assert create_res.status_code == 202, create_res.text
    job_id = create_res.json()["id"]

    # 3. Wait for completion
    final_job = _wait_for_terminal_job(client, job_id)
    assert final_job["status"] == "COMPLETED", final_job
    assert final_job["progress_percentage"] == 100.0
    assert final_job["confirmed_events_count"] >= 1
    assert final_job["evidence_generation_status"] == "COMPLETED"

    # 4. Verify SSE stream on completed job delivers snapshot + completed cleanly
    with client.stream("GET", f"/api/v1/processing/jobs/{job_id}/stream") as stream_res:
        assert stream_res.status_code == 200
        text = "".join(stream_res.iter_text())
        assert "event: snapshot" in text
        assert "event: completed" in text

    # 5. Verify job logs and summary (no internal filesystem paths exposed)
    logs_res = client.get(f"/api/v1/processing/jobs/{job_id}/logs")
    assert logs_res.status_code == 200
    assert logs_res.json()["total_items"] >= 2

    summary_res = client.get(f"/api/v1/processing/jobs/{job_id}/summary")
    assert summary_res.status_code == 200
    summary = summary_res.json()["summary"]
    assert summary is not None
    assert "video" not in summary

    # 6. Verify persisted events & filtering
    ev_list = client.get(f"/api/v1/processing/jobs/{job_id}/events?was_confirmed=true")
    assert ev_list.status_code == 200
    events = ev_list.json()["items"]
    assert len(events) >= 1
    event = events[0]
    event_id = event["id"]
    assert event["was_confirmed"] is True
    assert event["confidence_type"] == "heuristic_score_not_probability"
    assert "abandoned" not in event["description"].lower() or "does not establish" in event["description"].lower()

    ev_detail = client.get(f"/api/v1/events/{event_id}")
    assert ev_detail.status_code == 200
    detail_json = ev_detail.json()
    assert len(detail_json["evidence_items"]) >= 3

    # Also test /api/v1/videos/{video_id}/events
    vid_ev = client.get(f"/api/v1/videos/{video_id}/events")
    assert vid_ev.status_code == 200
    assert vid_ev.json()["pagination"]["total_items"] >= 1

    # 7. Verify evidence retrieval & download
    ev_files = client.get(f"/api/v1/events/{event_id}/evidence")
    assert ev_files.status_code == 200
    items = ev_files.json()["items"]
    types_present = {i["evidence_type"] for i in items}
    assert {"original_frame", "annotated_frame", "probability_map", "metadata"}.issubset(types_present)

    annotated_item = next(i for i in items if i["evidence_type"] == "annotated_frame")
    meta_res = client.get(f"/api/v1/evidence/{annotated_item['id']}")
    assert meta_res.status_code == 200
    dl_res = client.get(f"/api/v1/evidence/{annotated_item['id']}/download")
    assert dl_res.status_code == 200
    assert dl_res.headers["content-type"].startswith("image/jpeg")
    assert len(dl_res.content) > 100

    by_type = client.get(f"/api/v1/events/{event_id}/evidence/probability_map")
    assert by_type.status_code == 200
    assert by_type.headers["content-type"].startswith("image/png")

    # 8. Verify JSON, CSV, and ZIP exports
    exp_json = client.get(f"/api/v1/processing/jobs/{job_id}/export?format=json")
    assert exp_json.status_code == 200
    parsed_json = json.loads(exp_json.content.decode("utf-8"))
    assert len(parsed_json["events"]) >= 1

    exp_csv = client.get(f"/api/v1/processing/jobs/{job_id}/export?format=csv")
    assert exp_csv.status_code == 200
    rows = list(csv.DictReader(io.StringIO(exp_csv.content.decode("utf-8"))))
    assert len(rows) == len(parsed_json["events"])

    exp_zip = client.get(f"/api/v1/processing/jobs/{job_id}/export?format=zip")
    assert exp_zip.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exp_zip.content)) as zf:
        names = zf.namelist()
        assert "report.json" in names
        assert "events.csv" in names
        assert any(name.startswith("evidence/") for name in names)

    # 9. Verify analytics reflect actual DB records
    ov = client.get("/api/v1/analytics/overview").json()
    assert ov["total_uploaded_videos"] == 1
    assert ov["completed_jobs"] == 1
    assert ov["historical_confirmations"] >= 1

    assert client.get("/api/v1/analytics/videos").status_code == 200
    assert client.get("/api/v1/analytics/events").status_code == 200
    assert client.get("/api/v1/analytics/processing").status_code == 200
    assert client.get("/api/v1/analytics/timeline").status_code == 200

    # 10. Delete video safely -> preserves historical job & event records
    del_res = client.delete(f"/api/v1/videos/{video_id}")
    assert del_res.status_code == 200
    assert del_res.json()["file_removed"] is True
    assert del_res.json()["historical_records_preserved"] is True
    assert client.get(f"/api/v1/events/{event_id}").status_code == 200


def test_post_confirmation_degradation_preserved(client, tmp_path: Path):
    degraded_avi = create_synthetic_avi(
        tmp_path / "degraded.avi",
        duration_seconds=16.0,
        fps=10.0,
        static_box_from_second=1.0,
        unstable_after_second=8.0,
    )
    with degraded_avi.open("rb") as fh:
        up = client.post(
            "/api/v1/videos/upload",
            files={"file": ("degraded.avi", fh, "video/x-msvideo")},
        )
    assert up.status_code == 201
    video_id = up.json()["id"]

    job_res = client.post(
        "/api/v1/processing/jobs",
        json={
            "video_id": video_id,
            "reference_seconds": 0.0,
            "start_seconds": 1.0,
            "end_seconds": 15.0,
            "sampling_interval_seconds": 1.0,
            "configuration": {
                "temporal_config": {
                    "min_persistence_seconds": 4.0,
                    "min_observations": 4,
                    "settling_min_seconds": 2.0,
                    "min_motion_samples": 2,
                    "settling_motion_window_samples": 3,
                }
            },
        },
    )
    assert job_res.status_code == 202
    final = _wait_for_terminal_job(client, job_res.json()["id"])
    assert final["status"] == "COMPLETED"

    events = client.get(f"/api/v1/processing/jobs/{final['id']}/events?was_confirmed=true").json()["items"]
    assert len(events) >= 1
    confirmed = events[0]
    assert confirmed["was_confirmed"] is True
    assert confirmed["review_required"] is True
    assert confirmed["post_confirmation_degraded"] is True
    assert len(confirmed["post_confirmation_degradation_reasons"]) >= 1
