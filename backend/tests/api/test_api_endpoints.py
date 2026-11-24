"""API tests for health, system, settings, videos, processing, and error envelope."""

from __future__ import annotations

from pathlib import Path


def test_root_health_system_and_settings_endpoints(client):
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["project"] == "TRINETRA PSCD&L"

    r_health = client.get("/health")
    assert r_health.status_code == 200
    h = r_health.json()
    assert h["status"] == "healthy"
    assert h["database"] == "connected"
    assert h["storage"] == "writable"
    assert h["worker"] == "running"
    # Health must never force model loading
    assert h["model_initialized"] is False

    r_sys = client.get("/api/v1/system/status")
    assert r_sys.status_code == 200
    assert "does not independently establish object identity" in r_sys.json()["scientific_scope"]

    r_res = client.get("/api/v1/system/resources")
    assert r_res.status_code == 200
    assert r_res.json()["disk_free_bytes"] > 0

    r_mod = client.get("/api/v1/system/model")
    assert r_mod.status_code == 200
    assert r_mod.json()["feature_dimensions"] == [1, 768, 37, 37]
    assert r_mod.json()["output_logits_dimensions"] == [1, 1, 512, 512]
    assert r_mod.json()["model_initialized"] is False

    r_settings = client.get("/api/v1/settings")
    assert r_settings.status_code == 200
    assert len(r_settings.json()["items"]) >= 1

    r_upd = client.put("/api/v1/settings", json={"settings": {"ui.page_size": 25}})
    assert r_upd.status_code == 200
    keys = {item["key"]: item["value_json"] for item in r_upd.json()["items"]}
    assert keys["ui.page_size"] == 25

    r_bad_set = client.put("/api/v1/settings", json={"settings": {"api_secret_token": "abc"}})
    assert r_bad_set.status_code == 400
    assert r_bad_set.json()["success"] is False
    assert r_bad_set.json()["error"]["code"] == "FORBIDDEN_SETTING_KEY"


def test_video_crud_and_duplicate_policy(client, sample_avi_path: Path):
    with sample_avi_path.open("rb") as fh:
        res = client.post(
            "/api/v1/videos/upload",
            files={"file": ("scene.avi", fh, "video/x-msvideo")},
        )
    assert res.status_code == 201, res.text
    video = res.json()
    vid = video["id"]
    assert video["original_filename"] == "scene.avi"
    assert video["width"] == 160
    assert video["height"] == 120
    assert video["frame_count"] == 140
    assert "relative_storage_path" not in video  # internal path not exposed

    # Duplicate upload rejected with 409 and existing_video_id
    with sample_avi_path.open("rb") as fh:
        dup = client.post(
            "/api/v1/videos/upload",
            files={"file": ("scene_copy.avi", fh, "video/x-msvideo")},
        )
    assert dup.status_code == 409
    assert dup.json()["error"]["code"] == "DUPLICATE_VIDEO"
    assert dup.json()["error"]["details"]["existing_video_id"] == vid

    r_list = client.get("/api/v1/videos")
    assert r_list.status_code == 200
    assert r_list.json()["pagination"]["total_items"] == 1

    r_get = client.get(f"/api/v1/videos/{vid}")
    assert r_get.status_code == 200
    assert r_get.json()["id"] == vid

    r_meta = client.get(f"/api/v1/videos/{vid}/metadata")
    assert r_meta.status_code == 200
    assert r_meta.json()["fps"] == 10.0

    # Video stream with HTTP Range seeking support
    r_full = client.get(f"/api/v1/videos/{vid}/stream")
    assert r_full.status_code == 200
    assert r_full.headers.get("accept-ranges") == "bytes"
    assert len(r_full.content) == video["file_size_bytes"]

    r_range = client.get(f"/api/v1/videos/{vid}/stream", headers={"Range": "bytes=0-127"})
    assert r_range.status_code == 206
    assert r_range.headers["content-range"].startswith("bytes 0-127/")
    assert len(r_range.content) == 128

    r_bad_range = client.get(f"/api/v1/videos/{vid}/stream", headers={"Range": "bytes=999999999-"})
    assert r_bad_range.status_code == 416

    # Browser-compatible server frame & MJPEG playback endpoints for AVI/non-native codecs
    r_frame = client.get(f"/api/v1/videos/{vid}/frame?t=1.5")
    assert r_frame.status_code == 200
    assert r_frame.headers["content-type"].startswith("image/jpeg")
    assert r_frame.content[:3] == b"\xff\xd8\xff"

    with client.stream("GET", f"/api/v1/videos/{vid}/mjpeg?start=13.0&speed=2.0") as r_mjpeg:
        assert r_mjpeg.status_code == 200
        assert "multipart/x-mixed-replace" in r_mjpeg.headers["content-type"]
        first_bytes = b""
        for chunk in r_mjpeg.iter_bytes():
            first_bytes += chunk
            if b"\xff\xd8\xff" in first_bytes:
                break
        assert b"--frame" in first_bytes
        assert b"Content-Type: image/jpeg" in first_bytes
        assert b"\xff\xd8\xff" in first_bytes

    # Paginated evidence gallery endpoint
    r_ev = client.get("/api/v1/evidence?page=1&page_size=10")
    assert r_ev.status_code == 200
    assert r_ev.json()["pagination"]["total_items"] == 0

    r_del = client.delete(f"/api/v1/videos/{vid}")
    assert r_del.status_code == 200
    assert r_del.json()["file_removed"] is True
    assert r_del.json()["record_deleted"] is True

    r_missing = client.get(f"/api/v1/videos/{vid}")
    assert r_missing.status_code == 404
