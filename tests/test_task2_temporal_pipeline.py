from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.inference.evidence import EvidenceWriter
from src.inference.event_store import EventStore
from src.inference.temporal_engine import (
    TemporalConfig,
    TemporalPersistenceEngine,
)
from src.inference.tracker import RegionTracker


def _detection(x: int = 20) -> dict:
    return {
        "bbox": [x, 30, 20, 20],
        "center": [x + 10.0, 40.0],
        "area": 400,
        "mean_probability": 0.9,
        "max_probability": 0.95,
        "boundary": False,
    }


def _temporal_config(**overrides) -> TemporalConfig:
    values = {
        "min_persistence_seconds": 2.0,
        "min_observations": 3,
        "settling_min_seconds": 1.0,
        "min_motion_samples": 3,
        "settling_motion_window_samples": 3,
        "confidence_threshold": 0.0,
    }
    values.update(overrides)
    return TemporalConfig(**values)


def _update_ratios(
    engine: TemporalPersistenceEngine,
    tracker: RegionTracker,
    ratios: list[float],
) -> object:
    for second, ratio in enumerate(ratios):
        track = tracker.update([_detection()], float(second))[0]
        event = engine.update(
            track,
            float(second),
            {"region_background_ratio": ratio},
        )

    return event


def test_tracker_marks_missed_and_reacquired_observations():
    tracker = RegionTracker(max_missed_observations=2)
    track = tracker.update([_detection()], 0.0)[0]
    assert track.observed_in_latest_update
    assert not track.observed_in_previous_update

    track = tracker.update([], 1.0)[0]
    assert not track.observed_in_latest_update
    assert track.observed_in_previous_update

    track = tracker.update([_detection()], 2.0)[0]
    assert track.observed_in_latest_update
    assert not track.observed_in_previous_update
    assert track.missed_observations == 0
    assert len(track.missed_history) == 1


def test_recent_settling_window_and_full_history_are_both_reported():
    config = _temporal_config()
    engine = TemporalPersistenceEngine(config)
    tracker = RegionTracker()

    ratios = [8.0, 8.0, 0.4, 0.5, 0.6]
    event = _update_ratios(engine, tracker, ratios)

    assert event.motion_samples == 5
    assert event.motion_median_ratio == 0.6
    assert event.settling_motion_samples == 3
    assert event.settling_motion_median_ratio == 0.5
    assert event.state == "CONFIRMED"


def test_unlimited_motion_history_retains_full_event_history():
    engine = TemporalPersistenceEngine(_temporal_config())
    tracker = RegionTracker()
    event = _update_ratios(
        engine,
        tracker,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )

    assert engine.config.max_motion_history is None
    assert engine.motion_history[event.track_id] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )
    assert event.motion_samples == 7
    assert event.settling_motion_samples == 3
    assert event.settling_motion_median_ratio == pytest.approx(0.6)


def test_capped_motion_history_retains_only_the_configured_tail():
    engine = TemporalPersistenceEngine(
        _temporal_config(max_motion_history=3)
    )
    tracker = RegionTracker()
    event = _update_ratios(
        engine,
        tracker,
        [0.1, 0.2, 0.3, 0.4, 0.5],
    )

    assert engine.motion_history[event.track_id] == pytest.approx(
        [0.3, 0.4, 0.5]
    )
    assert event.motion_samples == 3
    assert event.motion_median_ratio == pytest.approx(0.4)
    assert event.settling_motion_median_ratio == pytest.approx(0.4)


def test_recent_motion_deterioration_requires_review_without_retracting_confirmation():
    engine = TemporalPersistenceEngine(_temporal_config())
    tracker = RegionTracker()
    confirmed = _update_ratios(engine, tracker, [0.2, 0.2, 0.2])
    assert confirmed.state == "CONFIRMED"
    confirmation_seconds = confirmed.confirmation_seconds

    track = tracker.update([_detection()], 3.0)[0]
    event = engine.update(
        track,
        3.0,
        {"region_background_ratio": 4.0},
    )

    assert event.state == "CONFIRMED"
    assert event.was_confirmed
    assert event.confirmation_seconds == confirmation_seconds
    assert event.review_required
    assert event.post_confirmation_degraded
    assert "recent_motion_settling" in (
        event.post_confirmation_degradation_reasons
    )


def test_tracker_expiration_and_simultaneous_track_association():
    tracker = RegionTracker(max_missed_observations=1)
    tracks = tracker.update([_detection(20), _detection(200)], 0.0)
    assert [track.track_id for track in tracks] == [1, 2]

    tracks = tracker.update([_detection(202), _detection(22)], 1.0)
    assert [track.track_id for track in tracks] == [1, 2]
    assert tracks[0].center[0] == 32.0
    assert tracks[1].center[0] == 212.0

    assert len(tracker.update([], 2.0)) == 2
    assert tracker.update([], 3.0) == []
    assert len(tracker.completed_tracks) == 2
    assert all(track.status == "COMPLETED" for track in tracker.completed_tracks)


def test_multiple_candidate_association_does_not_merge_neighbouring_tracks():
    tracker = RegionTracker(max_center_distance=50.0)
    tracker.update([_detection(20), _detection(100)], 0.0)
    tracks = tracker.update([_detection(102), _detection(18)], 1.0)

    assert len(tracks) == 2
    assert tracks[0].track_id == 1
    assert tracks[0].center[0] == 28.0
    assert tracks[1].track_id == 2
    assert tracks[1].center[0] == 112.0


def test_confirmation_completion_and_json_csv_export_consistency(tmp_path):
    engine = TemporalPersistenceEngine(_temporal_config())
    tracker = RegionTracker()
    event = _update_ratios(engine, tracker, [0.2, 0.2, 0.2])
    engine.complete(event.track_id, 2.0)

    exported = engine.export_events()
    assert exported[0]["state"] == "COMPLETED"
    assert exported[0]["was_confirmed"]
    assert exported[0]["confirmation_seconds"] == 2.0

    paths = EventStore(tmp_path).save(exported)
    with open(paths["json"], encoding="utf-8") as file:
        json_events = json.load(file)
    with open(paths["csv"], encoding="utf-8", newline="") as file:
        csv_events = list(csv.DictReader(file))

    assert len(json_events) == len(csv_events) == 1
    assert csv_events[0]["was_confirmed"] == "True"
    assert csv_events[0]["review_required"] == "False"


def test_missing_and_invalid_evidence_are_handled(tmp_path):
    store = EventStore(tmp_path / "events")
    paths = store.save([{"track_id": 1, "state": "COMPLETED"}])
    with open(paths["csv"], encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))

    assert row["evidence_count"] == "0"
    assert row["evidence_annotated_path"] == ""

    writer = EvidenceWriter(tmp_path / "evidence")
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Bounding-box"):
        writer.save(1, 0.0, frame, [0, 0, 0, 10], {})
    with pytest.raises(ValueError, match="Probability values"):
        writer.save(
            1,
            0.0,
            frame,
            [0, 0, 10, 10],
            {},
            np.full((20, 20), 2.0, dtype=np.float32),
        )


@pytest.mark.parametrize(
    "config",
    [
        {"max_motion_history": 3.5},
        {"max_motion_history": True},
        {"max_motion_history": 0},
        {"settling_motion_window_samples": 3.5},
        {"settling_motion_window_samples": True},
        {"min_motion_samples": 0},
    ],
)
def test_invalid_motion_history_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        TemporalPersistenceEngine(_temporal_config(**config))


def test_invalid_timestamps_are_rejected():
    tracker = RegionTracker()
    track = tracker.update([_detection()], 0.0)[0]
    with pytest.raises(ValueError, match="increase"):
        tracker.update([_detection()], 0.0)

    engine = TemporalPersistenceEngine(_temporal_config())
    engine.update(track, 0.0)
    with pytest.raises(ValueError, match="increase"):
        engine.update(track, 0.0)


def test_archived_track_final_miss_and_csv_fields(tmp_path):
    tracker = RegionTracker(max_missed_observations=0)
    config = TemporalConfig(
        min_persistence_seconds=1.0,
        min_observations=2,
        settling_min_seconds=0.0,
        min_motion_samples=1,
        settling_motion_window_samples=1,
        confidence_threshold=0.0,
    )
    engine = TemporalPersistenceEngine(config)
    track = tracker.update([_detection()], 0.0)[0]
    engine.update(track, 0.0)

    assert tracker.update([], 1.0) == []
    archived = tracker.completed_tracks[0]
    event = engine.update(archived, 1.0)
    engine.complete(archived.track_id, 1.0)

    assert event.total_missed_observations == 1
    exported = engine.export_events()
    paths = EventStore(tmp_path).save(exported)
    csv_text = open(paths["csv"], encoding="utf-8").read()
    header = csv_text.splitlines()[0]
    assert "settling_motion_median_ratio" in header
    assert "total_missed_observations" in header
    assert "was_confirmed" in header
