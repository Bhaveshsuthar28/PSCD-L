from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math


def calculate_iou(bbox_a: list[int], bbox_b: list[int]) -> float:
    """Calculate IoU for xywh bounding boxes."""
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection

    return float(intersection / union) if union > 0 else 0.0


def calculate_center_distance(
    center_a: list[float],
    center_b: list[float],
) -> float:
    """Calculate Euclidean distance between two centers."""
    return math.hypot(
        center_a[0] - center_b[0],
        center_a[1] - center_b[1],
    )


@dataclass
class Track:
    track_id: int
    first_seen_seconds: float
    last_seen_seconds: float
    bbox: list[int]
    center: list[float]
    area: int
    observations: int = 1
    missed_observations: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    missed_history: list[float] = field(default_factory=list)
    status: str = "ACTIVE"
    observed_in_latest_update: bool = True
    observed_in_previous_update: bool = False

    def update(
        self,
        detection: dict[str, Any],
        timestamp_seconds: float,
    ) -> None:
        self.bbox = list(detection["bbox"])
        self.center = list(detection["center"])
        self.area = int(detection["area"])
        self.last_seen_seconds = float(timestamp_seconds)
        self.observations += 1
        self.missed_observations = 0
        self.status = "ACTIVE"
        self.observed_in_latest_update = True

        self.history.append({
            "timestamp_seconds": float(timestamp_seconds),
            "bbox": list(self.bbox),
            "center": list(self.center),
            "area": self.area,
            "mean_probability": float(
                detection.get("mean_probability", 0.0)
            ),
            "max_probability": float(
                detection.get("max_probability", 0.0)
            ),
            "boundary": bool(detection.get("boundary", False)),
        })

    def record_missed_observation(
        self,
        timestamp_seconds: float,
    ) -> None:
        self.missed_observations += 1
        self.missed_history.append(float(timestamp_seconds))
        self.observed_in_latest_update = False


class RegionTracker:
    """
    Associate candidate regions across sequential observations.

    Active tracks remain in self.tracks.
    Expired tracks are retained in self.completed_tracks.
    """

    def __init__(
        self,
        max_center_distance: float = 60.0,
        min_iou: float = 0.05,
        max_missed_observations: int = 2,
    ):
        if max_center_distance <= 0:
            raise ValueError("max_center_distance must be positive.")

        if not 0 <= min_iou <= 1:
            raise ValueError("min_iou must be between 0 and 1.")

        if max_missed_observations < 0:
            raise ValueError("max_missed_observations cannot be negative.")

        self.max_center_distance = float(max_center_distance)
        self.min_iou = float(min_iou)
        self.max_missed_observations = int(max_missed_observations)

        self.tracks: list[Track] = []
        self.completed_tracks: list[Track] = []
        self.next_track_id = 1
        self.last_timestamp_seconds: float | None = None

    @staticmethod
    def _validate_detection(detection: dict[str, Any]) -> None:
        required = ("bbox", "center", "area")

        for key in required:
            if key not in detection:
                raise ValueError(f"Detection is missing '{key}'.")

        if len(detection["bbox"]) != 4:
            raise ValueError("Detection bbox must contain four values.")

        if len(detection["center"]) != 2:
            raise ValueError("Detection center must contain two values.")

        if int(detection["area"]) <= 0:
            raise ValueError("Detection area must be positive.")

    def update(
        self,
        detections: list[dict[str, Any]],
        timestamp_seconds: float,
    ) -> list[Track]:

        timestamp_seconds = float(timestamp_seconds)

        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("Timestamp must be finite and non-negative.")

        if (
            self.last_timestamp_seconds is not None
            and timestamp_seconds <= self.last_timestamp_seconds
        ):
            raise ValueError("Timestamps must increase globally.")

        for detection in detections:
            self._validate_detection(detection)

        self.last_timestamp_seconds = timestamp_seconds

        # Retain whether the track was observed on the immediately preceding
        # sampling instant. The runner uses this to avoid deriving optical-flow
        # evidence for a stale region after a missed detection.
        for track in self.tracks:
            track.observed_in_previous_update = (
                track.observed_in_latest_update
            )
            track.observed_in_latest_update = False

        pairs = []

        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                distance = calculate_center_distance(
                    track.center,
                    detection["center"],
                )

                iou = calculate_iou(
                    track.bbox,
                    detection["bbox"],
                )

                if (
                    distance > self.max_center_distance
                    or (
                        iou < self.min_iou
                        and distance > self.max_center_distance * 0.5
                    )
                ):
                    continue

                distance_cost = distance / self.max_center_distance
                cost = 0.65 * (1.0 - iou) + 0.35 * distance_cost

                pairs.append((cost, track_index, detection_index))

        pairs.sort(key=lambda item: item[0])

        assigned_tracks = set()
        assigned_detections = set()

        for _, track_index, detection_index in pairs:
            if track_index in assigned_tracks:
                continue

            if detection_index in assigned_detections:
                continue

            self.tracks[track_index].update(
                detections[detection_index],
                timestamp_seconds,
            )

            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)

        remaining_tracks = []

        for index, track in enumerate(self.tracks):
            if index not in assigned_tracks:
                track.record_missed_observation(timestamp_seconds)

            if track.missed_observations > self.max_missed_observations:
                track.status = "COMPLETED"
                self.completed_tracks.append(track)
            else:
                remaining_tracks.append(track)

        self.tracks = remaining_tracks

        for index, detection in enumerate(detections):
            if index in assigned_detections:
                continue

            track = Track(
                track_id=self.next_track_id,
                first_seen_seconds=timestamp_seconds,
                last_seen_seconds=timestamp_seconds,
                bbox=list(detection["bbox"]),
                center=list(detection["center"]),
                area=int(detection["area"]),
            )

            track.history.append({
                "timestamp_seconds": timestamp_seconds,
                "bbox": list(track.bbox),
                "center": list(track.center),
                "area": track.area,
                "mean_probability": float(
                    detection.get("mean_probability", 0.0)
                ),
                "max_probability": float(
                    detection.get("max_probability", 0.0)
                ),
                "boundary": bool(detection.get("boundary", False)),
            })

            self.tracks.append(track)
            self.next_track_id += 1

        return list(self.tracks)

    def get_active_tracks(self) -> list[dict[str, Any]]:
        return [self._track_summary(track) for track in self.tracks]

    def get_completed_tracks(self) -> list[dict[str, Any]]:
        return [
            self._track_summary(track)
            for track in self.completed_tracks
        ]

    def get_all_tracks(self) -> list[Track]:
        return sorted(
            self.completed_tracks + self.tracks,
            key=lambda track: track.track_id,
        )

    @staticmethod
    def _track_summary(track: Track) -> dict[str, Any]:
        return {
            "track_id": track.track_id,
            "status": track.status,
            "first_seen_seconds": track.first_seen_seconds,
            "last_seen_seconds": track.last_seen_seconds,
            "duration_seconds": (
                track.last_seen_seconds - track.first_seen_seconds
            ),
            "bbox": list(track.bbox),
            "center": list(track.center),
            "area": track.area,
            "observations": track.observations,
            "missed_observations": track.missed_observations,
            "total_missed_observations": len(track.missed_history),
            "missed_history": list(track.missed_history),
            "observed_in_latest_update": track.observed_in_latest_update,
            "observed_in_previous_update": (
                track.observed_in_previous_update
            ),
            "history": list(track.history),
        }
