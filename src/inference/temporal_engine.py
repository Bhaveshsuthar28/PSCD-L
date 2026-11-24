from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np


EVENT_STATES = (
    "CANDIDATE",
    "OBSERVING",
    "SETTLING",
    "CONFIRMED",
    "COMPLETED",
)


@dataclass
class TemporalConfig:
    min_persistence_seconds: float = 10.0
    min_observations: int = 6
    min_detection_consistency: float = 0.70
    max_center_deviation_pixels: float = 15.0
    min_bbox_stability: float = 0.50

    settling_min_seconds: float = 3.0
    min_motion_samples: int = 3
    max_settling_median_ratio: float = 1.0
    max_settling_p90_ratio: float = 1.5

    # None preserves statistics over the complete event lifetime. A positive
    # cap remains available for memory-constrained deployments.
    max_motion_history: int | None = None
    settling_motion_window_samples: int = 5
    confidence_threshold: float = 0.75


@dataclass
class TemporalEvent:
    track_id: int
    state: str
    first_seen_seconds: float
    last_seen_seconds: float

    observations: int = 0
    missed_observations: int = 0
    total_missed_observations: int = 0

    center_deviation_pixels: float = 0.0
    bbox_stability: float = 0.0
    detection_consistency: float = 0.0

    motion_samples: int = 0
    motion_median_ratio: float | None = None
    motion_p90_ratio: float | None = None

    settling_motion_samples: int = 0
    settling_motion_median_ratio: float | None = None
    settling_motion_p90_ratio: float | None = None

    confidence: float = 0.0
    boundary_candidate: bool = False
    was_confirmed: bool = False
    confirmation_seconds: float | None = None
    review_required: bool = False
    post_confirmation_degraded: bool = False
    degradation_first_seen_seconds: float | None = None
    post_confirmation_degradation_reasons: list[str] = field(
        default_factory=list
    )
    evidence_paths: list[dict[str, str]] = field(default_factory=list)

    evidence: list[dict[str, Any]] = field(default_factory=list)
    state_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.last_seen_seconds - self.first_seen_seconds)


class TemporalPersistenceEngine:
    """
    Converts RegionTracker tracks and optical-flow measurements
    into auditable temporal scene-change events.

    This engine does not classify semantic object categories.
    Motion is supporting evidence, not proof of object identity.
    """

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()
        self.events: dict[int, TemporalEvent] = {}
        self.motion_history: dict[int, list[float]] = {}
        self.last_timestamp_by_track: dict[int, float] = {}

        self._validate_config()

    def _validate_config(self) -> None:
        c = self.config

        if c.min_persistence_seconds <= 0:
            raise ValueError("min_persistence_seconds must be positive")

        if c.min_observations < 2:
            raise ValueError("min_observations must be at least 2")

        if not 0 < c.min_detection_consistency <= 1:
            raise ValueError("min_detection_consistency must be in (0, 1]")

        if c.max_center_deviation_pixels < 0:
            raise ValueError("max_center_deviation_pixels cannot be negative")

        if not 0 <= c.min_bbox_stability <= 1:
            raise ValueError("min_bbox_stability must be in [0, 1]")

        if c.settling_min_seconds < 0:
            raise ValueError("settling_min_seconds cannot be negative")

        if (
            isinstance(c.min_motion_samples, bool)
            or not isinstance(c.min_motion_samples, int)
            or c.min_motion_samples < 1
        ):
            raise ValueError("min_motion_samples must be positive")

        if c.max_settling_median_ratio <= 0:
            raise ValueError("max_settling_median_ratio must be positive")

        if c.max_settling_p90_ratio <= 0:
            raise ValueError("max_settling_p90_ratio must be positive")

        if not 0 <= c.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")

        if c.max_motion_history is not None:
            if (
                isinstance(c.max_motion_history, bool)
                or not isinstance(c.max_motion_history, int)
                or c.max_motion_history < 1
            ):
                raise ValueError(
                    "max_motion_history must be None or a positive integer"
                )

            if c.max_motion_history < c.min_motion_samples:
                raise ValueError(
                    "max_motion_history must cover min_motion_samples"
                )

        if (
            isinstance(c.settling_motion_window_samples, bool)
            or not isinstance(c.settling_motion_window_samples, int)
            or c.settling_motion_window_samples < c.min_motion_samples
        ):
            raise ValueError(
                "settling_motion_window_samples must cover min_motion_samples"
            )

        if (
            c.max_motion_history is not None
            and c.settling_motion_window_samples > c.max_motion_history
        ):
            raise ValueError(
                "settling_motion_window_samples cannot exceed max_motion_history"
            )

    @staticmethod
    def _bbox_iou(a: list[int], b: list[int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)

        intersection = max(0, right - left) * max(0, bottom - top)
        union = aw * ah + bw * bh - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _validate_timestamp(timestamp_seconds: float) -> float:
        value = float(timestamp_seconds)

        if not math.isfinite(value) or value < 0:
            raise ValueError("Timestamp must be finite and non-negative")

        return value

    def _transition(
        self,
        event: TemporalEvent,
        new_state: str,
        timestamp_seconds: float,
    ) -> None:
        if new_state not in EVENT_STATES:
            raise ValueError(f"Unknown event state: {new_state}")

        state_order = {
            "CANDIDATE": 0,
            "OBSERVING": 1,
            "SETTLING": 2,
            "CONFIRMED": 3,
            "COMPLETED": 4,
        }

        current_order = state_order[event.state]
        new_order = state_order[new_state]

        # Event lifecycle must never move backwards.
        if new_order <= current_order:
            return

        event.state = new_state

        if new_state == "CONFIRMED":
            event.was_confirmed = True
            event.confirmation_seconds = timestamp_seconds

        event.state_history.append({
            "timestamp_seconds": timestamp_seconds,
            "state": new_state,
        })
        
    def _calculate_spatial_evidence(self, track: Any) -> None:
        history = track.history

        centers = [
            item["center"]
            for item in history
            if "center" in item
        ]

        boxes = [
            item["bbox"]
            for item in history
            if "bbox" in item
        ]

        if centers:
            points = np.asarray(centers, dtype=np.float64)

            if not np.isfinite(points).all():
                raise ValueError("Track contains non-finite centers")

            centroid = points.mean(axis=0)

            deviations = np.linalg.norm(
                points - centroid,
                axis=1,
            )

            event = self.events[track.track_id]
            event.center_deviation_pixels = float(deviations.mean())

        if boxes:
            reference = boxes[len(boxes) // 2]

            ious = [
                self._bbox_iou(box, reference)
                for box in boxes
            ]

            self.events[track.track_id].bbox_stability = float(
                np.mean(ious)
            )

    def _calculate_detection_consistency(self, track: Any) -> float:
        observations = int(track.observations)
        missed = len(track.missed_history)

        total = observations + missed

        return observations / total if total else 0.0

    def _add_motion(
        self,
        track_id: int,
        motion_result: dict[str, Any] | None,
    ) -> None:
        if motion_result is None:
            return

        ratio = motion_result.get("region_background_ratio")

        if ratio is None:
            return

        ratio = float(ratio)

        if not math.isfinite(ratio) or ratio < 0:
            return

        history = self.motion_history.setdefault(track_id, [])
        history.append(ratio)

        if (
            self.config.max_motion_history is not None
            and len(history) > self.config.max_motion_history
        ):
            del history[:-self.config.max_motion_history]

    def _calculate_confidence(self, event: TemporalEvent) -> float:
        c = self.config

        persistence = min(
            event.duration_seconds / c.min_persistence_seconds,
            1.0,
        )

        observation_score = min(
            event.observations / c.min_observations,
            1.0,
        )

        detection_score = event.detection_consistency

        spatial_score = (
            0.5 * min(
                event.center_deviation_pixels
                <= c.max_center_deviation_pixels
                and event.observations >= 2,
                1.0,
            )
            + 0.5 * event.bbox_stability
        )

        motion_score = 0.0

        if event.settling_motion_samples >= c.min_motion_samples:
            median = event.settling_motion_median_ratio
            p90 = event.settling_motion_p90_ratio

            median_score = (
                1.0 if median <= c.max_settling_median_ratio
                else max(
                    0.0,
                    1.0 - median / (
                        2 * c.max_settling_median_ratio
                    ),
                )
            )

            p90_score = (
                1.0 if p90 <= c.max_settling_p90_ratio
                else max(
                    0.0,
                    1.0 - p90 / (
                        2 * c.max_settling_p90_ratio
                    ),
                )
            )

            motion_score = 0.5 * median_score + 0.5 * p90_score

        confidence = (
            0.20 * persistence
            + 0.15 * observation_score
            + 0.20 * detection_score
            + 0.20 * spatial_score
            + 0.25 * motion_score
        )

        return round(float(np.clip(confidence, 0.0, 1.0)), 4)

    def _record_post_confirmation_quality(
        self,
        event: TemporalEvent,
        timestamp: float,
        settling_ready: bool,
    ) -> None:
        """Record deterioration after confirmation without revising history."""
        if (
            not event.was_confirmed
            or event.confirmation_seconds is None
            or timestamp <= event.confirmation_seconds
        ):
            return

        reasons = []

        if (
            event.detection_consistency
            < self.config.min_detection_consistency
        ):
            reasons.append("detection_consistency")

        if (
            event.center_deviation_pixels
            > self.config.max_center_deviation_pixels
        ):
            reasons.append("center_deviation")

        if event.bbox_stability < self.config.min_bbox_stability:
            reasons.append("bbox_stability")

        if not settling_ready:
            reasons.append("recent_motion_settling")

        if event.confidence < self.config.confidence_threshold:
            reasons.append("heuristic_confidence")

        if not reasons:
            return

        event.review_required = True
        event.post_confirmation_degraded = True

        if event.degradation_first_seen_seconds is None:
            event.degradation_first_seen_seconds = timestamp

        for reason in reasons:
            if reason not in event.post_confirmation_degradation_reasons:
                event.post_confirmation_degradation_reasons.append(reason)

    def update(
        self,
        track: Any,
        timestamp_seconds: float,
        motion_result: dict[str, Any] | None = None,
    ) -> TemporalEvent:
        timestamp = self._validate_timestamp(timestamp_seconds)

        track_id = int(track.track_id)

        previous_timestamp = self.last_timestamp_by_track.get(track_id)

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(
                f"Timestamps must increase strictly for track {track_id}"
            )

        self.last_timestamp_by_track[track_id] = timestamp

        if track_id not in self.events:
            event = TemporalEvent(
                track_id=track_id,
                state="CANDIDATE",
                first_seen_seconds=float(track.first_seen_seconds),
                last_seen_seconds=float(track.last_seen_seconds),
            )

            event.state_history.append({
                "timestamp_seconds": timestamp,
                "state": "CANDIDATE",
            })

            self.events[track_id] = event
            self.motion_history[track_id] = []

        event = self.events[track_id]

        if event.state == "COMPLETED":
            raise ValueError("Cannot update a completed event")

        event.last_seen_seconds = float(track.last_seen_seconds)
        event.observations = int(track.observations)
        event.missed_observations = int(track.missed_observations)
        event.total_missed_observations = len(track.missed_history)
        event.boundary_candidate = any(
            bool(item.get("boundary", False))
            for item in track.history
        )

        self._add_motion(track_id, motion_result)

        ratios = self.motion_history[track_id]

        event.motion_samples = len(ratios)

        if ratios:
            # Preserve complete event history for audit and reporting.
            event.motion_median_ratio = float(np.median(ratios))
            event.motion_p90_ratio = float(np.percentile(ratios, 90))

            # Evaluate settling from only the most recent samples.
            recent_ratios = ratios[-self.config.settling_motion_window_samples:]
            event.settling_motion_samples = len(recent_ratios)
            event.settling_motion_median_ratio = float(
                np.median(recent_ratios)
            )
            event.settling_motion_p90_ratio = float(
                np.percentile(recent_ratios, 90)
            )

        event.detection_consistency = (
            self._calculate_detection_consistency(track)
        )

        self._calculate_spatial_evidence(track)

        event.confidence = self._calculate_confidence(event)

        event.evidence.append({
            "timestamp_seconds": timestamp,
            "bbox_xywh": list(track.bbox),
            "observed_in_latest_update": bool(
                track.observed_in_latest_update
            ),
            "observations": event.observations,
            "missed_observations": event.missed_observations,
            "total_missed_observations": (
                event.total_missed_observations
            ),
            "detection_consistency": round(
                event.detection_consistency, 4
            ),
            "center_deviation_pixels": round(
                event.center_deviation_pixels, 3
            ),
            "bbox_stability": round(event.bbox_stability, 4),
            "motion_ratio": (
                round(ratios[-1], 4) if ratios else None
            ),
            "motion_history_median_ratio": event.motion_median_ratio,
            "motion_history_p90_ratio": event.motion_p90_ratio,
            "settling_motion_samples": event.settling_motion_samples,
            "settling_motion_median_ratio": (
                event.settling_motion_median_ratio
            ),
            "settling_motion_p90_ratio": (
                event.settling_motion_p90_ratio
            ),
            "confidence": event.confidence,
            "confidence_kind": "heuristic_score_not_probability",
        })

        duration = event.duration_seconds

        if event.observations >= 2:
            self._transition(event, "OBSERVING", timestamp)

        settling_ready = (
            duration >= self.config.settling_min_seconds
            and event.settling_motion_samples
                >= self.config.min_motion_samples
            and event.settling_motion_median_ratio is not None
            and event.settling_motion_p90_ratio is not None
            and event.settling_motion_median_ratio
                <= self.config.max_settling_median_ratio
            and event.settling_motion_p90_ratio
                <= self.config.max_settling_p90_ratio
        )

        if settling_ready:
            self._transition(event, "SETTLING", timestamp)

        confirmation_ready = (
            duration >= self.config.min_persistence_seconds
            and event.observations >= self.config.min_observations
            and event.detection_consistency
                >= self.config.min_detection_consistency
            and event.center_deviation_pixels
                <= self.config.max_center_deviation_pixels
            and event.bbox_stability
                >= self.config.min_bbox_stability
            and settling_ready
            and event.confidence >= self.config.confidence_threshold
        )

        if confirmation_ready:
            self._transition(event, "CONFIRMED", timestamp)

        self._record_post_confirmation_quality(
            event,
            timestamp,
            settling_ready,
        )

        event.evidence[-1].update({
            "review_required": event.review_required,
            "post_confirmation_degraded": (
                event.post_confirmation_degraded
            ),
            "post_confirmation_degradation_reasons": list(
                event.post_confirmation_degradation_reasons
            ),
        })

        return event

    def complete(
        self,
        track_id: int,
        timestamp_seconds: float,
    ) -> TemporalEvent:
        timestamp = self._validate_timestamp(timestamp_seconds)

        if track_id not in self.events:
            raise KeyError(f"Unknown track ID: {track_id}")

        event = self.events[track_id]

        if timestamp < event.last_seen_seconds:
            raise ValueError("Completion timestamp cannot precede last observation")

        self._transition(event, "COMPLETED", timestamp)

        event.evidence.append({
            "timestamp_seconds": timestamp,
            "completion": True,
            "final_state_before_completion": (
                event.state_history[-2]["state"]
                if len(event.state_history) >= 2
                else "CANDIDATE"
            ),
            "confidence": event.confidence,
        })

        return event

    def get_event(self, track_id: int) -> TemporalEvent:
        return self.events[track_id]

    def get_all_events(self) -> list[TemporalEvent]:
        return list(self.events.values())

    def export_events(self) -> list[dict[str, Any]]:
        results = []

        for event in self.events.values():
            results.append({
                "track_id": event.track_id,
                "state": event.state,
                "first_seen_seconds": event.first_seen_seconds,
                "last_seen_seconds": event.last_seen_seconds,
                "duration_seconds": event.duration_seconds,
                "observations": event.observations,
                "missed_observations": event.missed_observations,
                "total_missed_observations": (
                    event.total_missed_observations
                ),
                "detection_consistency": event.detection_consistency,
                "center_deviation_pixels": event.center_deviation_pixels,
                "bbox_stability": event.bbox_stability,
                "motion_samples": event.motion_samples,
                "motion_median_ratio": event.motion_median_ratio,
                "motion_p90_ratio": event.motion_p90_ratio,
                "settling_motion_samples": event.settling_motion_samples,
                "settling_motion_median_ratio": (
                    event.settling_motion_median_ratio
                ),
                "settling_motion_p90_ratio": (
                    event.settling_motion_p90_ratio
                ),
                "confidence": event.confidence,
                "confidence_kind": "heuristic_score_not_probability",
                "boundary_candidate": event.boundary_candidate,
                "was_confirmed": event.was_confirmed,
                "confirmation_seconds": event.confirmation_seconds,
                "review_required": event.review_required,
                "post_confirmation_degraded": (
                    event.post_confirmation_degraded
                ),
                "degradation_first_seen_seconds": (
                    event.degradation_first_seen_seconds
                ),
                "post_confirmation_degradation_reasons": list(
                    event.post_confirmation_degradation_reasons
                ),
                "evidence_paths": event.evidence_paths,
                "evidence": event.evidence,
                "state_history": event.state_history,
            })

        return results
