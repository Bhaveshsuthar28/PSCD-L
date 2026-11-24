from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.inference.engine import TrinetraInferenceEngine
from src.inference.candidates import (
    CandidateConfig,
    extract_candidates,
)
from src.inference.tracker import RegionTracker
from src.inference.motion_analyzer import MotionAnalyzer
from src.inference.temporal_engine import (
    TemporalConfig,
    TemporalPersistenceEngine,
)
from src.inference.event_store import EventStore
from src.inference.evidence import EvidenceWriter


class VideoProcessingCancelled(RuntimeError):
    """Raised when cooperative cancellation is requested during video processing."""


class VideoRunner:
    """End-to-end TRINETRA temporal scene-change video pipeline."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        output_dir: str | Path = "outputs/temporal_events",
        sample_interval: float = 1.0,
        reference_seconds: float = 0.0,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        candidate_config: CandidateConfig | None = None,
        temporal_config: TemporalConfig | None = None,
        inference_engine: Any | None = None,
        close_engine_on_finish: bool = True,
        verbose: bool = True,
        strict_evidence: bool = True,
    ):
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")

        if reference_seconds < 0 or start_seconds < 0:
            raise ValueError("Reference and start times cannot be negative")

        if end_seconds is not None and end_seconds <= start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_interval = float(sample_interval)
        self.reference_seconds = float(reference_seconds)
        self.start_seconds = float(start_seconds)
        self.end_seconds = end_seconds

        self.candidate_config = candidate_config or CandidateConfig()
        self.temporal_config = temporal_config or TemporalConfig()

        self.tracker = RegionTracker()
        self.motion_analyzer = MotionAnalyzer()
        self.temporal_engine = TemporalPersistenceEngine(
            self.temporal_config
        )

        self.event_store = EventStore(
            self.output_dir / "events"
        )

        self.evidence_writer = EvidenceWriter(
            self.output_dir / "evidence"
        )

        if inference_engine is not None:
            self.inference_engine = inference_engine
        elif checkpoint_path is None:
            self.inference_engine = TrinetraInferenceEngine()
        else:
            self.inference_engine = TrinetraInferenceEngine(
                checkpoint_path=checkpoint_path
            )

        self.close_engine_on_finish = bool(close_engine_on_finish)
        self.verbose = bool(verbose)
        self.strict_evidence = bool(strict_evidence)
        self.saved_evidence_track_ids: set[int] = set()
        self.evidence_errors: list[dict[str, Any]] = []

    @staticmethod
    def _read_frame(
        capture: cv2.VideoCapture,
        frame_index: int,
    ) -> np.ndarray:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = capture.read()

        if not success or frame is None:
            raise RuntimeError(
                f"Unable to read video frame {frame_index}"
            )

        return frame

    @staticmethod
    def _validate_probability_map(
        probability_map: np.ndarray,
        frame: np.ndarray,
    ) -> np.ndarray:
        probability_map = np.asarray(probability_map)

        if probability_map.ndim == 3 and probability_map.shape[0] == 1:
            probability_map = probability_map[0]

        if probability_map.ndim != 2:
            raise ValueError(
                "Inference output must be a 2D probability map; "
                f"received shape {probability_map.shape}"
            )

        if probability_map.shape != frame.shape[:2]:
            raise ValueError(
                "Inference output dimensions do not match video frame: "
                f"{probability_map.shape} vs {frame.shape[:2]}"
            )

        if not np.isfinite(probability_map).all():
            raise ValueError("Inference returned non-finite values")

        if probability_map.min() < 0 or probability_map.max() > 1:
            raise ValueError(
                "Inference output must contain probabilities in [0, 1]"
            )

        return probability_map.astype(np.float32, copy=False)

    def _save_confirmed_evidence(
        self,
        track: Any,
        event: Any,
        timestamp: float,
        frame: np.ndarray,
        probability_map: np.ndarray,
    ) -> None:
        track_id = int(track.track_id)

        if track_id in self.saved_evidence_track_ids:
            return

        metadata = {
            "state": event.state,
            "confidence": event.confidence,
            "confidence_kind": "heuristic_score_not_probability",
            "duration_seconds": event.duration_seconds,
            "observations": event.observations,
            "detection_consistency": event.detection_consistency,
            "center_deviation_pixels": (
                event.center_deviation_pixels
            ),
            "bbox_stability": event.bbox_stability,
            "motion_median_ratio": event.motion_median_ratio,
            "motion_p90_ratio": event.motion_p90_ratio,
            "settling_motion_samples": event.settling_motion_samples,
            "settling_motion_median_ratio": (
                event.settling_motion_median_ratio
            ),
            "settling_motion_p90_ratio": (
                event.settling_motion_p90_ratio
            ),
            "boundary_candidate": event.boundary_candidate,
            "motion_identity_note": (
                "Motion is settling evidence and does not prove object "
                "identity."
            ),
        }

        try:
            evidence_paths = self.evidence_writer.save(
                track_id=track_id,
                timestamp_seconds=timestamp,
                frame=frame,
                bbox=track.bbox,
                event_metadata=metadata,
                probability_map=probability_map,
            )
            event.evidence_paths.append(evidence_paths)
            self.saved_evidence_track_ids.add(track_id)
        except Exception as exc:
            if self.strict_evidence:
                raise
            self.evidence_errors.append(
                {
                    "track_id": track_id,
                    "timestamp_seconds": float(timestamp),
                    "error": str(exc),
                }
            )

    def _complete_archived_tracks(self, timestamp: float) -> None:
        for track in self.tracker.get_all_tracks():
            if track.status != "COMPLETED":
                continue

            track_id = int(track.track_id)

            if track_id not in self.temporal_engine.events:
                continue

            event = self.temporal_engine.get_event(track_id)

            if event.state != "COMPLETED":
                # The tracker archived this track after recording its final
                # missed sample. Persist it before closure so exported
                # consistency and missed-count evidence remains exact.
                self.temporal_engine.update(track, timestamp)
                completion_time = max(
                    timestamp,
                    event.last_seen_seconds,
                )

                self.temporal_engine.complete(
                    track_id,
                    completion_time,
                )

    def run(
        self,
        video_path: str | Path,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        reference_capture = None

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = float(capture.get(cv2.CAP_PROP_FPS))

        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise RuntimeError("Video has invalid FPS")

        duration = frame_count / fps

        if self.reference_seconds >= duration:
            capture.release()
            raise ValueError("Reference time exceeds video duration")

        if self.start_seconds >= duration:
            capture.release()
            raise ValueError("Start time exceeds video duration")

        effective_end = (
            min(self.end_seconds, duration)
            if self.end_seconds is not None
            else duration
        )

        if effective_end <= self.start_seconds:
            capture.release()
            raise ValueError("Invalid processing time window")

        reference_capture = cv2.VideoCapture(str(video_path))

        if not reference_capture.isOpened():
            capture.release()
            reference_capture.release()
            raise RuntimeError("Unable to open reference video capture")

        processed_frames = 0
        total_candidates = 0
        previous_sampled_frame = None
        notified_confirmed: set[int] = set()
        notified_review: set[int] = set()

        try:
            if cancel_check is not None and cancel_check():
                raise VideoProcessingCancelled("Processing cancelled before reference extraction.")

            reference_index = int(
                round(self.reference_seconds * fps)
            )

            reference_frame = self._read_frame(
                reference_capture,
                reference_index,
            )

            self.inference_engine.set_reference(reference_frame)

            start_index = int(
                round(self.start_seconds * fps)
            )

            end_index = min(
                frame_count - 1,
                int(round(effective_end * fps)),
            )

            sample_stride = max(
                1,
                int(round(self.sample_interval * fps)),
            )

            expected_total_samples = (
                ((end_index - start_index) // sample_stride) + 1
                if end_index >= start_index
                else 0
            )

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                start_index,
            )

            frame_index = start_index

            if self.verbose:
                print("\nTRINETRA VIDEO PROCESSING")
                print("=" * 65)
                print("Video:", video_path)
                print(f"FPS: {fps:.2f}")
                print(f"Duration: {duration:.2f}s")
                print(f"Reference: {self.reference_seconds:.2f}s")
                print(f"Processing: {self.start_seconds:.2f}s–{effective_end:.2f}s")
                print(f"Sampling interval: {self.sample_interval:.2f}s")

            while frame_index <= end_index:
                success, frame = capture.read()

                if not success or frame is None:
                    break

                should_process = (
                    (frame_index - start_index) % sample_stride == 0
                )

                if not should_process:
                    frame_index += 1
                    continue

                if cancel_check is not None and cancel_check():
                    raise VideoProcessingCancelled("Processing cancelled during frame loop.")

                timestamp = frame_index / fps

                probability_map = self.inference_engine.predict(frame)

                probability_map = self._validate_probability_map(
                    probability_map,
                    frame,
                )

                mask, candidates = extract_candidates(
                    probability_map,
                    self.candidate_config,
                )

                updated_tracks = self.tracker.update(
                    candidates,
                    timestamp,
                )

                total_candidates += len(candidates)

                for track in updated_tracks:
                    track_id = int(track.track_id)

                    motion_result = None

                    if (
                        previous_sampled_frame is not None
                        and track.observed_in_latest_update
                        and track.observed_in_previous_update
                        and len(track.history) >= 2
                    ):
                        motion_result = (
                            self.motion_analyzer.analyze_pair(
                                previous_sampled_frame,
                                frame,
                                track.bbox,
                            )
                        )

                    event = self.temporal_engine.update(
                        track,
                        timestamp,
                        motion_result,
                    )

                    if event.state == "CONFIRMED":
                        self._save_confirmed_evidence(
                            track,
                            event,
                            timestamp,
                            frame,
                            probability_map,
                        )
                        if event.was_confirmed and track_id not in notified_confirmed:
                            notified_confirmed.add(track_id)
                            if event_callback is not None:
                                event_callback({
                                    "type": "event_confirmed",
                                    "track_id": track_id,
                                    "timestamp_seconds": timestamp,
                                    "confidence": event.confidence,
                                    "bbox_xywh": list(track.bbox),
                                })

                    if event.review_required and track_id not in notified_review:
                        notified_review.add(track_id)
                        if event_callback is not None:
                            event_callback({
                                "type": "review_required",
                                "track_id": track_id,
                                "timestamp_seconds": timestamp,
                                "reasons": list(event.post_confirmation_degradation_reasons),
                            })

                self._complete_archived_tracks(timestamp)

                processed_frames += 1

                confirmed_count = sum(
                    event.was_confirmed
                    for event in self.temporal_engine.get_all_events()
                )
                review_count = sum(
                    event.review_required
                    for event in self.temporal_engine.get_all_events()
                )

                if progress_callback is not None:
                    progress_callback({
                        "stage": "PROCESSING_FRAMES",
                        "processed_frames": processed_frames,
                        "total_frames": expected_total_samples,
                        "timestamp_seconds": timestamp,
                        "total_candidates": total_candidates,
                        "active_tracks": len(self.tracker.tracks),
                        "confirmed_events": confirmed_count,
                        "review_required_events": review_count,
                    })

                if self.verbose and processed_frames % 10 == 0:
                    confirmed = sum(
                        event.state == "CONFIRMED"
                        for event in self.temporal_engine.get_all_events()
                    )

                    print(
                        f"Processed: {processed_frames} | "
                        f"Candidates: {total_candidates} | "
                        f"Tracks: {len(self.temporal_engine.events)} | "
                        f"Confirmed: {confirmed} | "
                        f"Time: {timestamp:.2f}s"
                    )

                previous_sampled_frame = frame.copy()
                frame_index += 1

            final_timestamp = min(
                effective_end,
                max(
                    self.start_seconds,
                    (frame_index - 1) / fps,
                ),
            )

            # Close any remaining active temporal events.
            for track in self.tracker.get_all_tracks():
                track_id = int(track.track_id)

                if track_id not in self.temporal_engine.events:
                    continue

                event = self.temporal_engine.get_event(track_id)

                if event.state != "COMPLETED":
                    self.temporal_engine.complete(
                        track_id,
                        max(final_timestamp, event.last_seen_seconds),
                    )

            events = self.temporal_engine.export_events()

            paths = self.event_store.save(events)

            confirmed_events = [
                event for event in events
                if event["was_confirmed"]
            ]

            review_required_events = [
                event for event in confirmed_events
                if event["review_required"]
            ]

            summary = {
                "video": str(video_path.resolve()),
                "fps": fps,
                "video_duration_seconds": duration,
                "reference_seconds": self.reference_seconds,
                "processing_start_seconds": self.start_seconds,
                "processing_end_seconds": effective_end,
                "sample_interval_seconds": self.sample_interval,
                "candidate_config": asdict(self.candidate_config),
                "temporal_config": asdict(self.temporal_config),
                "confidence_note": (
                    "confidence is a heuristic score, not a calibrated "
                    "probability; motion supports settling assessment and "
                    "does not establish object identity"
                ),
                "processed_frames": processed_frames,
                "total_candidates": total_candidates,
                "total_tracks": len(events),
                "confirmed_event_count": len(confirmed_events),
                "review_required_event_count": len(
                    review_required_events
                ),
                "non_review_confirmed_event_count": (
                    len(confirmed_events) - len(review_required_events)
                ),
                "evidence_track_ids": sorted(
                    self.saved_evidence_track_ids
                ),
                "evidence_errors": list(self.evidence_errors),
                "event_files": paths,
            }

            summary_path = self.output_dir / "run_summary.json"

            with summary_path.open("w", encoding="utf-8") as file:
                json.dump(
                    summary,
                    file,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )

            summary["summary_file"] = str(summary_path.resolve())

            if self.verbose:
                print("\nVIDEO PROCESSING COMPLETE")
                print("=" * 65)
                print(json.dumps(summary, indent=2))

            return summary

        finally:
            capture.release()

            if reference_capture is not None:
                reference_capture.release()

            try:
                if self.close_engine_on_finish:
                    self.inference_engine.close()
                else:
                    self.inference_engine.clear_reference()
            except Exception as error:
                if self.verbose:
                    print("Inference cleanup warning:", error)


def main():
    parser = argparse.ArgumentParser(
        description="TRINETRA temporal scene-change video runner"
    )

    parser.add_argument(
        "--video",
        default="Dataset/ABODA/video10.avi",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--output",
        default="outputs/temporal_events/video10",
    )

    parser.add_argument(
        "--reference",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--start",
        type=float,
        default=35.0,
    )

    parser.add_argument(
        "--end",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    runner = VideoRunner(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        sample_interval=args.interval,
        reference_seconds=args.reference,
        start_seconds=args.start,
        end_seconds=args.end,
    )

    runner.run(args.video)


if __name__ == "__main__":
    main()
