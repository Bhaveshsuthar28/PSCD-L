"""Adapter integrating Task 2 VideoRunner with the backend worker."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

from backend.app.core.config import Settings
from backend.app.integrations.trinetra_engine import ManagedInferenceEngine
from src.inference.candidates import CandidateConfig
from src.inference.temporal_engine import TemporalConfig
from src.inference.video_runner import VideoRunner


class VideoRunnerAdapter:
    """Builds and runs the existing Task 2 VideoRunner without duplicating logic."""

    def __init__(
        self,
        settings: Settings,
        managed_engine: ManagedInferenceEngine,
    ) -> None:
        self.settings = settings
        self.managed_engine = managed_engine

    @staticmethod
    def build_configs(config_snapshot: dict[str, Any] | None) -> tuple[CandidateConfig, TemporalConfig]:
        snapshot = config_snapshot or {}
        cand_overrides = snapshot.get("candidate_config") or {}
        temp_overrides = snapshot.get("temporal_config") or {}
        return CandidateConfig(**cand_overrides), TemporalConfig(**temp_overrides)

    def execute(
        self,
        *,
        video_path: Path,
        output_dir: Path,
        reference_seconds: float,
        start_seconds: float,
        end_seconds: float | None,
        sampling_interval_seconds: float,
        configuration_snapshot: dict[str, Any] | None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        candidate_cfg, temporal_cfg = self.build_configs(configuration_snapshot)
        engine = self.managed_engine.acquire_engine()

        runner = VideoRunner(
            checkpoint_path=self.settings.resolved_checkpoint_path,
            output_dir=output_dir,
            sample_interval=sampling_interval_seconds,
            reference_seconds=reference_seconds,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            candidate_config=candidate_cfg,
            temporal_config=temporal_cfg,
            inference_engine=engine,
            close_engine_on_finish=not self.settings.keep_model_loaded_across_jobs,
            verbose=False,
            strict_evidence=False,
        )

        t0 = time.perf_counter()
        summary = runner.run(
            video_path,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            event_callback=event_callback,
        )
        elapsed = round(time.perf_counter() - t0, 4)

        events = runner.temporal_engine.export_events()
        return {
            "summary": summary,
            "events": events,
            "evidence_errors": list(runner.evidence_errors),
            "elapsed_seconds": elapsed,
        }
