from pathlib import Path
import csv
import json
import os
import tempfile
from typing import Any


CSV_FIELDS = [
    "track_id",
    "state",
    "first_seen_seconds",
    "last_seen_seconds",
    "duration_seconds",
    "observations",
    "missed_observations",
    "total_missed_observations",
    "detection_consistency",
    "center_deviation_pixels",
    "bbox_stability",
    "motion_samples",
    "motion_median_ratio",
    "motion_p90_ratio",
    "settling_motion_samples",
    "settling_motion_median_ratio",
    "settling_motion_p90_ratio",
    "confidence",
    "confidence_kind",
    "boundary_candidate",
    "was_confirmed",
    "confirmation_seconds",
    "review_required",
    "post_confirmation_degraded",
    "degradation_first_seen_seconds",
    "post_confirmation_degradation_reasons",
    "evidence_count",
    "evidence_annotated_path",
]


class EventStore:
    """Persist temporal events as JSON and CSV."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, destination: Path, writer) -> None:
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=self.output_dir,
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                writer(temporary)

                temporary.flush()
                os.fsync(temporary.fileno())

            os.replace(temporary_path, destination)

        except Exception:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
            raise

    def save(self, events: list[dict[str, Any]]) -> dict[str, str]:
        if not isinstance(events, list):
            raise TypeError("events must be a list")

        for event in events:
            if not isinstance(event, dict):
                raise TypeError("Each event must be a dictionary")

            if "track_id" not in event or "state" not in event:
                raise ValueError("Event requires track_id and state")

        json_path = self.output_dir / "events.json"
        csv_path = self.output_dir / "events.csv"

        def write_json(file):
            json.dump(
                events,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        def write_csv(file):
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS,
                extrasaction="ignore",
            )

            writer.writeheader()

            for event in events:
                evidence_paths = event.get("evidence_paths", [])
                latest_evidence = (
                    evidence_paths[-1]
                    if isinstance(evidence_paths, list) and evidence_paths
                    else {}
                )

                if not isinstance(latest_evidence, dict):
                    latest_evidence = {}
                writer.writerow({
                    field: event.get(field)
                    for field in CSV_FIELDS
                } | {
                    "evidence_count": len(evidence_paths),
                    "evidence_annotated_path": latest_evidence.get(
                        "annotated"
                    ),
                })

        self._atomic_write(json_path, write_json)
        self._atomic_write(csv_path, write_csv)

        return {
            "json": str(json_path.resolve()),
            "csv": str(csv_path.resolve()),
        }

    def load(self) -> list[dict[str, Any]]:
        json_path = self.output_dir / "events.json"

        if not json_path.exists():
            return []

        with json_path.open("r", encoding="utf-8") as file:
            events = json.load(file)

        if not isinstance(events, list):
            raise ValueError("Stored events must be a JSON list")

        return events
