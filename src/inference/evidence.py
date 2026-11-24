from pathlib import Path
import json
import re
from typing import Any

import cv2
import numpy as np


class EvidenceWriter:
    """Save visual and metadata evidence for temporal events."""

    def __init__(self, output_dir: str | Path, jpeg_quality: int = 95):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        self.jpeg_quality = jpeg_quality

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a NumPy array")

        if frame.dtype != np.uint8:
            raise ValueError("Frame must have uint8 dtype")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must have shape HxWx3 (BGR)")

        if frame.size == 0:
            raise ValueError("Frame cannot be empty")

    @staticmethod
    def _safe_track_id(track_id: Any) -> str:
        value = str(track_id)

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            raise ValueError("Invalid track ID for evidence filename")

        return value

    def save(
        self,
        track_id: int,
        timestamp_seconds: float,
        frame: np.ndarray,
        bbox: list[int] | tuple[int, int, int, int],
        event_metadata: dict[str, Any],
        probability_map: np.ndarray | None = None,
    ) -> dict[str, str]:

        self._validate_frame(frame)

        timestamp = float(timestamp_seconds)

        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Timestamp must be finite and non-negative")

        if len(bbox) != 4:
            raise ValueError("bbox must contain x, y, width, height")

        x, y, width, height = [int(value) for value in bbox]

        if width <= 0 or height <= 0:
            raise ValueError("Bounding-box dimensions must be positive")

        image_height, image_width = frame.shape[:2]

        x1 = max(0, min(x, image_width))
        y1 = max(0, min(y, image_height))
        x2 = max(0, min(x + width, image_width))
        y2 = max(0, min(y + height, image_height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box does not overlap the frame")

        track_name = self._safe_track_id(track_id)

        event_dir = self.output_dir / f"track_{track_name}"
        event_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{timestamp:010.3f}".replace(".", "_")

        original_path = event_dir / f"{prefix}_original.jpg"
        annotated_path = event_dir / f"{prefix}_annotated.jpg"
        metadata_path = event_dir / f"{prefix}_metadata.json"

        original = frame.copy()
        annotated = frame.copy()

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (0, 255, 0),
            2,
        )

        label = f"Track {track_name} | {timestamp:.2f}s"

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        if not cv2.imwrite(str(original_path), original, params):
            raise OSError(f"Failed to write {original_path}")

        if not cv2.imwrite(str(annotated_path), annotated, params):
            raise OSError(f"Failed to write {annotated_path}")

        paths = {
            "original": str(original_path.resolve()),
            "annotated": str(annotated_path.resolve()),
        }

        if probability_map is not None:
            probability_map = np.asarray(probability_map)

            if probability_map.ndim != 2:
                raise ValueError("Probability map must be two-dimensional")

            if not np.isfinite(probability_map).all():
                raise ValueError("Probability map contains non-finite values")

            if probability_map.min() < 0 or probability_map.max() > 1:
                raise ValueError("Probability values must be in [0, 1]")

            probability_image = (
                probability_map * 255
            ).astype(np.uint8)

            probability_path = event_dir / f"{prefix}_probability.png"

            if not cv2.imwrite(str(probability_path), probability_image):
                raise OSError(f"Failed to write {probability_path}")

            paths["probability"] = str(probability_path.resolve())

        metadata = {
            "track_id": track_name,
            "timestamp_seconds": timestamp,
            "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
            "frame_width": image_width,
            "frame_height": image_height,
            "event": event_metadata,
            "files": {
                key: Path(value).name
                for key, value in paths.items()
            },
        }

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        paths["metadata"] = str(metadata_path.resolve())

        return paths
