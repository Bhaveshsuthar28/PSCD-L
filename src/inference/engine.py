from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from src.inference.preprocessing import (
    preprocess_frame,
    restore_probability_map,
)
from src.lightning_module import TrinetraLightningModule
from src.models.backbone import DINOv2Backbone


EXPECTED_CHECKPOINT_SHA256 = (
    "790BC43661096E8F45B513E774C1BE1D"
    "6734D44093D3FD5AA5063A8DA6C90BF9"
)

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "training"
    / "checkpoints"
    / "trinetra-33-0.1482.ckpt"
)


class TrinetraInferenceEngine:

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ) -> None:

        self.checkpoint_path = Path(checkpoint_path)

        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.decoder = None
        self.backbone = None
        self.reference_features = None
        self.reference_dimensions = None
        self.loaded = False

        self.load()

    @staticmethod
    def calculate_sha256(path: Path) -> str:

        sha256 = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                sha256.update(chunk)

        return sha256.hexdigest().upper()

    def load(self) -> None:

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        actual_hash = self.calculate_sha256(
            self.checkpoint_path
        )

        if actual_hash != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(
                "Checkpoint SHA256 verification failed.\n"
                f"Expected: {EXPECTED_CHECKPOINT_SHA256}\n"
                f"Actual:   {actual_hash}"
            )

        print("Checkpoint SHA256 verified.")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if "state_dict" not in checkpoint:
            raise RuntimeError(
                "Checkpoint does not contain state_dict."
            )

        hyperparameters = dict(
            checkpoint.get("hyper_parameters", {})
        )

        self.decoder = TrinetraLightningModule(
            **hyperparameters
        )

        self.decoder.load_state_dict(
            checkpoint["state_dict"],
            strict=True,
        )

        self.decoder.to(self.device)
        self.decoder.eval()

        print("TRINETRA decoder loaded successfully.")

        self.backbone = DINOv2Backbone(
            model_name="facebook/dinov2-base",
            device=str(self.device),
        )

        self.backbone.eval()

        print("DINOv2 backbone loaded successfully.")

        self.loaded = True

        print(f"Device: {self.device}")

        if self.device.type == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(self.device),
            )

    def _validate_frame(self, frame_bgr: np.ndarray) -> None:

        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("Expected a NumPy video frame.")

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected BGR frame [H, W, 3], got {frame_bgr.shape}."
            )

        if frame_bgr.dtype != np.uint8:
            raise TypeError(
                f"Expected uint8 frame, got {frame_bgr.dtype}."
            )

        if not self.loaded:
            raise RuntimeError(
                "Inference engine has not loaded successfully."
            )

    @torch.inference_mode()
    def extract_features(
        self,
        frame_bgr: np.ndarray,
    ):

        self._validate_frame(frame_bgr)

        image_tensor, metadata = preprocess_frame(frame_bgr)

        features = self.backbone(image_tensor)

        if features.shape != (1, 768, 37, 37):
            raise RuntimeError(
                f"Unexpected DINOv2 feature shape: "
                f"{tuple(features.shape)}"
            )

        return features.float(), metadata

    @torch.inference_mode()
    def set_reference(
        self,
        reference_frame_bgr: np.ndarray,
    ) -> None:

        features, metadata = self.extract_features(
            reference_frame_bgr
        )

        self.reference_features = features.detach()

        self.reference_dimensions = (
            metadata.original_width,
            metadata.original_height,
        )

        print(
            "Reference frame cached:",
            self.reference_dimensions,
        )

    @torch.inference_mode()
    def predict(
        self,
        current_frame_bgr: np.ndarray,
    ) -> np.ndarray:

        if self.reference_features is None:
            raise RuntimeError(
                "Reference frame not set. "
                "Call set_reference() first."
            )

        self._validate_frame(current_frame_bgr)

        current_dimensions = (
            current_frame_bgr.shape[1],
            current_frame_bgr.shape[0],
        )

        if current_dimensions != self.reference_dimensions:
            raise ValueError(
                "Current frame dimensions differ from reference. "
                f"Reference: {self.reference_dimensions}, "
                f"Current: {current_dimensions}"
            )

        current_features, metadata = self.extract_features(
            current_frame_bgr
        )

        logits = self.decoder(
            self.reference_features,
            current_features,
        )

        if logits.shape != (1, 1, 512, 512):
            raise RuntimeError(
                f"Unexpected decoder output: {tuple(logits.shape)}"
            )

        probabilities = torch.sigmoid(logits.float())

        probability_map = restore_probability_map(
            probabilities,
            metadata,
        )

        probability_map = probability_map.clamp(0.0, 1.0)

        result = probability_map.cpu().numpy()

        if not np.isfinite(result).all():
            raise RuntimeError(
                "Inference produced NaN or infinite values."
            )

        return result

    def clear_reference(self) -> None:

        self.reference_features = None
        self.reference_dimensions = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def close(self) -> None:

        self.clear_reference()

        self.backbone = None
        self.decoder = None
        self.loaded = False

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
