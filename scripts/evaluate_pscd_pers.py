
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import DINOv2Backbone
from src.lightning_module import TrinetraLightningModule


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DIRECTIONS = ("B", "F", "L", "R")

IMAGE_SIZE = 512
MODEL_SIZE = 518
PATCH_SIZE = 14

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MASK_THRESHOLD = 128
PREDICTION_THRESHOLD = 0.5

DEFAULT_INDEX = PROJECT_ROOT / "outputs/index/pscd_pers.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Dataset/PSCD_pers"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/training/checkpoints/trinetra-33-0.1482.ckpt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/evaluation/pscd_pers"


# ------------------------------------------------------------
# PATH RESOLUTION
# ------------------------------------------------------------

def resolve_dataset_path(
    stored_path: str,
    dataset_root: Path,
) -> Path:

    candidate = Path(stored_path)

    if candidate.is_file():
        return candidate

    windows_path = PureWindowsPath(stored_path)
    parts = windows_path.parts

    try:
        dataset_position = parts.index("PSCD_pers")
    except ValueError:
        dataset_position = -1

    if dataset_position >= 0:
        relative_parts = parts[dataset_position + 1:]
        candidate = dataset_root.joinpath(*relative_parts)

        if candidate.is_file():
            return candidate

    candidate = dataset_root / windows_path.name

    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Cannot resolve dataset file: {stored_path}"
    )


# ------------------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------------------

def preprocess_image(
    image_path: Path,
    device: torch.device,
) -> torch.Tensor:

    image = Image.open(image_path).convert("RGB")

    original_width, original_height = image.size

    if original_width != 3325 or original_height != 1920:
        raise ValueError(
            f"Unexpected image dimensions: {image.size}"
        )

    resized_width = IMAGE_SIZE

    resized_height = round(
        original_height * IMAGE_SIZE / original_width
    )

    image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    image_tensor = to_tensor(image)

    image_tensor = normalize(
        image_tensor,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

    canvas = torch.zeros(
        (3, IMAGE_SIZE, IMAGE_SIZE),
        dtype=torch.float32,
    )

    top = (IMAGE_SIZE - resized_height) // 2
    left = (IMAGE_SIZE - resized_width) // 2

    canvas[
        :,
        top:top + resized_height,
        left:left + resized_width,
    ] = image_tensor

    # Match the existing 518x518 DINOv2 preprocessing.
    canvas = F.pad(
        canvas,
        (0, 6, 0, 6),
        mode="constant",
        value=0.0,
    )

    return canvas.unsqueeze(0).to(device)


# ------------------------------------------------------------
# ORIGINAL-RESOLUTION MASK LOADING
# ------------------------------------------------------------

def load_binary_mask(
    path: Path,
) -> np.ndarray:

    mask = Image.open(path).convert("L")

    if mask.size != (3325, 1920):
        raise ValueError(
            f"Unexpected mask dimensions: {mask.size}, file={path}"
        )

    array = np.asarray(mask)

    return array >= MASK_THRESHOLD


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def calculate_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, float]:

    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = (
        2.0 * precision * recall
        / (precision + recall + eps)
    )

    iou = tp / (tp + fp + fn + eps)

    dice = (
        2.0 * tp
        / (2.0 * tp + fp + fn + eps)
    )

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn + eps)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "accuracy": accuracy,
    }


# ------------------------------------------------------------
# CHECKPOINT LOADING
# ------------------------------------------------------------

def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> TrinetraLightningModule:

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Only load a checkpoint produced by this trusted project.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    hyperparameters = dict(
        checkpoint.get("hyper_parameters", {})
    )

    # Do not tune the threshold using PSCD-PERS.
    hyperparameters["threshold"] = PREDICTION_THRESHOLD

    model = TrinetraLightningModule(
        **hyperparameters
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


# ------------------------------------------------------------
# MAIN EVALUATION
# ------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print("=" * 65)
    print("TRINETRA - PSCD-PERS ZERO-SHOT EVALUATION")
    print("=" * 65)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Checkpoint:", args.checkpoint)
    print("Prediction threshold:", PREDICTION_THRESHOLD)

    # --------------------------------------------------------
    # Validate index
    # --------------------------------------------------------

    with args.index.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if len(records) != 3080:
        raise ValueError(
            f"Expected 3080 records, found {len(records)}"
        )

    sample_ids = set()

    direction_counts = defaultdict(int)

    for record in records:

        required = {
            "sample_id",
            "dataset",
            "t0_path",
            "t1_path",
            "mask_path",
        }

        missing = required - record.keys()

        if missing:
            raise ValueError(
                f"Missing fields in record: {missing}"
            )

        if record["dataset"] != "pscd_pers":
            raise ValueError(
                f"Unexpected dataset: {record['dataset']}"
            )

        sample_id = record["sample_id"]

        if sample_id in sample_ids:
            raise ValueError(
                f"Duplicate sample ID: {sample_id}"
            )

        sample_ids.add(sample_id)

        direction = sample_id.rsplit("_", 1)[-1]

        if direction not in DIRECTIONS:
            raise ValueError(
                f"Unknown camera direction: {sample_id}"
            )

        direction_counts[direction] += 1

    if any(direction_counts[d] != 770 for d in DIRECTIONS):
        raise ValueError(
            f"Unexpected direction distribution: {dict(direction_counts)}"
        )

    print("\nDataset validation: PASSED")
    print("Total samples:", len(records))
    print("Directions:", dict(direction_counts))

    # --------------------------------------------------------
    # Calibration masks
    # --------------------------------------------------------

    calibration_masks = {}

    for direction in DIRECTIONS:

        path = (
            args.dataset_root
            / "cmask"
            / f"mask_sensor_{direction}.png"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        calibration_masks[direction] = load_binary_mask(path)

        print(
            f"Calibration {direction}: "
            f"{calibration_masks[direction].mean() * 100:.4f}% valid"
        )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading trained decoder checkpoint...")

    decoder = load_trained_model(
        args.checkpoint,
        device,
    )

    print("Decoder checkpoint loaded successfully.")

    print("\nLoading frozen DINOv2 backbone...")

    backbone = DINOv2Backbone(
        model_name="facebook/dinov2-base",
        device=str(device),
    )

    backbone.eval()

    print("DINOv2 loaded successfully.")

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dir = args.output / "predictions"

    if args.save_predictions:
        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------------

    global_counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    direction_counts_metrics = {
        direction: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "samples": 0,
            "macro_metrics": defaultdict(float),
        }
        for direction in DIRECTIONS
    }

    per_sample_results = []

    # --------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------

    print("\nStarting evaluation...\n")

    with torch.inference_mode():

        for index, record in enumerate(records, start=1):

            sample_id = record["sample_id"]

            direction = sample_id.rsplit("_", 1)[-1]

            t0_path = resolve_dataset_path(
                record["t0_path"],
                args.dataset_root,
            )

            t1_path = resolve_dataset_path(
                record["t1_path"],
                args.dataset_root,
            )

            mask_path = resolve_dataset_path(
                record["mask_path"],
                args.dataset_root,
            )

            # Load original-resolution ground truth.
            ground_truth = load_binary_mask(mask_path)

            valid_region = calibration_masks[direction]

            # Load and preprocess images.
            t0 = preprocess_image(t0_path, device)
            t1 = preprocess_image(t1_path, device)

            # Frozen DINOv2 feature extraction.
            features_t0 = backbone(t0).float()
            features_t1 = backbone(t1).float()

            # Decoder inference.
            logits = decoder(features_t0, features_t1)

            probabilities = torch.sigmoid(logits)

            # Model output: [1, 1, 512, 512].
            probability_canvas = probabilities[0, 0].float()

            # The original image occupies rows 108:404.
            probability_image_region = probability_canvas[
                108:404,
                0:512,
            ]

            # Return predictions to original image dimensions.
            original_probability = F.interpolate(
                probability_image_region[None, None],
                size=(1920, 3325),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            prediction = (
                original_probability.cpu().numpy()
                >= PREDICTION_THRESHOLD
            )

            # Apply valid-region mask only for scoring.
            evaluated_prediction = prediction & valid_region
            evaluated_target = ground_truth & valid_region

            tp = int(
                np.count_nonzero(
                    evaluated_prediction & evaluated_target
                )
            )

            fp = int(
                np.count_nonzero(
                    evaluated_prediction & ~evaluated_target & valid_region
                )
            )

            fn = int(
                np.count_nonzero(
                    ~evaluated_prediction & evaluated_target
                )
            )

            tn = int(
                np.count_nonzero(
                    ~evaluated_prediction
                    & ~evaluated_target
                    & valid_region
                )
            )

            metrics = calculate_metrics(
                tp, fp, fn, tn
            )

            # Update global counts.
            global_counts["tp"] += tp
            global_counts["fp"] += fp
            global_counts["fn"] += fn
            global_counts["tn"] += tn

            # Update direction counts.
            direction_result = direction_counts_metrics[direction]

            direction_result["tp"] += tp
            direction_result["fp"] += fp
            direction_result["fn"] += fn
            direction_result["tn"] += tn
            direction_result["samples"] += 1

            for key, value in metrics.items():
                direction_result["macro_metrics"][key] += value

            per_sample_results.append({
                "sample_id": sample_id,
                "direction": direction,
                "valid_pixels": int(valid_region.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                **metrics,
            })

            # Optional prediction saving.
            if args.save_predictions:

                output_mask = (
                    prediction.astype(np.uint8) * 255
                )

                Image.fromarray(
                    output_mask,
                    mode="L",
                ).save(
                    prediction_dir / f"{sample_id}.png"
                )

            # Progress.
            if index % 100 == 0 or index == len(records):

                print(
                    f"Processed {index}/{len(records)} "
                    f"({index / len(records) * 100:.1f}%)"
                )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    global_metrics = calculate_metrics(
        global_counts["tp"],
        global_counts["fp"],
        global_counts["fn"],
        global_counts["tn"],
    )

    direction_results = {}

    for direction in DIRECTIONS:

        data = direction_counts_metrics[direction]

        macro_metrics = {
            key: value / data["samples"]
            for key, value in data["macro_metrics"].items()
        }

        micro_metrics = calculate_metrics(
            data["tp"],
            data["fp"],
            data["fn"],
            data["tn"],
        )

        direction_results[direction] = {
            "samples": data["samples"],
            "confusion_counts": {
                key: data[key]
                for key in ("tp", "fp", "fn", "tn")
            },
            "micro_metrics": micro_metrics,
            "macro_metrics": macro_metrics,
        }

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = args.output / "per_sample_metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(per_sample_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(per_sample_results)

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    results = {
        "dataset": "PSCD-PERS",
        "evaluation_type": "zero_shot_cross_domain",
        "checkpoint": str(args.checkpoint),
        "samples": len(records),
        "prediction_threshold": PREDICTION_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "preprocessing": {
            "original_size": [3325, 1920],
            "resized_size": [512, 296],
            "canvas_size": [512, 512],
            "model_input_size": [518, 518],
            "letterbox_top": 108,
            "letterbox_bottom": 108,
            "bottom_right_model_padding": 6,
        },
        "global_confusion_counts": global_counts,
        "global_micro_metrics": global_metrics,
        "direction_results": direction_results,
    }

    json_path = args.output / "evaluation_results.json"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PSCD-PERS ZERO-SHOT EVALUATION RESULTS")
    print("=" * 65)

    print("\nGLOBAL MICRO METRICS")

    for key, value in global_metrics.items():
        print(f"{key.upper():12s}: {value:.6f}")

    print("\nDIRECTION-WISE MICRO F1")

    for direction in DIRECTIONS:

        f1 = direction_results[direction][
            "micro_metrics"
        ]["f1"]

        print(
            f"{direction}: "
            f"F1={f1:.6f}"
        )

    print("\nSaved files:")
    print("JSON:", json_path)
    print("CSV:", csv_path)

    if args.save_predictions:
        print("Predictions:", prediction_dir)

    print("\nZERO-SHOT EVALUATION COMPLETED.")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Trinetra PSCD-PERS zero-shot evaluation"
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
