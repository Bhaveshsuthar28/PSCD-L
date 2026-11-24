
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path, PureWindowsPath

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

IMAGE_SIZE = 512
MODEL_SIZE = 518
PATCH_SIZE = 14

ORIGINAL_WIDTH = 4096
ORIGINAL_HEIGHT = 1152

RESIZED_WIDTH = 512
RESIZED_HEIGHT = 144

LETTERBOX_TOP = 184
LETTERBOX_BOTTOM = 184

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MASK_THRESHOLD = 128
PREDICTION_THRESHOLD = 0.5

DEFAULT_INDEX = PROJECT_ROOT / "outputs/index/pscd.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Dataset/PSCD"

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/training/checkpoints/trinetra-33-0.1482.ckpt"
)

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/evaluation/pscd_zero_shot"


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

    parts = PureWindowsPath(stored_path).parts

    try:
        dataset_position = next(
            i for i, part in enumerate(parts)
            if part.casefold() == "pscd"
        )
    except StopIteration:
        dataset_position = -1

    if dataset_position >= 0:
        relative_parts = parts[dataset_position + 1:]
        candidate = dataset_root.joinpath(*relative_parts)

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve PSCD file: {stored_path}"
    )


# ------------------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------------------

def preprocess_image(
    image_path: Path,
    device: torch.device,
) -> torch.Tensor:

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        if image.size != (ORIGINAL_WIDTH, ORIGINAL_HEIGHT):
            raise ValueError(
                f"Unexpected image dimensions: {image.size}, "
                f"file={image_path}"
            )

        image = image.resize(
            (RESIZED_WIDTH, RESIZED_HEIGHT),
            Image.Resampling.BILINEAR,
        )

        tensor = to_tensor(image)

    tensor = normalize(
        tensor,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

    canvas = torch.zeros(
        (3, IMAGE_SIZE, IMAGE_SIZE),
        dtype=torch.float32,
    )

    canvas[
        :,
        LETTERBOX_TOP:LETTERBOX_TOP + RESIZED_HEIGHT,
        :
    ] = tensor

    # Match the training DINOv2 input dimensions.
    canvas = F.pad(
        canvas,
        (0, 6, 0, 6),
        mode="constant",
        value=0.0,
    )

    if tuple(canvas.shape) != (3, MODEL_SIZE, MODEL_SIZE):
        raise RuntimeError(
            f"Unexpected model input shape: {tuple(canvas.shape)}"
        )

    return canvas.unsqueeze(0).to(device)


# ------------------------------------------------------------
# MASK LOADING
# ------------------------------------------------------------

def load_binary_mask(path: Path) -> np.ndarray:

    with Image.open(path) as image:
        mask = image.convert("L")

        if mask.size != (ORIGINAL_WIDTH, ORIGINAL_HEIGHT):
            raise ValueError(
                f"Unexpected mask dimensions: {mask.size}, "
                f"file={path}"
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

    eps = 1e-8

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
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" not in checkpoint:
        raise ValueError("Invalid Lightning checkpoint.")

    hyperparameters = dict(
        checkpoint.get("hyper_parameters", {})
    )

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
# INDEX VALIDATION
# ------------------------------------------------------------

def validate_index(index_path: Path) -> list[dict]:

    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    with index_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("PSCD index must be a JSON list.")

    if len(records) != 770:
        raise ValueError(
            f"Expected 770 PSCD samples, found {len(records)}"
        )

    sample_ids = set()

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
                f"Missing fields: {missing}"
            )

        if record["dataset"] != "pscd":
            raise ValueError(
                f"Unexpected dataset: {record['dataset']}"
            )

        sample_id = record["sample_id"]

        if sample_id in sample_ids:
            raise ValueError(
                f"Duplicate sample ID: {sample_id}"
            )

        sample_ids.add(sample_id)

    print("PSCD index validation: PASSED")
    print("Samples:", len(records))

    return records


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print("=" * 65)
    print("TRINETRA - PSCD ZERO-SHOT EVALUATION")
    print("=" * 65)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Checkpoint:", args.checkpoint)
    print("Threshold:", PREDICTION_THRESHOLD)

    records = validate_index(args.index)

    # Validate dataset root.
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)

    # Load models.
    print("\nLoading trained decoder...")

    decoder = load_trained_model(
        args.checkpoint,
        device,
    )

    print("Decoder loaded.")

    print("\nLoading frozen DINOv2 backbone...")

    backbone = DINOv2Backbone(
        model_name="facebook/dinov2-base",
        device=str(device),
    )

    backbone.eval()

    print("Backbone loaded.")

    # Output directories.
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

    # Global confusion counts.
    global_counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    per_sample_results = []

    macro_accumulator = {
        key: 0.0
        for key in (
            "precision",
            "recall",
            "f1",
            "iou",
            "dice",
            "accuracy",
        )
    }

    # Evaluation.
    print("\nStarting evaluation...\n")

    with torch.inference_mode():

        for index, record in enumerate(records, start=1):

            sample_id = record["sample_id"]

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

            # Ground truth at original resolution.
            ground_truth = load_binary_mask(mask_path)

            # Preprocess both images.
            t0 = preprocess_image(t0_path, device)
            t1 = preprocess_image(t1_path, device)

            # Frozen DINOv2 feature extraction.
            features_t0 = backbone(t0).float()
            features_t1 = backbone(t1).float()

            # Decoder inference.
            logits = decoder(features_t0, features_t1)

            probabilities = torch.sigmoid(logits.float())

            if tuple(probabilities.shape) != (1, 1, 512, 512):
                raise ValueError(
                    f"Unexpected decoder output: "
                    f"{tuple(probabilities.shape)}"
                )

            probability_canvas = probabilities[0, 0]

            # Extract the original image region.
            probability_image_region = probability_canvas[
                LETTERBOX_TOP:
                LETTERBOX_TOP + RESIZED_HEIGHT,
                0:RESIZED_WIDTH,
            ]

            # Restore predictions to original resolution.
            original_probability = F.interpolate(
                probability_image_region[None, None],
                size=(ORIGINAL_HEIGHT, ORIGINAL_WIDTH),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            prediction = (
                original_probability.cpu().numpy()
                >= PREDICTION_THRESHOLD
            )

            # Calculate confusion counts.
            tp = int(
                np.count_nonzero(
                    prediction & ground_truth
                )
            )

            fp = int(
                np.count_nonzero(
                    prediction & ~ground_truth
                )
            )

            fn = int(
                np.count_nonzero(
                    ~prediction & ground_truth
                )
            )

            tn = int(
                np.count_nonzero(
                    ~prediction & ~ground_truth
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

            # Update macro metrics.
            for key, value in metrics.items():
                macro_accumulator[key] += value

            per_sample_results.append({
                "sample_id": sample_id,
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

            if index % 50 == 0 or index == len(records):
                print(
                    f"Processed {index}/{len(records)} "
                    f"({index / len(records) * 100:.1f}%)"
                )

    # Final micro metrics.
    micro_metrics = calculate_metrics(
        global_counts["tp"],
        global_counts["fp"],
        global_counts["fn"],
        global_counts["tn"],
    )

    # Final macro metrics.
    macro_metrics = {
        key: value / len(records)
        for key, value in macro_accumulator.items()
    }

    # Save CSV.
    csv_path = args.output / "per_sample_metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fields = list(per_sample_results[0].keys())

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(per_sample_results)

    # Save JSON.
    results = {
        "dataset": "PSCD",
        "evaluation_type": "zero_shot_cross_domain",
        "samples": len(records),
        "checkpoint": str(args.checkpoint.resolve()),
        "prediction_threshold": PREDICTION_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "preprocessing": {
            "original_size": [ORIGINAL_WIDTH, ORIGINAL_HEIGHT],
            "resized_size": [RESIZED_WIDTH, RESIZED_HEIGHT],
            "canvas_size": [IMAGE_SIZE, IMAGE_SIZE],
            "model_input_size": [MODEL_SIZE, MODEL_SIZE],
            "letterbox_top": LETTERBOX_TOP,
            "letterbox_bottom": LETTERBOX_BOTTOM,
            "bottom_right_model_padding": 6,
            "interpolation": "bilinear",
            "normalization": "ImageNet",
        },
        "global_confusion_counts": global_counts,
        "micro_metrics": micro_metrics,
        "macro_metrics": macro_metrics,
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

    # Report.
    print("\n" + "=" * 65)
    print("PSCD ZERO-SHOT EVALUATION RESULTS")
    print("=" * 65)

    print("\nMICRO METRICS")

    for key, value in micro_metrics.items():
        print(f"{key.upper():12s}: {value:.6f}")

    print("\nMACRO METRICS")

    for key, value in macro_metrics.items():
        print(f"{key.upper():12s}: {value:.6f}")

    print("\nConfusion counts:")
    for key, value in global_counts.items():
        print(f"{key.upper():12s}: {value:,}")

    print("\nSaved files:")
    print("JSON:", json_path)
    print("CSV:", csv_path)

    if args.save_predictions:
        print("Predictions:", prediction_dir)

    print("\nPSCD ZERO-SHOT EVALUATION COMPLETED.")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Trinetra PSCD zero-shot evaluation"
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
