
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import lightning.pytorch as pl
from PIL import Image
from torch.utils.data import DataLoader

from data.datasets import VLCMUCDataset
from src.lightning_module import TrinetraLightningModule


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INDEX = (
    PROJECT_ROOT / "outputs" / "index" / "vl_cmu_test.json"
)

DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "vl_cmu_cd_v2"
)

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Dataset" / "VL-CMU-CD-binary255"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "evaluation" / "vl_cmu_test"
)


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Trinetra on the official VL-CMU test set."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained Lightning checkpoint.",
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Also save probability maps as 8-bit PNG files.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU evaluation.",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_args(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    for path in [
        args.index,
        args.cache_dir,
        args.dataset_root,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found: {path}"
            )

    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if not 0 < args.threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")

    with args.index.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Official test index must be a JSON list.")

    if len(records) != 429:
        raise ValueError(
            f"Expected 429 official test samples, found {len(records)}."
        )

    for record in records:
        if record.get("split") != "test":
            raise ValueError(
                f"Non-test sample found: {record.get('sample_id')}"
            )

        if record.get("dataset") != "vl_cmu":
            raise ValueError(
                f"Unexpected dataset: {record.get('dataset')}"
            )

    sample_ids = [record["sample_id"] for record in records]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in official test index.")

    print("Official test index validated: 429 samples.")


# ---------------------------------------------------------
# CHECKPOINT LOADING
# ---------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device):
    print("\nLoading checkpoint...")

    # The checkpoint is a trusted local Lightning checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" not in checkpoint:
        raise ValueError(
            "Invalid Lightning checkpoint: state_dict is missing."
        )

    model = TrinetraLightningModule(
        learning_rate=1e-4,
        weight_decay=1e-4,
        bce_weight=1.0,
        dice_weight=1.0,
        threshold=0.5,
        max_epochs=50,
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    print("Checkpoint loaded successfully.")
    print("Decoder parameters:", sum(
        p.numel() for p in model.model.parameters()
    ))

    return model


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(tp, fp, fn, tn):
    epsilon = 1e-8

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = (
        2 * precision * recall
        / (precision + recall + epsilon)
    )

    iou = tp / (tp + fp + fn + epsilon)

    dice = (
        2 * tp
        / (2 * tp + fp + fn + epsilon)
    )

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn + epsilon)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "accuracy": accuracy,
    }


def confusion_counts(prediction, target):
    prediction = prediction.bool()
    target = target.bool()

    tp = torch.logical_and(prediction, target).sum().item()
    fp = torch.logical_and(prediction, ~target).sum().item()
    fn = torch.logical_and(~prediction, target).sum().item()
    tn = torch.logical_and(~prediction, ~target).sum().item()

    return tp, fp, fn, tn


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate(model, loader, device, args):
    aggregate_tp = 0
    aggregate_fp = 0
    aggregate_fn = 0
    aggregate_tn = 0

    per_sample_results = []

    predictions_dir = args.output_dir / "predictions"
    probabilities_dir = args.output_dir / "probabilities"

    predictions_dir.mkdir(parents=True, exist_ok=True)

    if args.save_probabilities:
        probabilities_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"

    print("\nStarting official test evaluation...")
    print("Threshold:", args.threshold)
    print("AMP:", use_amp)

    with torch.inference_mode():

        for batch_index, batch in enumerate(loader):
            t0 = batch["t0"].to(device, non_blocking=True)
            t1 = batch["t1"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(t0, t1)

            probabilities = torch.sigmoid(logits.float())

            predictions = probabilities >= args.threshold
            targets = mask >= 0.5

            batch_size = t0.shape[0]

            for index in range(batch_size):
                sample_id = batch["sample_id"][index]

                prediction = predictions[index, 0]
                target = targets[index, 0]

                tp, fp, fn, tn = confusion_counts(
                    prediction,
                    target,
                )

                aggregate_tp += tp
                aggregate_fp += fp
                aggregate_fn += fn
                aggregate_tn += tn

                sample_metrics = calculate_metrics(
                    tp, fp, fn, tn
                )

                per_sample_results.append({
                    "sample_id": sample_id,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    **sample_metrics,
                })

                prediction_image = (
                    prediction.to(torch.uint8).cpu().numpy() * 255
                )

                Image.fromarray(
                    prediction_image,
                    mode="L",
                ).save(
                    predictions_dir / f"{sample_id}.png"
                )

                if args.save_probabilities:
                    probability_image = (
                        probabilities[index, 0]
                        .clamp(0, 1)
                        .mul(255)
                        .to(torch.uint8)
                        .cpu()
                        .numpy()
                    )

                    Image.fromarray(
                        probability_image,
                        mode="L",
                    ).save(
                        probabilities_dir / f"{sample_id}.png"
                    )

            if (
                (batch_index + 1) % 25 == 0
                or batch_index + 1 == len(loader)
            ):
                print(
                    f"Processed {batch_index + 1}/{len(loader)} batches"
                )

    aggregate_metrics = calculate_metrics(
        aggregate_tp,
        aggregate_fp,
        aggregate_fn,
        aggregate_tn,
    )

    return (
        aggregate_metrics,
        per_sample_results,
        aggregate_tp,
        aggregate_fp,
        aggregate_fn,
        aggregate_tn,
    )


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_results(
    args,
    metrics,
    per_sample_results,
    tp,
    fp,
    fn,
    tn,
):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "per_sample_metrics.csv"

    fields = [
        "sample_id",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "iou",
        "dice",
        "accuracy",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_sample_results)

    macro_metrics = {}

    metric_names = [
        "precision",
        "recall",
        "f1",
        "iou",
        "dice",
        "accuracy",
    ]

    for name in metric_names:
        macro_metrics[name] = float(
            np.mean([
                result[name]
                for result in per_sample_results
            ])
        )

    results = {
        "dataset": "VL-CMU-CD-binary255",
        "split": "official_test",
        "num_samples": len(per_sample_results),
        "threshold": args.threshold,
        "checkpoint": str(args.checkpoint.resolve()),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "micro_metrics": {
            key: float(value)
            for key, value in metrics.items()
        },
        "macro_metrics": macro_metrics,
    }

    json_path = args.output_dir / "evaluation_results.json"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    return csv_path, json_path, results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    args = parse_args()

    validate_args(args)

    pl.seed_everything(42, workers=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print("=" * 65)
    print("TRINETRA OFFICIAL VL-CMU TEST EVALUATION")
    print("=" * 65)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Dataset
    test_dataset = VLCMUCDataset(
        index_path=args.index,
        cache_dir=args.cache_dir,
        dataset_root=args.dataset_root,
        split="test",
    )

    if len(test_dataset) != 429:
        raise RuntimeError(
            f"Expected 429 test samples, found {len(test_dataset)}."
        )

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # Model
    model = load_model(
        args.checkpoint,
        device,
    )

    # Evaluate
    results = evaluate(
        model,
        loader,
        device,
        args,
    )

    metrics, per_sample, tp, fp, fn, tn = results

    # Save
    csv_path, json_path, saved_results = save_results(
        args,
        metrics,
        per_sample,
        tp,
        fp,
        fn,
        tn,
    )

    # Display
    print("\n" + "=" * 65)
    print("OFFICIAL TEST RESULTS")
    print("=" * 65)

    print("Samples:", len(per_sample))

    print("\nMICRO METRICS (global pixel counts)")
    for key, value in metrics.items():
        print(f"{key.upper():12s}: {value:.6f}")

    print("\nMACRO METRICS (mean per image)")
    for key, value in saved_results["macro_metrics"].items():
        print(f"{key.upper():12s}: {value:.6f}")

    print("\nConfusion matrix:")
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)

    print("\nSaved results:")
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Predictions:", args.output_dir / "predictions")

    if args.save_probabilities:
        print("Probabilities:", args.output_dir / "probabilities")

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
