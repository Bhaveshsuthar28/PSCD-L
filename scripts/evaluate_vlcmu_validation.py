
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.datasets import VLCMUCDataset
from src.lightning_module import TrinetraLightningModule


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INDEX = (
    PROJECT_ROOT / "outputs/index/vl_cmu_train_val_split.json"
)

DEFAULT_CACHE = (
    PROJECT_ROOT / "data/processed/vl_cmu_cd_v2"
)

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Dataset/VL-CMU-CD-binary255"
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/training/checkpoints/trinetra-00-0.3455.ckpt"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/evaluation/vl_cmu_validation"
)

THRESHOLD = 0.5


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
        2 * precision * recall
        / (precision + recall + eps)
    )

    iou = tp / (tp + fp + fn + eps)

    dice = (
        2 * tp
        / (2 * tp + fp + fn + eps)
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


def load_model(
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

    hyperparameters = dict(
        checkpoint.get("hyper_parameters", {})
    )

    hyperparameters["threshold"] = THRESHOLD

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


def evaluate(args: argparse.Namespace) -> None:

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print("=" * 65)
    print("TRINETRA - VL-CMU VALIDATION EVALUATION")
    print("=" * 65)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Checkpoint:", args.checkpoint)
    print("Threshold:", THRESHOLD)

    # --------------------------------------------------------
    # Validate group-aware split
    # --------------------------------------------------------

    with args.index.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError("Expected a train/validation split manifest.")

    train_records = manifest.get("train", [])
    val_records = manifest.get("val", [])

    if len(train_records) != 3360:
        raise ValueError(
            f"Expected 3360 training samples, got {len(train_records)}"
        )

    if len(val_records) != 372:
        raise ValueError(
            f"Expected 372 validation samples, got {len(val_records)}"
        )

    train_ids = {x["sample_id"] for x in train_records}
    val_ids = {x["sample_id"] for x in val_records}

    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise ValueError(
            f"Train/validation sample overlap: {list(overlap)[:5]}"
        )

    train_groups = {
        sample_id.rsplit("_", 1)[0]
        for sample_id in train_ids
    }

    val_groups = {
        sample_id.rsplit("_", 1)[0]
        for sample_id in val_ids
    }

    group_overlap = train_groups.intersection(val_groups)

    if group_overlap:
        raise ValueError(
            f"Train/validation group leakage: {list(group_overlap)[:5]}"
        )

    print("\nSplit validation: PASSED")
    print("Training samples:", len(train_ids))
    print("Validation samples:", len(val_ids))
    print("Overlapping samples:", len(overlap))
    print("Overlapping groups:", len(group_overlap))

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = VLCMUCDataset(
        index_path=args.index,
        cache_dir=args.cache,
        dataset_root=args.dataset_root,
        split="val",
    )

    if len(dataset) != 372:
        raise ValueError(
            f"Validation dataset returned {len(dataset)} samples."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("\nValidation dataset loaded:", len(dataset))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        args.checkpoint,
        device,
    )

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    global_counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    per_sample_results = []

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nStarting validation inference...\n")

    with torch.inference_mode():

        for batch_index, batch in enumerate(loader, start=1):

            t0 = batch["t0"].to(
                device,
                non_blocking=True,
            )

            t1 = batch["t1"].to(
                device,
                non_blocking=True,
            )

            targets = (
                batch["mask"].to(
                    device,
                    non_blocking=True,
                ) >= 0.5
            )

            logits = model(t0, t1)

            predictions = (
                torch.sigmoid(logits) >= THRESHOLD
            )

            for i in range(t0.shape[0]):

                pred = predictions[i]
                target = targets[i]

                tp = int((pred & target).sum().item())
                fp = int((pred & ~target).sum().item())
                fn = int((~pred & target).sum().item())
                tn = int((~pred & ~target).sum().item())

                counts = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }

                for key, value in counts.items():
                    global_counts[key] += value

                metrics = calculate_metrics(
                    tp,
                    fp,
                    fn,
                    tn,
                )

                sample_id = batch["sample_id"][i]

                per_sample_results.append({
                    "sample_id": sample_id,
                    **counts,
                    **metrics,
                })

            if (
                batch_index % 25 == 0
                or batch_index == len(loader)
            ):
                processed = min(
                    batch_index * args.batch_size,
                    len(dataset),
                )

                print(
                    f"Processed {processed}/{len(dataset)} "
                    f"({processed / len(dataset) * 100:.1f}%)"
                )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    micro_metrics = calculate_metrics(
        global_counts["tp"],
        global_counts["fp"],
        global_counts["fn"],
        global_counts["tn"],
    )

    metric_names = (
        "precision",
        "recall",
        "f1",
        "iou",
        "dice",
        "accuracy",
    )

    macro_metrics = {
        key: sum(row[key] for row in per_sample_results)
        / len(per_sample_results)
        for key in metric_names
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    results = {
        "dataset": "VL-CMU-CD",
        "split": "validation",
        "evaluation_type": "in_domain_validation",
        "checkpoint": str(args.checkpoint),
        "samples": len(dataset),
        "threshold": THRESHOLD,
        "confusion_counts": global_counts,
        "micro_metrics": micro_metrics,
        "macro_metrics": macro_metrics,
    }

    json_path = args.output / "evaluation_results.json"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("VL-CMU VALIDATION RESULTS")
    print("=" * 65)

    print("\nMICRO METRICS")

    for key, value in micro_metrics.items():
        print(f"{key.upper():12s}: {value:.6f}")

    print("\nMACRO METRICS")

    for key, value in macro_metrics.items():
        print(f"{key.upper():12s}: {value:.6f}")

    print("\nSaved files:")
    print("JSON:", json_path)
    print("CSV:", csv_path)

    print("\nVL-CMU VALIDATION COMPLETED.")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Evaluate Trinetra on VL-CMU validation data."
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
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
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
