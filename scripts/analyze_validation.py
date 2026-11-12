
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from data.datasets import VLCMUCDataset
from src.lightning_module import TrinetraLightningModule


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs"
    / "training"
    / "checkpoints"
    / "trinetra-00-0.3455.ckpt"
)

DEFAULT_INDEX = (
    ROOT
    / "outputs"
    / "index"
    / "vl_cmu_train_val_split.json"
)

DEFAULT_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "vl_cmu_cd_v2"
)

DEFAULT_DATASET_ROOT = (
    ROOT
    / "Dataset"
    / "VL-CMU-CD-binary255"
)

DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "validation_analysis"
    / "pilot"
)


# =========================================================
# THRESHOLDS
# =========================================================

THRESHOLDS = np.round(
    np.arange(0.20, 0.801, 0.05),
    2,
).tolist()

BASELINE_THRESHOLD = 0.50


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Quantitative validation analysis for Trinetra."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
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
        "--cpu",
        action="store_true",
    )

    return parser.parse_args()


# =========================================================
# VALIDATION
# =========================================================

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
                f"Required path does not exist: {path}"
            )

    if args.batch_size < 1:
        raise ValueError(
            "Batch size must be at least 1."
        )

    if args.num_workers < 0:
        raise ValueError(
            "Number of workers cannot be negative."
        )


# =========================================================
# MODEL LOADING
# =========================================================

def load_model(checkpoint_path, device):

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" not in checkpoint:
        raise ValueError(
            "Invalid Lightning checkpoint: state_dict missing."
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

    return model


# =========================================================
# METRICS
# =========================================================

def safe_divide(numerator, denominator):

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def calculate_metrics(tp, fp, fn, tn):

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2 * tp,
        2 * tp + fp + fn,
    )

    iou = safe_divide(
        tp,
        tp + fp + fn,
    )

    dice = safe_divide(
        2 * tp,
        2 * tp + fp + fn,
    )

    accuracy = safe_divide(
        tp + tn,
        tp + fp + fn + tn,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "accuracy": accuracy,
    }


# =========================================================
# LOAD VALIDATION DATA
# =========================================================

def create_validation_loader(args):

    dataset = VLCMUCDataset(
        index_path=args.index,
        cache_dir=args.cache_dir,
        dataset_root=args.dataset_root,
        split="val",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
        persistent_workers=args.num_workers > 0,
    )

    return dataset, loader


# =========================================================
# VALIDATION ANALYSIS
# =========================================================

def run_analysis(model, loader, device):

    threshold_count = len(THRESHOLDS)

    threshold_tensor = torch.tensor(
        THRESHOLDS,
        device=device,
        dtype=torch.float32,
    ).view(
        threshold_count,
        1,
        1,
        1,
        1,
    )

    # Global pixel counts for each threshold.
    global_counts = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
        for threshold in THRESHOLDS
    }

    # Per-image confusion counts.
    sample_counts = {}

    processed = 0

    print("\nStarting full validation analysis...")
    print("Thresholds:", THRESHOLDS)

    with torch.inference_mode():

        for batch_number, batch in enumerate(loader, start=1):

            t0 = batch["t0"].to(
                device,
                non_blocking=True,
            )

            t1 = batch["t1"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            use_amp = device.type == "cuda"

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = model(t0, t1)

            probabilities = torch.sigmoid(
                logits.float()
            )

            predictions = (
                probabilities.unsqueeze(0)
                >= threshold_tensor
            )

            ground_truth = (
                masks >= 0.5
            ).unsqueeze(0)

            # Shape: [thresholds, batch, 1, height, width]
            tp = (
                predictions & ground_truth
            ).sum(dim=(2, 3, 4))

            fp = (
                predictions & ~ground_truth
            ).sum(dim=(2, 3, 4))

            fn = (
                ~predictions & ground_truth
            ).sum(dim=(2, 3, 4))

            tn = (
                ~predictions & ~ground_truth
            ).sum(dim=(2, 3, 4))

            # Transfer only the small confusion matrices.
            tp_cpu = tp.cpu().numpy()
            fp_cpu = fp.cpu().numpy()
            fn_cpu = fn.cpu().numpy()
            tn_cpu = tn.cpu().numpy()

            batch_size = len(batch["sample_id"])

            for sample_index in range(batch_size):

                sample_id = batch["sample_id"][sample_index]

                sample_counts[sample_id] = {}

                for threshold_index, threshold in enumerate(
                    THRESHOLDS
                ):

                    counts = {
                        "tp": int(
                            tp_cpu[threshold_index, sample_index]
                        ),
                        "fp": int(
                            fp_cpu[threshold_index, sample_index]
                        ),
                        "fn": int(
                            fn_cpu[threshold_index, sample_index]
                        ),
                        "tn": int(
                            tn_cpu[threshold_index, sample_index]
                        ),
                    }

                    sample_counts[sample_id][threshold] = counts

                    for key in counts:
                        global_counts[threshold][key] += counts[key]

                processed += 1

            if batch_number % 25 == 0 or batch_number == len(loader):

                print(
                    f"Processed {processed}/{len(loader.dataset)} "
                    "validation samples"
                )

    return global_counts, sample_counts, processed


# =========================================================
# AGGREGATE METRICS
# =========================================================

def calculate_threshold_results(
    global_counts,
    sample_counts,
):

    results = []

    for threshold in THRESHOLDS:

        counts = global_counts[threshold]

        micro = calculate_metrics(
            counts["tp"],
            counts["fp"],
            counts["fn"],
            counts["tn"],
        )

        per_image_metrics = []

        for sample_id in sample_counts:

            sample = sample_counts[sample_id][threshold]

            metrics = calculate_metrics(
                sample["tp"],
                sample["fp"],
                sample["fn"],
                sample["tn"],
            )

            per_image_metrics.append(metrics)

        macro = {
            metric: float(
                np.mean([
                    sample[metric]
                    for sample in per_image_metrics
                ])
            )
            for metric in [
                "precision",
                "recall",
                "f1",
                "iou",
                "dice",
                "accuracy",
            ]
        }

        results.append({
            "threshold": threshold,
            "micro": micro,
            "macro": macro,
            "counts": counts,
        })

    return results


# =========================================================
# PER-IMAGE CSV
# =========================================================

def save_per_sample_csv(
    output_path,
    sample_counts,
    selected_threshold,
):

    fields = [
        "sample_id",
        "threshold",
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
        "ground_truth_positive_pixels",
        "predicted_positive_pixels",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for sample_id, threshold_data in sample_counts.items():

            for threshold_name, threshold in [
                ("baseline_0.50", BASELINE_THRESHOLD),
                ("selected_validation", selected_threshold),
            ]:

                counts = threshold_data[threshold]

                metrics = calculate_metrics(
                    counts["tp"],
                    counts["fp"],
                    counts["fn"],
                    counts["tn"],
                )

                writer.writerow({
                    "sample_id": sample_id,
                    "threshold": threshold_name,
                    **counts,
                    **metrics,
                    "ground_truth_positive_pixels": (
                        counts["tp"] + counts["fn"]
                    ),
                    "predicted_positive_pixels": (
                        counts["tp"] + counts["fp"]
                    ),
                })


# =========================================================
# THRESHOLD SWEEP CSV
# =========================================================

def save_threshold_csv(output_path, results):

    fields = [
        "threshold",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_iou",
        "micro_dice",
        "micro_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_iou",
        "macro_dice",
        "macro_accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for result in results:

            row = {
                "threshold": result["threshold"],
                **{
                    f"micro_{key}": value
                    for key, value in result["micro"].items()
                },
                **{
                    f"macro_{key}": value
                    for key, value in result["macro"].items()
                },
                **result["counts"],
            }

            writer.writerow(row)


# =========================================================
# GRAPH
# =========================================================

def save_threshold_plot(output_path, results):

    thresholds = [
        result["threshold"]
        for result in results
    ]

    macro_f1 = [
        result["macro"]["f1"]
        for result in results
    ]

    micro_f1 = [
        result["micro"]["f1"]
        for result in results
    ]

    macro_iou = [
        result["macro"]["iou"]
        for result in results
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds,
        macro_f1,
        marker="o",
        label="Macro F1",
    )

    plt.plot(
        thresholds,
        micro_f1,
        marker="s",
        label="Micro F1",
    )

    plt.plot(
        thresholds,
        macro_iou,
        marker="^",
        label="Macro IoU",
    )

    plt.axvline(
        BASELINE_THRESHOLD,
        linestyle="--",
        label="Baseline threshold 0.50",
    )

    plt.xlabel("Prediction threshold")
    plt.ylabel("Metric score")

    plt.title(
        "Trinetra - Validation Threshold Analysis"
    )

    plt.xticks(thresholds)
    plt.ylim(0, 1.05)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    validate_args(args)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print("=" * 70)
    print("TRINETRA VALIDATION ANALYSIS")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Checkpoint:", args.checkpoint)

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

    dataset, loader = create_validation_loader(args)

    print("\nValidation samples:", len(dataset))

    if len(dataset) != 372:
        raise ValueError(
            "Expected 372 validation samples, "
            f"but found {len(dataset)}. "
            "Check the validation manifest."
        )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = load_model(
        args.checkpoint,
        device,
    )

    # -----------------------------------------------------
    # INFERENCE AND METRICS
    # -----------------------------------------------------

    global_counts, sample_counts, processed = run_analysis(
        model,
        loader,
        device,
    )

    if processed != len(dataset):
        raise RuntimeError(
            f"Expected {len(dataset)} samples, "
            f"processed {processed}."
        )

    results = calculate_threshold_results(
        global_counts,
        sample_counts,
    )

    # -----------------------------------------------------
    # THRESHOLD SELECTION
    # -----------------------------------------------------

    # Primary selection criterion: macro F1.
    best_result = max(
        results,
        key=lambda result: result["macro"]["f1"],
    )

    baseline_result = next(
        result
        for result in results
        if result["threshold"] == BASELINE_THRESHOLD
    )

    selected_threshold = best_result["threshold"]

    # -----------------------------------------------------
    # SAVE REPORTS
    # -----------------------------------------------------

    save_threshold_csv(
        args.output_dir / "threshold_sweep.csv",
        results,
    )

    save_per_sample_csv(
        args.output_dir / "per_sample_metrics.csv",
        sample_counts,
        selected_threshold,
    )

    save_threshold_plot(
        args.output_dir / "threshold_analysis.png",
        results,
    )

    report = {
        "project": "Trinetra",
        "evaluation_split": "validation",
        "number_of_samples": processed,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "thresholds_evaluated": THRESHOLDS,
        "baseline_threshold": BASELINE_THRESHOLD,
        "selection_criterion": "macro_f1",
        "selected_validation_threshold": selected_threshold,
        "baseline_metrics": {
            "micro": baseline_result["micro"],
            "macro": baseline_result["macro"],
            "counts": baseline_result["counts"],
        },
        "selected_threshold_metrics": {
            "micro": best_result["micro"],
            "macro": best_result["macro"],
            "counts": best_result["counts"],
        },
        "threshold_results": results,
        "metric_definition": (
            "Per-image metrics use zero_division=0. "
            "Macro metrics are unweighted averages across images. "
            "Micro metrics aggregate pixel confusion counts."
        ),
    }

    with (
        args.output_dir / "validation_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # TERMINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION ANALYSIS COMPLETED")
    print("=" * 70)

    print("Samples evaluated:", processed)

    print("\nBASELINE THRESHOLD:", BASELINE_THRESHOLD)

    print(
        "Baseline macro F1:",
        f"{baseline_result['macro']['f1']:.4f}",
    )

    print(
        "Baseline micro F1:",
        f"{baseline_result['micro']['f1']:.4f}",
    )

    print(
        "Baseline macro IoU:",
        f"{baseline_result['macro']['iou']:.4f}",
    )

    print("\nBEST VALIDATION THRESHOLD:", selected_threshold)

    print(
        "Best macro F1:",
        f"{best_result['macro']['f1']:.4f}",
    )

    print(
        "Best micro F1:",
        f"{best_result['micro']['f1']:.4f}",
    )

    print(
        "Best macro IoU:",
        f"{best_result['macro']['iou']:.4f}",
    )

    print("\nOUTPUT FILES:")

    print(
        args.output_dir / "validation_report.json"
    )

    print(
        args.output_dir / "threshold_sweep.csv"
    )

    print(
        args.output_dir / "per_sample_metrics.csv"
    )

    print(
        args.output_dir / "threshold_analysis.png"
    )

    print("\nALL VALIDATION ANALYSIS TESTS COMPLETED.")


if __name__ == "__main__":
    main()
