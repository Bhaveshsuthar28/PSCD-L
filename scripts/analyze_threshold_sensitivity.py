import sys
from pathlib import Path as _ProjectPath

_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image

from data.datasets import VLCMUCDataset


INDEX = Path("outputs/index/vl_cmu_test.json")
CACHE = Path("data/processed/vl_cmu_cd_v2")
ROOT = Path("Dataset/VL-CMU-CD-binary255")

PROB_DIR = Path(
    "outputs/evaluation/vl_cmu_official_test_probabilities/probabilities"
)

OUTPUT = Path("outputs/analysis/threshold_sensitivity")

THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.05), 2)


def metrics(tp, fp, fn, tn):
    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def counts(pred, target):
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    return tp, fp, fn, tn


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not PROB_DIR.is_dir():
        raise FileNotFoundError(PROB_DIR)

    print("Loading official test dataset...")

    dataset = VLCMUCDataset(
        index_path=INDEX,
        cache_dir=CACHE,
        dataset_root=ROOT,
        split="test",
    )

    if len(dataset) != 429:
        raise RuntimeError(f"Expected 429 samples, found {len(dataset)}")

    print("Loading probability maps and ground-truth masks...")

    samples = []

    for i in range(len(dataset)):
        item = dataset[i]

        sample_id = item["sample_id"]
        probability_path = PROB_DIR / f"{sample_id}.png"

        if not probability_path.is_file():
            raise FileNotFoundError(probability_path)

        probability = (
            np.asarray(
                Image.open(probability_path).convert("L"),
                dtype=np.float32,
            ) / 255.0
        )

        target = item["mask"].detach().cpu().numpy().squeeze() >= 0.5

        if probability.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {sample_id}: "
                f"probability={probability.shape}, target={target.shape}"
            )

        route_pass = "_".join(sample_id.split("_")[:2])

        samples.append({
            "sample_id": sample_id,
            "route_pass": route_pass,
            "probability": probability,
            "target": target,
        })

        if (i + 1) % 50 == 0 or i + 1 == len(dataset):
            print(f"Loaded {i + 1}/{len(dataset)} samples")

    global_rows = []
    route_rows = []

    print("\nCalculating threshold sensitivity...")

    for threshold in THRESHOLDS:
        total = np.zeros(4, dtype=np.int64)
        per_image_f1 = []

        route_counts = {}

        for sample in samples:
            pred = sample["probability"] >= threshold
            target = sample["target"]

            tp, fp, fn, tn = counts(pred, target)

            total += [tp, fp, fn, tn]

            image_metrics = metrics(tp, fp, fn, tn)
            per_image_f1.append(image_metrics["f1"])

            route = sample["route_pass"]

            if route not in route_counts:
                route_counts[route] = np.zeros(4, dtype=np.int64)

            route_counts[route] += [tp, fp, fn, tn]

        tp, fp, fn, tn = map(int, total)
        global_metrics = metrics(tp, fp, fn, tn)

        global_rows.append({
            "threshold": float(threshold),
            **global_metrics,
            "macro_f1": float(np.mean(per_image_f1)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

        for route, values in route_counts.items():
            rtp, rfp, rfn, rtn = map(int, values)

            route_rows.append({
                "threshold": float(threshold),
                "route_pass": route,
                **metrics(rtp, rfp, rfn, rtn),
                "tp": rtp,
                "fp": rfp,
                "fn": rfn,
                "tn": rtn,
            })

    global_path = OUTPUT / "threshold_sweep.csv"
    route_path = OUTPUT / "route_threshold_sweep.csv"

    with global_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=global_rows[0].keys())
        writer.writeheader()
        writer.writerows(global_rows)

    with route_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=route_rows[0].keys())
        writer.writeheader()
        writer.writerows(route_rows)

    print("\n" + "=" * 75)
    print("GLOBAL THRESHOLD SENSITIVITY")
    print("=" * 75)

    for row in global_rows:
        print(
            f"Threshold {row['threshold']:.2f} | "
            f"Precision {row['precision']:.4f} | "
            f"Recall {row['recall']:.4f} | "
            f"F1 {row['f1']:.4f} | "
            f"IoU {row['iou']:.4f} | "
            f"Macro F1 {row['macro_f1']:.4f}"
        )

    best = max(global_rows, key=lambda x: x["f1"])

    print("\nBest diagnostic threshold on this test set:")
    print("Threshold:", best["threshold"])
    print("Micro F1:", round(best["f1"], 6))
    print("Precision:", round(best["precision"], 6))
    print("Recall:", round(best["recall"], 6))

    print("\nSaved:")
    print(global_path)
    print(route_path)

    print(
        "\nIMPORTANT: This is diagnostic analysis only. "
        "Do not select the final model threshold using official test results."
    )


if __name__ == "__main__":
    main()
