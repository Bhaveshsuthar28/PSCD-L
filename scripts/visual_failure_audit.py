import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = PROJECT_ROOT / "outputs/index/vl_cmu_test.json"

DATASET_ROOT = PROJECT_ROOT / "Dataset/VL-CMU-CD-binary255"

PREDICTION_DIR = (
    PROJECT_ROOT / "outputs/evaluation/vl_cmu_official_test/predictions"
)

PROBABILITY_DIR = (
    PROJECT_ROOT /
    "outputs/evaluation/vl_cmu_official_test_probabilities/probabilities"
)

METRICS_CSV = (
    PROJECT_ROOT /
    "outputs/evaluation/vl_cmu_official_test/per_sample_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/analysis/visual_failure_audit"
)

SAMPLE_IDS = [
    "085_1_00_0",
    "145_1_02_0",
    "139_1_05_0",
    "045_1_00_0",
    "060_1_00_0",
    "107_1_00_0",
]

THRESHOLD = 0.5


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def load_rgb(path):
    return np.asarray(
        Image.open(path).convert("RGB")
    )


def load_gray(path):
    return np.asarray(
        Image.open(path).convert("L")
    )


def resolve_image_path(record, field, folder):
    original_path = Path(record[field])

    if original_path.is_file():
        return original_path

    fallback = (
        DATASET_ROOT /
        "test" /
        folder /
        f"{record['sample_id']}.png"
    )

    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        f"Could not find {field} for {record['sample_id']}\n"
        f"Index path: {original_path}\n"
        f"Fallback: {fallback}"
    )


def load_metrics():
    if not METRICS_CSV.is_file():
        return {}

    results = {}

    with METRICS_CSV.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        for row in csv.DictReader(file):
            results[row["sample_id"]] = row

    return results


def add_image(ax, image, title, cmap=None, vmin=None, vmax=None):
    ax.imshow(
        image,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INDEX_PATH.is_file():
        raise FileNotFoundError(INDEX_PATH)

    if not PREDICTION_DIR.is_dir():
        raise FileNotFoundError(PREDICTION_DIR)

    if not PROBABILITY_DIR.is_dir():
        raise FileNotFoundError(PROBABILITY_DIR)

    records = json.loads(
        INDEX_PATH.read_text(encoding="utf-8")
    )

    record_map = {
        record["sample_id"]: record
        for record in records
    }

    metrics_map = load_metrics()

    all_figures = []

    print("=" * 70)
    print("TRINETRA VISUAL FAILURE AUDIT")
    print("=" * 70)

    for sample_id in SAMPLE_IDS:

        print(f"\nProcessing: {sample_id}")

        if sample_id not in record_map:
            raise KeyError(
                f"Sample missing from index: {sample_id}"
            )

        record = record_map[sample_id]

        # Load original images.
        t0_path = resolve_image_path(
            record, "t0_path", "t0"
        )

        t1_path = resolve_image_path(
            record, "t1_path", "t1"
        )

        mask_path = resolve_image_path(
            record, "mask_path", "mask"
        )

        t0 = load_rgb(t0_path)
        t1 = load_rgb(t1_path)

        # Load ground truth and threshold consistently.
        mask_gray = load_gray(mask_path)
        ground_truth = mask_gray >= 128

        # Load model prediction.
        prediction_path = (
            PREDICTION_DIR / f"{sample_id}.png"
        )

        prediction = load_gray(prediction_path) >= 128

        # Load saved probability map.
        probability_path = (
            PROBABILITY_DIR / f"{sample_id}.png"
        )

        probability = (
            load_gray(probability_path).astype(np.float32)
            / 255.0
        )

        if not (
            t0.shape[:2]
            == t1.shape[:2]
            == ground_truth.shape
            == prediction.shape
            == probability.shape
        ):
            raise ValueError(
                f"Image dimensions do not match for {sample_id}"
            )

        # Build overlays.
        overlay = t1.copy()

        # Green = correct detection.
        # Red = false positive.
        # Blue = false negative.
        tp = prediction & ground_truth
        fp = prediction & ~ground_truth
        fn = ~prediction & ground_truth

        overlay[tp] = (
            0.45 * overlay[tp] + 0.55 * np.array([0, 255, 0])
        ).astype(np.uint8)

        overlay[fp] = (
            0.45 * overlay[fp] + 0.55 * np.array([255, 0, 0])
        ).astype(np.uint8)

        overlay[fn] = (
            0.45 * overlay[fn] + 0.55 * np.array([0, 100, 255])
        ).astype(np.uint8)

        # Metrics for title.
        metric = metrics_map.get(sample_id, {})

        f1 = float(metric.get("f1", 0))
        precision = float(metric.get("precision", 0))
        recall = float(metric.get("recall", 0))
        iou = float(metric.get("iou", 0))

        # Six-column comparison.
        fig, axes = plt.subplots(
            1, 6,
            figsize=(22, 5)
        )

        fig.suptitle(
            f"{sample_id} | "
            f"F1: {f1:.3f} | "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"IoU: {iou:.3f}",
            fontsize=13,
            fontweight="bold"
        )

        add_image(axes[0], t0, "T0 - Reference")
        add_image(axes[1], t1, "T1 - Current")

        add_image(
            axes[2],
            ground_truth,
            "Ground Truth",
            cmap="gray",
            vmin=0,
            vmax=1
        )

        add_image(
            axes[3],
            prediction,
            f"Prediction @ {THRESHOLD}",
            cmap="gray",
            vmin=0,
            vmax=1
        )

        add_image(
            axes[4],
            probability,
            "Probability Map",
            cmap="magma",
            vmin=0,
            vmax=1
        )

        add_image(
            axes[5],
            overlay,
            "Error Overlay"
        )

        fig.tight_layout(rect=[0, 0, 1, 0.88])

        output_path = (
            OUTPUT_DIR / f"{sample_id}_audit.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        all_figures.append((sample_id, fig))

        plt.close(fig)

        print("Saved:", output_path)

    # -----------------------------------------------------
    # COMBINED REPORT
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        len(SAMPLE_IDS),
        6,
        figsize=(22, 5 * len(SAMPLE_IDS))
    )

    for row, sample_id in enumerate(SAMPLE_IDS):

        record = record_map[sample_id]

        t0 = load_rgb(
            resolve_image_path(record, "t0_path", "t0")
        )

        t1 = load_rgb(
            resolve_image_path(record, "t1_path", "t1")
        )

        gt = (
            load_gray(
                resolve_image_path(record, "mask_path", "mask")
            ) >= 128
        )

        pred = (
            load_gray(
                PREDICTION_DIR / f"{sample_id}.png"
            ) >= 128
        )

        prob = (
            load_gray(
                PROBABILITY_DIR / f"{sample_id}.png"
            ).astype(np.float32) / 255.0
        )

        overlay = t1.copy()

        tp = pred & gt
        fp = pred & ~gt
        fn = ~pred & gt

        overlay[tp] = (
            0.45 * overlay[tp] + 0.55 * np.array([0, 255, 0])
        ).astype(np.uint8)

        overlay[fp] = (
            0.45 * overlay[fp] + 0.55 * np.array([255, 0, 0])
        ).astype(np.uint8)

        overlay[fn] = (
            0.45 * overlay[fn] + 0.55 * np.array([0, 100, 255])
        ).astype(np.uint8)

        images = [
            (t0, "T0", None),
            (t1, "T1", None),
            (gt, "Ground Truth", "gray"),
            (pred, "Prediction", "gray"),
            (prob, "Probability", "magma"),
            (overlay, "Error Overlay", None),
        ]

        for col, (image, title, cmap) in enumerate(images):

            axes[row, col].imshow(
                image,
                cmap=cmap,
                vmin=0 if cmap else None,
                vmax=1 if cmap else None
            )

            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(
                    title,
                    fontsize=10,
                    fontweight="bold"
                )

        axes[row, 0].set_ylabel(
            sample_id,
            fontsize=10,
            fontweight="bold"
        )

    fig.suptitle(
        "TRINETRA - SIX SAMPLE VISUAL FAILURE AUDIT\n"
        "Green: True Positive | Red: False Positive | Blue: False Negative",
        fontsize=15,
        fontweight="bold"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    combined_path = OUTPUT_DIR / "combined_failure_audit.png"

    fig.savefig(
        combined_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\n" + "=" * 70)
    print("VISUAL AUDIT COMPLETED")
    print("=" * 70)
    print("Output directory:", OUTPUT_DIR)
    print("Combined report:", combined_path)


if __name__ == "__main__":
    main()
