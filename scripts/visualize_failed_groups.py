
from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Subset

from data.datasets import VLCMUCDataset
from src.lightning_module import TrinetraLightningModule


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT / "outputs" / "training" / "checkpoints"
    / "trinetra-00-0.3455.ckpt"
)

INDEX = ROOT / "outputs" / "index" / "vl_cmu_train_val_split.json"

CACHE = ROOT / "data" / "processed" / "vl_cmu_cd_v2"

DATASET_ROOT = ROOT / "Dataset" / "VL-CMU-CD-binary255"

OUTPUT = ROOT / "outputs" / "visualizations" / "failed_groups"

SAMPLE_IDS = [
    f"{group}_{rotation}"
    for group in [
        "131_1_01",
        "131_1_03",
        "131_1_04",
    ]
    for rotation in ["0", "90", "180", "270"]
]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Visualize Trinetra's failed validation groups."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
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


def resolve_dataset_path(path_string):

    parts = PureWindowsPath(str(path_string)).parts

    folder = "VL-CMU-CD-binary255"

    matching_index = next(
        (
            i for i, part in enumerate(parts)
            if part.casefold() == folder.casefold()
        ),
        None,
    )

    if matching_index is None:
        candidate = Path(path_string)

        if candidate.is_absolute() and candidate.is_file():
            return candidate

        raise ValueError(
            f"Cannot resolve dataset path: {path_string}"
        )

    relative_parts = parts[matching_index + 1:]

    path = DATASET_ROOT.joinpath(*relative_parts)

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def load_model(device):

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" not in checkpoint:
        raise ValueError("Invalid Lightning checkpoint.")

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

    return model


def load_rgb(path):

    image = Image.open(path).convert("RGB")

    if image.size != (512, 512):
        image = image.resize(
            (512, 512),
            Image.Resampling.BILINEAR,
        )

    return image


def get_font(size=21):

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]

    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)

    return ImageFont.load_default()


def calculate_metrics(prediction, target):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 0.0
    )
    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def make_heatmap(probability):

    array = np.clip(
        probability * 255,
        0,
        255,
    ).astype(np.uint8)

    return ImageOps.colorize(
        Image.fromarray(array, mode="L"),
        black="#081d58",
        mid="#f03b20",
        white="#ffffcc",
        midpoint=128,
    )


def make_overlay(image, prediction):

    image = image.convert("RGBA")

    red = Image.new(
        "RGBA",
        image.size,
        (255, 35, 35, 115),
    )

    transparent = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    transparent.paste(red, (0, 0), prediction)

    return Image.alpha_composite(
        image,
        transparent,
    ).convert("RGB")


def make_comparison(
    sample_id,
    t0,
    t1,
    target,
    prediction,
    probability,
    overlay,
    metrics,
):

    width = 512
    height = 512
    margin = 14
    label_height = 36
    header_height = 100

    canvas = Image.new(
        "RGB",
        (2 * width + 3 * margin, header_height + 3 * (height + label_height + margin)),
        "#171b25",
    )

    draw = ImageDraw.Draw(canvas)

    title_font = get_font(25)
    label_font = get_font(20)
    metric_font = get_font(17)

    draw.text(
        (margin, 10),
        f"TRINETRA | {sample_id}",
        fill="white",
        font=title_font,
    )

    metric_text = (
        f"F1: {metrics['f1']:.4f} | "
        f"IoU: {metrics['iou']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f}"
    )

    draw.text(
        (margin, 47),
        metric_text,
        fill="#d0d5e0",
        font=metric_font,
    )

    tiles = [
        ("T0 - Reference", t0),
        ("T1 - Current", t1),
        ("Ground truth", target),
        ("Prediction", prediction),
        ("Probability heatmap", probability),
        ("Prediction overlay", overlay),
    ]

    for index, (label, image) in enumerate(tiles):

        row = index // 2
        column = index % 2

        x = margin + column * (width + margin)

        y = (
            header_height
            + margin
            + row * (height + label_height + margin)
        )

        draw.text(
            (x, y - label_height),
            label,
            fill="white",
            font=label_font,
        )

        canvas.paste(
            image.convert("RGB").resize((width, height)),
            (x, y),
        )

    return canvas


def main():

    args = parse_args()

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    if not INDEX.is_file():
        raise FileNotFoundError(INDEX)

    if not 0 < args.threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print("=" * 65)
    print("TRINETRA FAILED GROUP DIAGNOSTICS")
    print("=" * 65)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    with INDEX.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    validation_records = manifest["val"]

    if len(validation_records) != 372:
        raise ValueError(
            f"Expected 372 validation records, "
            f"found {len(validation_records)}."
        )

    record_lookup = {
        record["sample_id"]: record
        for record in validation_records
    }

    missing = [
        sample_id
        for sample_id in SAMPLE_IDS
        if sample_id not in record_lookup
    ]

    if missing:
        raise ValueError(
            f"Requested samples missing from validation: {missing}"
        )

    # Dataset ordering follows the validation manifest ordering.
    selected_indices = [
        index
        for index, record in enumerate(validation_records)
        if record["sample_id"] in SAMPLE_IDS
    ]

    if len(selected_indices) != len(SAMPLE_IDS):
        raise RuntimeError("Incorrect number of selected samples.")

    dataset = VLCMUCDataset(
        index_path=INDEX,
        cache_dir=CACHE,
        dataset_root=DATASET_ROOT,
        split="val",
    )

    if len(dataset) != len(validation_records):
        raise RuntimeError(
            "Dataset ordering/length does not match the manifest."
        )

    selected_dataset = Subset(
        dataset,
        selected_indices,
    )

    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = load_model(device)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics = []

    print("\nSelected samples:", len(SAMPLE_IDS))
    print("Threshold:", args.threshold)
    print("\nStarting inference...\n")

    with torch.inference_mode():

        for batch in loader:

            t0_features = batch["t0"].to(
                device,
                non_blocking=True,
            )

            t1_features = batch["t1"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):

                logits = model(
                    t0_features,
                    t1_features,
                )

            probabilities = torch.sigmoid(
                logits.float()
            )

            predictions = (
                probabilities >= args.threshold
            )

            for index, sample_id in enumerate(batch["sample_id"]):

                record = record_lookup[sample_id]

                t0 = load_rgb(
                    resolve_dataset_path(record["t0_path"])
                )

                t1 = load_rgb(
                    resolve_dataset_path(record["t1_path"])
                )

                target_array = (
                    batch["mask"][index, 0].numpy() >= 0.5
                )

                prediction_array = (
                    predictions[index, 0].cpu().numpy()
                )

                probability_array = (
                    probabilities[index, 0].cpu().numpy()
                )

                metrics = calculate_metrics(
                    prediction_array,
                    target_array,
                )

                metrics["sample_id"] = sample_id
                metrics["threshold"] = args.threshold

                all_metrics.append(metrics)

                target_image = Image.fromarray(
                    target_array.astype(np.uint8) * 255,
                    mode="L",
                )

                prediction_image = Image.fromarray(
                    prediction_array.astype(np.uint8) * 255,
                    mode="L",
                )

                probability_image = make_heatmap(
                    probability_array
                )

                overlay_image = make_overlay(
                    t1,
                    prediction_image,
                )

                comparison = make_comparison(
                    sample_id,
                    t0,
                    t1,
                    target_image,
                    prediction_image,
                    probability_image,
                    overlay_image,
                    metrics,
                )

                comparison.save(
                    OUTPUT / f"{sample_id}_comparison.png"
                )

                print(
                    f"{sample_id} | "
                    f"F1={metrics['f1']:.4f} | "
                    f"IoU={metrics['iou']:.4f} | "
                    f"FP={metrics['fp']} | "
                    f"FN={metrics['fn']}"
                )

    report = {
        "checkpoint": str(CHECKPOINT.resolve()),
        "split": "val",
        "threshold": args.threshold,
        "samples": all_metrics,
        "mean_f1": float(
            np.mean([item["f1"] for item in all_metrics])
        ),
        "mean_iou": float(
            np.mean([item["iou"] for item in all_metrics])
        ),
    }

    with (
        OUTPUT / "failed_groups_report.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("FAILED GROUP DIAGNOSTICS COMPLETED")
    print("=" * 65)

    print("Samples processed:", len(all_metrics))
    print("Mean F1:", f"{report['mean_f1']:.4f}")
    print("Mean IoU:", f"{report['mean_iou']:.4f}")
    print("Output:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
