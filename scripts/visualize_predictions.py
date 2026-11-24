
from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader

from data.datasets import VLCMUCDataset
from src.lightning_module import TrinetraLightningModule


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INDEX = ROOT / "outputs" / "index" / "vl_cmu_train_val_split.json"

DEFAULT_CACHE = ROOT / "data" / "processed" / "vl_cmu_cd_v2"

DEFAULT_DATASET_ROOT = ROOT / "Dataset" / "VL-CMU-CD-binary255"

DEFAULT_CHECKPOINT = (
    ROOT / "outputs" / "training" / "checkpoints"
    / "trinetra-33-0.1482.ckpt"
)

DEFAULT_OUTPUT = ROOT / "outputs" / "visualizations" / "pilot"


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Trinetra change detection visualizations."
    )

    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
    )

    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

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
            raise FileNotFoundError(f"Path not found: {path}")

    if args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1.")


# ---------------------------------------------------------
# PORTABLE ORIGINAL IMAGE PATH RESOLUTION
# ---------------------------------------------------------

def resolve_dataset_path(path_string, dataset_root):
    """
    Resolve paths stored as absolute Windows paths or relative paths.

    The dataset_root must point to:
    Dataset/VL-CMU-CD-binary255
    """

    raw_path = str(path_string)

    windows_path = PureWindowsPath(raw_path)

    parts = windows_path.parts

    dataset_folder = "VL-CMU-CD-binary255"

    matching_index = next(
        (
            i for i, part in enumerate(parts)
            if part.casefold() == dataset_folder.casefold()
        ),
        None,
    )

    if matching_index is not None:
        relative_parts = parts[matching_index + 1:]
        resolved = Path(dataset_root).joinpath(*relative_parts)

    else:
        normal_path = Path(raw_path)

        if normal_path.is_absolute() and normal_path.is_file():
            resolved = normal_path
        else:
            resolved = Path(dataset_root) / normal_path

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Original image not found: {resolved}"
        )

    return resolved


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

def load_model(checkpoint_path, device):

    print("\nLoading trained checkpoint...")

    # Only load trusted local checkpoints.
    checkpoint = torch.load(
        checkpoint_path,
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

    print("Checkpoint loaded.")
    print(
        "Trainable decoder parameters:",
        sum(p.numel() for p in model.model.parameters()),
    )

    return model


# ---------------------------------------------------------
# IMAGE UTILITIES
# ---------------------------------------------------------

def load_rgb(path):
    image = Image.open(path).convert("RGB")

    if image.size != (512, 512):
        image = image.resize(
            (512, 512),
            Image.Resampling.BILINEAR,
        )

    return image


def load_font(size=22):
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]

    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)

    return ImageFont.load_default()


def create_prediction_overlay(image, prediction):
    """
    Highlight predicted change regions in red.
    """

    image = image.convert("RGBA")

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    red_layer = Image.new(
        "RGBA",
        image.size,
        (255, 40, 40, 115),
    )

    mask = prediction.convert("L")

    overlay.paste(red_layer, (0, 0), mask)

    return Image.alpha_composite(image, overlay).convert("RGB")


def create_probability_heatmap(probability):
    """
    Convert probability values into a colored heatmap.
    """

    probability_array = np.clip(
        probability * 255,
        0,
        255,
    ).astype(np.uint8)

    grayscale = Image.fromarray(
        probability_array,
        mode="L",
    )

    return ImageOps.colorize(
        grayscale,
        black="#081d58",
        mid="#f03b20",
        white="#ffffcc",
        midpoint=128,
    )


# ---------------------------------------------------------
# COMPARISON IMAGE
# ---------------------------------------------------------

def create_comparison(
    sample_id,
    t0,
    t1,
    ground_truth,
    prediction,
    probability,
    overlay,
):
    """
    Generate a 2-column, 3-row comparison image.
    """

    tile_size = 512
    label_height = 45
    margin = 12

    columns = 2
    rows = 3

    canvas_width = (
        columns * tile_size
        + (columns + 1) * margin
    )

    canvas_height = (
        rows * (tile_size + label_height)
        + (rows + 1) * margin
        + 65
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        "#171b25",
    )

    draw = ImageDraw.Draw(canvas)
    font = load_font(22)
    title_font = load_font(27)

    draw.text(
        (margin, 12),
        f"TRINETRA CHANGE DETECTION | {sample_id}",
        fill="white",
        font=title_font,
    )

    tiles = [
        ("T0 - Reference image", t0),
        ("T1 - Current image", t1),
        ("Ground truth", ground_truth),
        ("Predicted change mask", prediction),
        ("Probability heatmap", probability),
        ("Prediction overlay", overlay),
    ]

    start_y = 60

    for index, (label, image) in enumerate(tiles):

        row = index // columns
        column = index % columns

        x = margin + column * (tile_size + margin)

        y = (
            start_y
            + margin
            + row * (tile_size + label_height + margin)
        )

        draw.text(
            (x, y - label_height),
            label,
            fill="white",
            font=font,
        )

        canvas.paste(
            image.convert("RGB").resize(
                (tile_size, tile_size)
            ),
            (x, y),
        )

    return canvas


# ---------------------------------------------------------
# MAIN VISUALIZATION
# ---------------------------------------------------------

def main():

    args = parse_args()
    validate_args(args)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print("=" * 65)
    print("TRINETRA INFERENCE AND VISUALIZATION")
    print("=" * 65)

    print("Split:", args.split)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # -----------------------------------------------------
    # READ ORIGINAL INDEX RECORDS
    # -----------------------------------------------------

    with args.index.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError(
            "Expected the group-aware train/validation manifest."
        )

    records = manifest[args.split][:args.limit]

    if not records:
        raise ValueError("No records found for requested split.")

    record_lookup = {
        record["sample_id"]: record
        for record in records
    }

    print("Samples selected:", len(records))

    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    dataset = VLCMUCDataset(
        index_path=args.index,
        cache_dir=args.cache_dir,
        dataset_root=args.dataset_root,
        split=args.split,
    )

    # Preserve the selected manifest sample order.
    dataset_indices = {
        dataset[index]["sample_id"]: index
        for index in range(len(dataset))
    }

    selected_indices = []

    for sample_id in record_lookup:
        if sample_id not in dataset_indices:
            raise ValueError(
                f"Sample missing from dataset: {sample_id}"
            )

        selected_indices.append(
            dataset_indices[sample_id]
        )

    from torch.utils.data import Subset

    selected_dataset = Subset(
        dataset,
        selected_indices,
    )

    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = load_model(
        args.checkpoint,
        device,
    )

    # -----------------------------------------------------
    # OUTPUT DIRECTORIES
    # -----------------------------------------------------

    output = args.output_dir / args.split

    directories = {
        "t0": output / "t0",
        "t1": output / "t1",
        "ground_truth": output / "ground_truth",
        "predictions": output / "predictions",
        "probabilities": output / "probabilities",
        "overlays": output / "overlays",
        "comparisons": output / "comparisons",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # INFERENCE
    # -----------------------------------------------------

    processed = 0

    print("\nStarting inference...")

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

            masks = batch["mask"]

            use_amp = device.type == "cuda"

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(
                    t0_features,
                    t1_features,
                )

            probabilities = torch.sigmoid(logits.float())

            predictions = (
                probabilities >= args.threshold
            ).to(torch.uint8)

            for i in range(len(batch["sample_id"])):

                sample_id = batch["sample_id"][i]

                record = record_lookup[sample_id]

                t0_path = resolve_dataset_path(
                    record["t0_path"],
                    args.dataset_root,
                )

                t1_path = resolve_dataset_path(
                    record["t1_path"],
                    args.dataset_root,
                )

                t0_image = load_rgb(t0_path)
                t1_image = load_rgb(t1_path)

                ground_truth_array = (
                    masks[i, 0].numpy() >= 0.5
                ).astype(np.uint8) * 255

                prediction_array = (
                    predictions[i, 0].cpu().numpy()
                ).astype(np.uint8) * 255

                probability_array = (
                    probabilities[i, 0]
                    .cpu()
                    .numpy()
                )

                ground_truth_image = Image.fromarray(
                    ground_truth_array,
                    mode="L",
                )

                prediction_image = Image.fromarray(
                    prediction_array,
                    mode="L",
                )

                probability_image = create_probability_heatmap(
                    probability_array
                )

                overlay_image = create_prediction_overlay(
                    t1_image,
                    prediction_image,
                )

                # Save individual images.
                t0_image.save(
                    directories["t0"] / f"{sample_id}.png"
                )

                t1_image.save(
                    directories["t1"] / f"{sample_id}.png"
                )

                ground_truth_image.save(
                    directories["ground_truth"] / f"{sample_id}.png"
                )

                prediction_image.save(
                    directories["predictions"] / f"{sample_id}.png"
                )

                probability_image.save(
                    directories["probabilities"] / f"{sample_id}.png"
                )

                overlay_image.save(
                    directories["overlays"] / f"{sample_id}.png"
                )

                # Save comparison image.
                comparison = create_comparison(
                    sample_id,
                    t0_image,
                    t1_image,
                    ground_truth_image,
                    prediction_image,
                    probability_image,
                    overlay_image,
                )

                comparison.save(
                    directories["comparisons"]
                    / f"{sample_id}_comparison.png"
                )

                processed += 1

                print(
                    f"[{processed}/{len(records)}] "
                    f"Saved visualization: {sample_id}"
                )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "threshold": args.threshold,
        "samples_visualized": processed,
        "output_directory": str(output.resolve()),
    }

    with (output / "visualization_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("VISUALIZATION COMPLETED")
    print("=" * 65)

    print("Samples processed:", processed)
    print("Output directory:", output)
    print("Comparison images:", directories["comparisons"])


if __name__ == "__main__":
    main()
