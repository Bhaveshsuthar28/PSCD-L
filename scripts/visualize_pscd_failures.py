import json
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path.cwd()
INDEX = ROOT / "outputs/index/pscd.json"
CSV = ROOT / "outputs/evaluation/pscd_zero_shot/per_sample_metrics.csv"
PRED = ROOT / "outputs/evaluation/pscd_zero_shot/predictions"
OUT = ROOT / "outputs/evaluation/pscd_zero_shot/visual_audit.png"

# Load dataset index
with open(INDEX, "r", encoding="utf-8") as f:
    records = json.load(f)

record_map = {r["sample_id"]: r for r in records}

# Load per-sample metrics, preserving leading zeros
with open(CSV, "r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda r: float(r["f1"]))

# Select 2 worst, 2 middle, and 2 best samples
selected = [
    rows[0],
    rows[1],
    rows[len(rows) // 2 - 1],
    rows[len(rows) // 2],
    rows[-2],
    rows[-1],
]

def resolve_dataset_path(old_path):
    marker = "PSCD\\"
    parts = old_path.split(marker, 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot resolve dataset path: {old_path}")
    return ROOT / "Dataset" / "PSCD" / parts[1]

# Image layout
CELL_W = 640
CELL_H = 180
HEADER_H = 42
GAP = 12
COLS = 4
ROWS = len(selected)

canvas = Image.new(
    "RGB",
    (COLS * CELL_W + (COLS + 1) * GAP,
     ROWS * (CELL_H + HEADER_H) + (ROWS + 1) * GAP),
    "white"
)
draw = ImageDraw.Draw(canvas)

labels = ["T0", "T1", "GROUND TRUTH", "PREDICTION"]

for row_index, metric in enumerate(selected):
    sid = metric["sample_id"]
    record = record_map[sid]

    paths = [
        resolve_dataset_path(record["t0_path"]),
        resolve_dataset_path(record["t1_path"]),
        resolve_dataset_path(record["mask_path"]),
        PRED / f"{sid}.png",
    ]

    y = GAP + row_index * (CELL_H + HEADER_H)

    f1 = float(metric["f1"])
    recall = float(metric["recall"])
    precision = float(metric["precision"])

    draw.text(
        (GAP, y),
        f"Sample {sid} | F1: {f1:.4f} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f}",
        fill="black"
    )

    for col, (label, path) in enumerate(zip(labels, paths)):
        x = GAP + col * (CELL_W + GAP)
        draw.text((x, y + 20), label, fill="black")

        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")

        img = Image.open(path).convert("RGB")
        img.thumbnail((CELL_W, CELL_H), Image.Resampling.BILINEAR)

        # Center image in its cell
        px = x + (CELL_W - img.width) // 2
        py = y + HEADER_H + (CELL_H - img.height) // 2
        canvas.paste(img, (px, py))

canvas.save(OUT)
print(f"\nVisual audit saved: {OUT}")
print(f"Samples included: {[r['sample_id'] for r in selected]}")
