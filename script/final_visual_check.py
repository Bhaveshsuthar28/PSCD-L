import json
import random

import numpy as np
from PIL import Image, ImageDraw

from config import OUTPUT_ROOT


INDEX_DIR = OUTPUT_ROOT / "index"
OUTPUT = OUTPUT_ROOT / "final_visual_check.png"

random.seed(42)


DATASETS = {
    "PSCD": INDEX_DIR / "pscd.json",
    "PSCD_pers": INDEX_DIR / "pscd_pers.json",
    "VL-CMU": INDEX_DIR / "vl_cmu_train.json",
}


def load_sample(path):
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    return random.choice(samples)


def prepare_image(path, size=(320, 240)):
    image = Image.open(path).convert("RGB")
    image.thumbnail(size)
    return image


def prepare_mask(path, size=(320, 240)):
    mask = Image.open(path).convert("L")
    mask.thumbnail(size)
    return mask


def make_overlay(image, mask):
    image = image.copy()
    array = np.asarray(image).copy()
    mask_array = np.asarray(mask)

    changed = mask_array > 0

    array[changed] = (
        0.5 * array[changed] +
        0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    return Image.fromarray(array)


def make_row(name, sample):
    t0 = prepare_image(sample["t0_path"])
    t1 = prepare_image(sample["t1_path"])
    mask = prepare_mask(sample["mask_path"])

    overlay = make_overlay(t1, mask)

    row = Image.new("RGB", (1280, 280), "white")

    row.paste(t0, (0, 40))
    row.paste(t1, (320, 40))

    mask_rgb = Image.merge(
        "RGB",
        (mask, mask, mask)
    )

    row.paste(mask_rgb, (640, 40))
    row.paste(overlay, (960, 40))

    draw = ImageDraw.Draw(row)

    draw.text((10, 10), f"{name} - t0", fill="black")
    draw.text((330, 10), "t1", fill="black")
    draw.text((650, 10), "mask", fill="black")
    draw.text((970, 10), "overlay", fill="black")

    return row


def main():
    rows = []

    for name, path in DATASETS.items():
        sample = load_sample(path)

        print(
            f"{name}: "
            f"{sample['sample_id']}"
        )

        rows.append(
            make_row(name, sample)
        )

    output = Image.new(
        "RGB",
        (1280, 280 * len(rows)),
        "white"
    )

    for i, row in enumerate(rows):
        output.paste(row, (0, i * 280))

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.save(OUTPUT)

    print(f"\n[PASS] Saved: {OUTPUT}")
    print("STEP 8A: PASS")


if __name__ == "__main__":
    main()