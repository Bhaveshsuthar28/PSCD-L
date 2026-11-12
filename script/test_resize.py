import json

import numpy as np
from PIL import Image

from config import OUTPUT_ROOT


INDEX_DIR = OUTPUT_ROOT / "index"

IMAGE_SIZE = 518


def resize_image(image):
    scale = IMAGE_SIZE / min(image.size)

    width = round(image.width * scale)
    height = round(image.height * scale)

    return image.resize(
        (width, height),
        Image.Resampling.BICUBIC
    )


def resize_mask(mask):
    scale = IMAGE_SIZE / min(mask.size)

    width = round(mask.width * scale)
    height = round(mask.height * scale)

    return mask.resize(
        (width, height),
        Image.Resampling.NEAREST
    )


def test_sample(index_name, rule):
    path = INDEX_DIR / f"{index_name}.json"

    with open(path, "r", encoding="utf-8") as f:
        sample = json.load(f)[0]

    image = Image.open(sample["t0_path"]).convert("RGB")
    mask = Image.open(sample["mask_path"]).convert("L")

    resized_image = resize_image(image)
    resized_mask = resize_mask(mask)

    mask_array = np.asarray(resized_mask)

    if rule == "binary":
        mask_array = (mask_array > 0).astype(np.float32)

    assert min(resized_image.size) == IMAGE_SIZE
    assert resized_image.mode == "RGB"

    assert min(resized_mask.size) == IMAGE_SIZE
    assert set(np.unique(mask_array)).issubset({0.0, 1.0})

    print(
        f"[PASS] {index_name}: "
        f"{image.size} -> {resized_image.size} | "
        f"mask -> {resized_mask.size}"
    )


def main():
    test_sample("pscd", "binary")
    test_sample("pscd_pers", "binary")
    test_sample("vl_cmu_train", "binary")

    print("\nSTEP 6C: PASS")


if __name__ == "__main__":
    main()