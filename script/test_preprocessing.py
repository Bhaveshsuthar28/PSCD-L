import json

import numpy as np

from config import OUTPUT_ROOT
from preprocessing import load_image, load_mask


INDEX_DIR = OUTPUT_ROOT / "index"


def test_sample(index_name):
    path = INDEX_DIR / f"{index_name}.json"

    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    sample = samples[0]

    image_t0 = load_image(sample["t0_path"])
    image_t1 = load_image(sample["t1_path"])

    if index_name == "pscd":
        rule = "binary_0_255"
    elif index_name == "pscd_pers":
        rule = "binary_0_255"
    else:
        rule = "mask > 0"

    mask = load_mask(sample["mask_path"], rule)

    assert image_t0.mode == "RGB"
    assert image_t1.mode == "RGB"

    assert image_t0.size == image_t1.size
    assert mask.shape[::-1] == image_t0.size

    assert mask.dtype == np.float32

    values = np.unique(mask)

    assert set(values).issubset({0.0, 1.0})

    print(
        f"[PASS] {index_name}: "
        f"image={image_t0.size} "
        f"mask_values={values.tolist()}"
    )


def main():
    test_sample("pscd")
    test_sample("pscd_pers")
    test_sample("vl_cmu_train")

    print("\nSTEP 6B: PASS")


if __name__ == "__main__":
    main()