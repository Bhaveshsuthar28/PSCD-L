from pathlib import Path


PROJECT_ROOT = Path(
    r"C:\Users\bhave\OneDrive\Desktop\AI Projects\PSCD&L"
)

DATASET_ROOT = PROJECT_ROOT / "Dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


DATASETS = {
    "pscd": {
        "name": "PSCD",
        "root": DATASET_ROOT / "PSCD",
        "t0": "t0",
        "t1": "t1",
        "mask": "mask",
        "size": (4096, 1152),
        "image_mode": "RGB",
        "mask_mode": "L",
        "mask_rule": "binary_0_255",
    },

    "pscd_pers": {
        "name": "PSCD_pers",
        "root": DATASET_ROOT / "PSCD_pers",
        "t0": "t0",
        "t1": "t1",
        "mask": "mask",
        "size": (3325, 1920),
        "image_mode": "RGB",
        "mask_mode": "L",
        "mask_rule": "binary_0_255",
    },

    "vl_cmu": {
        "name": "VL-CMU-CD",
        "root": DATASET_ROOT / "VL-CMU-CD-binary255",
        "train": {
            "t0": "train/t0",
            "t1": "train/t1",
            "mask": "train/mask",
        },
        "test": {
            "t0": "test/t0",
            "t1": "test/t1",
            "mask": "test/mask",
        },
        "size": (512, 512),
        "image_mode": "RGB",
        "mask_mode": "L",
        "mask_rule": "mask > 0",
    },
}


ABODA = {
    "name": "ABODA",
    "root": DATASET_ROOT / "ABODA",
    "extension": ".avi",
}