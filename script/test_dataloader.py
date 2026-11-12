from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import OUTPUT_ROOT
from dataset import ChangeDetectionDataset


INDEX_DIR = OUTPUT_ROOT / "index"


DATASETS = {
    "pscd": INDEX_DIR / "pscd.json",
    "pscd_pers": INDEX_DIR / "pscd_pers.json",
    "vl_cmu_train": INDEX_DIR / "vl_cmu_train.json",
}


def test_loader(name, index_path):
    dataset = ChangeDetectionDataset(index_path)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    t0 = batch["t0"]
    t1 = batch["t1"]
    mask = batch["mask"]

    assert t0.ndim == 4
    assert t1.ndim == 4
    assert mask.ndim == 4

    assert t0.shape[0] == 2
    assert t1.shape[0] == 2
    assert mask.shape[0] == 2

    assert t0.shape[1] == 3
    assert t1.shape[1] == 3
    assert mask.shape[1] == 1

    assert t0.shape[2:] == t1.shape[2:]
    assert t0.shape[2:] == mask.shape[2:]

    assert t0.dtype == torch.float32
    assert t1.dtype == torch.float32
    assert mask.dtype == torch.float32

    print(
        f"[PASS] {name}: "
        f"t0={tuple(t0.shape)} "
        f"t1={tuple(t1.shape)} "
        f"mask={tuple(mask.shape)}"
    )


def main():
    for name, path in DATASETS.items():
        test_loader(name, path)

    print("\nSTEP 7C: PASS")


if __name__ == "__main__":
    main()