import random

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

    random.seed(42)

    indices = random.sample(
        range(len(dataset)),
        4
    )

    for index in indices:
        sample = dataset[index]

        t0 = sample["t0"]
        t1 = sample["t1"]
        mask = sample["mask"]

        assert torch.isfinite(t0).all()
        assert torch.isfinite(t1).all()
        assert torch.isfinite(mask).all()

        assert t0.shape[1:] == t1.shape[1:]
        assert t0.shape[1:] == mask.shape[1:]

        values = torch.unique(mask)

        assert torch.all(
            (values == 0.0) |
            (values == 1.0)
        )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    for _ in range(3):
        batch = next(iter(loader))

        assert torch.isfinite(batch["t0"]).all()
        assert torch.isfinite(batch["t1"]).all()
        assert torch.isfinite(batch["mask"]).all()

    print(f"[PASS] {name}")


def main():
    for name, path in DATASETS.items():
        test_loader(name, path)

    print("\nSTEP 7D: PASS")


if __name__ == "__main__":
    main()