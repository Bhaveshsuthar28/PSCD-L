import torch

from config import OUTPUT_ROOT
from dataset import ChangeDetectionDataset


INDEX_DIR = OUTPUT_ROOT / "index"


DATASETS = {
    "pscd": INDEX_DIR / "pscd.json",
    "pscd_pers": INDEX_DIR / "pscd_pers.json",
    "vl_cmu_train": INDEX_DIR / "vl_cmu_train.json",
}


def validate_sample(name, index_path):
    dataset = ChangeDetectionDataset(index_path)

    sample = dataset[0]

    t0 = sample["t0"]
    t1 = sample["t1"]
    mask = sample["mask"]

    assert isinstance(t0, torch.Tensor)
    assert isinstance(t1, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    assert t0.ndim == 3
    assert t1.ndim == 3
    assert mask.ndim == 3

    assert t0.shape[0] == 3
    assert t1.shape[0] == 3
    assert mask.shape[0] == 1

    assert t0.shape[1:] == t1.shape[1:]
    assert t0.shape[1:] == mask.shape[1:]

    assert t0.dtype == torch.float32
    assert t1.dtype == torch.float32
    assert mask.dtype == torch.float32

    assert 0.0 <= t0.min() <= 1.0
    assert 0.0 <= t1.min() <= 1.0

    assert t0.max() <= 1.0
    assert t1.max() <= 1.0

    mask_values = torch.unique(mask)

    assert torch.all(
        (mask_values == 0.0) |
        (mask_values == 1.0)
    )

    print(
        f"[PASS] {name}: "
        f"t0={tuple(t0.shape)} "
        f"t1={tuple(t1.shape)} "
        f"mask={tuple(mask.shape)}"
    )


def main():
    for name, path in DATASETS.items():
        validate_sample(name, path)

    print("\nSTEP 7B: PASS")


if __name__ == "__main__":
    main()