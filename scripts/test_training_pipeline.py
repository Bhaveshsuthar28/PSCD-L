
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.datasets import VLCMUCDataset


ROOT = Path(__file__).resolve().parents[1]

INDEX = ROOT / "outputs" / "index" / "vl_cmu_train_val_split.json"
CACHE = ROOT / "data" / "processed" / "vl_cmu_cd_v2"
DATASET_ROOT = (
    ROOT
    / "Dataset"
    / "VL-CMU-CD-binary255"
)


def main():
    print("=" * 60)
    print("TRINETRA TRAINING PIPELINE SMOKE TEST")
    print("=" * 60)

    for path in [INDEX, CACHE, DATASET_ROOT]:
        print(f"{path}: {'FOUND' if path.exists() else 'MISSING'}")
        assert path.exists(), f"Required path missing: {path}"

    train_dataset = VLCMUCDataset(
        index_path=INDEX,
        cache_dir=CACHE,
        dataset_root=DATASET_ROOT,
        split="train",
    )

    val_dataset = VLCMUCDataset(
        index_path=INDEX,
        cache_dir=CACHE,
        dataset_root=DATASET_ROOT,
        split="val",
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for name, loader in [
        ("TRAIN", train_loader),
        ("VALIDATION", val_loader),
    ]:
        print(f"\n{name} BATCH TEST")
        batch = next(iter(loader))

        for key in ["t0", "t1", "mask"]:
            tensor = batch[key]

            print(
                f"{key}: shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, "
                f"finite={torch.isfinite(tensor).all().item()}"
            )

            assert torch.isfinite(tensor).all(), (
                f"Non-finite values found in {key}"
            )

        assert batch["t0"].shape == (1, 768, 37, 37)
        assert batch["t1"].shape == (1, 768, 37, 37)
        assert batch["mask"].shape == (1, 1, 512, 512)

        assert batch["mask"].min() >= 0
        assert batch["mask"].max() <= 1

        print("Batch validation: PASSED")

    print("\n" + "=" * 60)
    print("ALL TRAINING PIPELINE SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
