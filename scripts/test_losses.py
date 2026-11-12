
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.decoder import ChangeDetectionDecoder
from src.models.losses import (
    BinaryDiceLoss,
    CombinedChangeDetectionLoss,
)
from data.datasets import VLCMUCDataset


def test_loss_with_synthetic_data(device: torch.device) -> None:
    print("\n1. SYNTHETIC LOSS TEST")

    batch_size = 2

    logits = torch.randn(
        batch_size, 1, 512, 512,
        device=device,
        requires_grad=True,
    )

    targets = torch.randint(
        0, 2,
        (batch_size, 1, 512, 512),
        device=device,
    ).float()

    criterion = CombinedChangeDetectionLoss(
        bce_weight=1.0,
        dice_weight=1.0,
    ).to(device)

    components = criterion.compute_components(
        logits,
        targets,
    )

    for name, value in components.items():
        print(f"{name.upper()} LOSS: {value.item():.6f}")

    assert torch.isfinite(components["total"])
    assert torch.isfinite(components["bce"])
    assert torch.isfinite(components["dice"])

    components["total"].backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0

    print("Synthetic loss and gradient test: PASSED")


def test_perfect_prediction() -> None:
    print("\n2. PERFECT PREDICTION TEST")

    targets = torch.tensor(
        [[[[0.0, 1.0], [1.0, 0.0]]]]
    )

    logits = torch.where(
        targets == 1,
        torch.tensor(20.0),
        torch.tensor(-20.0),
    )

    criterion = CombinedChangeDetectionLoss()

    loss = criterion(logits, targets)

    print(f"Perfect prediction loss: {loss.item():.8f}")

    assert loss.item() < 0.001

    print("Perfect prediction test: PASSED")


def test_empty_mask() -> None:
    print("\n3. EMPTY MASK TEST")

    logits = torch.zeros(2, 1, 32, 32)
    targets = torch.zeros(2, 1, 32, 32)

    criterion = CombinedChangeDetectionLoss()

    loss = criterion(logits, targets)

    assert torch.isfinite(loss)
    assert loss.item() > 0

    print(f"Empty-mask loss: {loss.item():.6f}")
    print("Empty mask test: PASSED")


def test_invalid_target() -> None:
    print("\n4. INVALID TARGET TEST")

    logits = torch.zeros(1, 1, 8, 8)
    targets = torch.full((1, 1, 8, 8), 2.0)

    criterion = CombinedChangeDetectionLoss()

    try:
        criterion(logits, targets)
    except ValueError:
        print("Invalid target correctly rejected: PASSED")
        return

    raise AssertionError("Invalid targets were not rejected.")


def test_real_dataset(
    project_root: Path,
    device: torch.device,
) -> None:
    print("\n5. REAL VL-CMU DATASET TEST")

    dataset = VLCMUCDataset(
        index_path=(
            project_root
            / "outputs/index/vl_cmu_train_val_split.json"
        ),
        cache_dir=(
            project_root
            / "data/processed/vl_cmu_cd_v2"
        ),
        dataset_root=(
            project_root
            / "Dataset/VL-CMU-CD-binary255"
        ),
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    t0 = batch["t0"].to(device)
    t1 = batch["t1"].to(device)
    masks = batch["mask"].to(device)

    print(f"t0 shape: {tuple(t0.shape)}")
    print(f"t1 shape: {tuple(t1.shape)}")
    print(f"Mask shape: {tuple(masks.shape)}")
    print(f"Mask minimum: {masks.min().item()}")
    print(f"Mask maximum: {masks.max().item()}")
    print(
        "Changed-pixel proportion: "
        f"{masks.mean().item():.6f}"
    )

    decoder = ChangeDetectionDecoder().to(device)

    decoder.eval()

    with torch.no_grad():
        logits = decoder(t0, t1)

    criterion = CombinedChangeDetectionLoss().to(device)

    components = criterion.compute_components(
        logits,
        masks,
    )

    print(f"Logits shape: {tuple(logits.shape)}")

    for name, value in components.items():
        print(f"{name.upper()} LOSS: {value.item():.6f}")

    assert logits.shape == masks.shape
    assert torch.isfinite(components["total"])

    print("Real dataset loss test: PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip the real VL-CMU dataset test.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("TRINETRA LOSS FUNCTION TEST")
    print("=" * 55)
    print(f"Project root: {project_root}")
    print(f"Device: {device}")

    test_loss_with_synthetic_data(device)
    test_perfect_prediction()
    test_empty_mask()
    test_invalid_target()

    if not args.skip_dataset:
        test_real_dataset(project_root, device)

    print("\n" + "=" * 55)
    print("ALL LOSS TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
