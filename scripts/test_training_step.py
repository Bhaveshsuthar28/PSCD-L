
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.datasets import VLCMUCDataset
from src.lightning_module import TrinetraLightningModule


ROOT = Path(__file__).resolve().parents[1]

INDEX = ROOT / "outputs" / "index" / "vl_cmu_train_val_split.json"
CACHE = ROOT / "data" / "processed" / "vl_cmu_cd_v2"
DATASET_ROOT = ROOT / "Dataset" / "VL-CMU-CD-binary255"


def main():
    print("=" * 60)
    print("TRINETRA REAL TRAINING STEP TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    dataset = VLCMUCDataset(
        index_path=INDEX,
        cache_dir=CACHE,
        dataset_root=DATASET_ROOT,
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    batch = next(iter(loader))

    t0 = batch["t0"].to(device, non_blocking=True)
    t1 = batch["t1"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    model = TrinetraLightningModule(
        learning_rate=1e-4,
        weight_decay=1e-4,
        max_epochs=50,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    print("\nRunning FP16 mixed-precision forward pass...")

    scaler = torch.amp.GradScaler("cuda")

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        logits = model(t0, t1)
        loss = model.criterion(logits.float(), mask.float())

    print("Logits shape:", tuple(logits.shape))
    print("Loss:", loss.item())

    if not torch.isfinite(loss):
        raise RuntimeError("Training loss is not finite.")

    print("\nRunning backward pass...")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    gradient_count = sum(
        1
        for parameter in model.parameters()
        if parameter.grad is not None
    )

    if gradient_count == 0:
        raise RuntimeError("No gradients were generated.")

    scaler.step(optimizer)
    scaler.update()

    print("Parameters with gradients:", gradient_count)
    print("Optimizer step: PASSED")
    print("Gradient scaler scale:", scaler.get_scale())

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print("\nGPU MEMORY")
    print("-" * 40)
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Peak:      {peak:.2f} GB")

    print("\n" + "=" * 60)
    print("REAL TRAINING STEP TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
