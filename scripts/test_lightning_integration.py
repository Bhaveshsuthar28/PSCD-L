
from pathlib import Path

import torch
import lightning.pytorch as pl

from torch.utils.data import DataLoader, Subset
from lightning.pytorch.callbacks import ModelCheckpoint

from data.datasets import VLCMUCDataset
from src.lightning_module import TrinetraLightningModule


ROOT = Path(__file__).resolve().parents[1]

INDEX = ROOT / "outputs" / "index" / "vl_cmu_train_val_split.json"
CACHE = ROOT / "data" / "processed" / "vl_cmu_cd_v2"
DATASET_ROOT = ROOT / "Dataset" / "VL-CMU-CD-binary255"

OUTPUT_DIR = ROOT / "outputs" / "lightning_smoke_test"


def main():
    print("=" * 65)
    print("TRINETRA LIGHTNING INTEGRATION TEST")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("PyTorch:", torch.__version__)
    print("Lightning:", pl.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    pl.seed_everything(42, workers=True)

    # -----------------------------------------------------
    # DATASETS
    # -----------------------------------------------------

    print("\nLoading datasets...")

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

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Training or validation dataset is empty.")

    # One sample per split for this integration test.
    train_subset = Subset(train_dataset, [0])
    val_subset = Subset(val_dataset, [0])

    train_loader = DataLoader(
        train_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # -----------------------------------------------------
    # LIGHTNING MODULE
    # -----------------------------------------------------

    print("\nInitializing model...")

    model = TrinetraLightningModule(
        learning_rate=1e-4,
        weight_decay=1e-4,
        bce_weight=1.0,
        dice_weight=1.0,
        threshold=0.5,
        max_epochs=1,
    )

    # -----------------------------------------------------
    # CHECKPOINT CALLBACK
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR / "checkpoints",
        filename="integration-test-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # -----------------------------------------------------
    # LIGHTNING TRAINER
    # -----------------------------------------------------

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",

        max_epochs=1,

        limit_train_batches=1,
        limit_val_batches=1,

        num_sanity_val_steps=0,

        callbacks=[checkpoint_callback],

        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,

        deterministic=True,
        benchmark=False,

        log_every_n_steps=1,
        default_root_dir=str(OUTPUT_DIR),
    )

    # -----------------------------------------------------
    # FIT
    # -----------------------------------------------------

    print("\nStarting one-batch Lightning training test...")

    torch.cuda.reset_peak_memory_stats()

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("LIGHTNING INTEGRATION RESULTS")
    print("=" * 65)

    print("Trainer global step:", trainer.global_step)
    print("Completed epochs:", trainer.current_epoch)

    print("Best checkpoint:", checkpoint_callback.best_model_path)
    print("Last checkpoint:", checkpoint_callback.last_model_path)

    best_checkpoint = Path(checkpoint_callback.best_model_path)
    last_checkpoint = Path(checkpoint_callback.last_model_path)

    if not best_checkpoint.is_file():
        raise RuntimeError("Best checkpoint was not created.")

    if not last_checkpoint.is_file():
        raise RuntimeError("Last checkpoint was not created.")

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    print(f"Peak GPU memory: {peak_memory:.2f} GB")

    if trainer.global_step != 1:
        raise RuntimeError(
            f"Expected one optimizer step, got {trainer.global_step}."
        )

    print("\n" + "=" * 65)
    print("LIGHTNING INTEGRATION TEST PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
