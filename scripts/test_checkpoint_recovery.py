
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

SOURCE_CHECKPOINT = (
    ROOT
    / "outputs"
    / "lightning_smoke_test"
    / "checkpoints"
    / "last.ckpt"
)

OUTPUT_DIR = ROOT / "outputs" / "checkpoint_recovery_test"


def main():
    print("=" * 65)
    print("TRINETRA CHECKPOINT RECOVERY TEST")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Source checkpoint not found: {SOURCE_CHECKPOINT}"
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Source checkpoint:", SOURCE_CHECKPOINT)

    pl.seed_everything(42, workers=True)

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

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

    train_loader = DataLoader(
        Subset(train_dataset, [0]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        Subset(val_dataset, [0]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = TrinetraLightningModule(
        learning_rate=1e-4,
        weight_decay=1e-4,
        bce_weight=1.0,
        dice_weight=1.0,
        threshold=0.5,
        max_epochs=2,
    )

    # -----------------------------------------------------
    # NEW CHECKPOINT LOCATION
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR / "checkpoints",
        filename="recovery-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # -----------------------------------------------------
    # TRAINER
    # -----------------------------------------------------

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",

        # Original checkpoint completed epoch 0.
        # Two total epochs means one additional epoch.
        max_epochs=2,

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
    # RESUME
    # -----------------------------------------------------

    print("\nResuming training from checkpoint...")

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(SOURCE_CHECKPOINT),
    )

    # -----------------------------------------------------
    # VERIFY
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("CHECKPOINT RECOVERY RESULTS")
    print("=" * 65)

    print("Global step:", trainer.global_step)
    print("Completed epochs:", trainer.current_epoch)

    print("Best checkpoint:", checkpoint_callback.best_model_path)
    print("Last checkpoint:", checkpoint_callback.last_model_path)

    expected_checkpoint = OUTPUT_DIR / "checkpoints" / "last.ckpt"

    if not expected_checkpoint.is_file():
        raise RuntimeError("Recovered checkpoint was not saved.")

    if trainer.global_step != 2:
        raise RuntimeError(
            f"Expected global step 2, got {trainer.global_step}."
        )

    print("\nCheckpoint recovery: PASSED")

    print("=" * 65)
    print("ALL CHECKPOINT RECOVERY TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
