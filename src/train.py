
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from data.datasets import VLCMUCDataset
from src.lightning_module import TrinetraLightningModule


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INDEX = (
    PROJECT_ROOT / "outputs" / "index"
    / "vl_cmu_train_val_split.json"
)

DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "vl_cmu_cd_v2"
)

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Dataset" / "VL-CMU-CD-binary255"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Trinetra change-detection decoder."
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)

    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=4,
    )

    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--precision",
        choices=[
            "auto",
            "bf16-mixed",
            "16-mixed",
            "32-true",
        ],
        default="auto",
        help=(
            "Training precision. Auto selects BF16 when supported "
            "on CUDA, otherwise FP16 mixed precision on CUDA."
        ),
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional Lightning checkpoint to resume from.",
    )

    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.batch_size < 1 or args.val_batch_size < 1:
        raise ValueError("Batch sizes must be at least 1.")

    if args.accumulate_grad_batches < 1:
        raise ValueError(
            "--accumulate-grad-batches must be at least 1."
        )

    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")

    if args.bce_weight < 0 or args.dice_weight < 0:
        raise ValueError("Loss weights cannot be negative.")

    if args.bce_weight + args.dice_weight == 0:
        raise ValueError("At least one loss weight must be positive.")

    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1.")

    if args.early_stopping_patience < 1:
        raise ValueError(
            "--early-stopping-patience must be at least 1."
        )

    if not args.index.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {args.index}"
        )

    if not args.cache_dir.is_dir():
        raise FileNotFoundError(
            f"Feature cache directory not found: {args.cache_dir}"
        )

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {args.dataset_root}"
        )

    if args.resume_from is not None:
        if not args.resume_from.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.resume_from}"
            )


def validate_split_manifest(index_path: Path) -> None:
    with index_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError(
            "Expected a train/validation split manifest dictionary."
        )

    if "train" not in manifest or "val" not in manifest:
        raise ValueError(
            "Split manifest must contain 'train' and 'val' arrays."
        )

    train_records = manifest["train"]
    val_records = manifest["val"]

    if not isinstance(train_records, list) or not isinstance(
        val_records, list
    ):
        raise ValueError("Train and validation entries must be lists.")

    if not train_records or not val_records:
        raise ValueError(
            "Train and validation splits must both contain samples."
        )

    train_ids = {item["sample_id"] for item in train_records}
    val_ids = {item["sample_id"] for item in val_records}

    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} overlapping samples."
        )

    print("\nSPLIT MANIFEST")
    print("-" * 50)
    print(f"Training samples:   {len(train_records):,}")
    print(f"Validation samples: {len(val_records):,}")
    print(f"Overlapping IDs:    {len(overlap)}")


def create_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:

    print("\nLoading training dataset...")

    train_dataset = VLCMUCDataset(
        index_path=args.index,
        cache_dir=args.cache_dir,
        dataset_root=args.dataset_root,
        split="train",
    )

    print("Loading validation dataset...")

    val_dataset = VLCMUCDataset(
        index_path=args.index,
        cache_dir=args.cache_dir,
        dataset_root=args.dataset_root,
        split="val",
    )

    common_loader_args = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }

    if args.num_workers > 0:
        common_loader_args["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **common_loader_args,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )

    print("\nDATALOADERS")
    print("-" * 50)
    print(f"Training batches:      {len(train_loader):,}")
    print(f"Validation batches:    {len(val_loader):,}")
    print(f"Training batch size:   {args.batch_size}")
    print(f"Validation batch size: {args.val_batch_size}")
    print(f"Workers:               {args.num_workers}")

    return train_loader, val_loader


def select_precision(requested: str, use_cuda: bool) -> str:
    if not use_cuda:
        if requested not in ("auto", "32-true"):
            raise RuntimeError(
                f"Precision '{requested}' requires CUDA. "
                "This run would otherwise use CPU."
            )
        return "32-true"

    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        return "16-mixed"

    if requested == "bf16-mixed":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 was requested, but this GPU does not report "
                "BF16 support."
            )

    return requested


def create_callbacks(args: argparse.Namespace) -> list:
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="trinetra-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(
            logging_interval="epoch"
        ),
    ]

    if not args.disable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=args.early_stopping_patience,
                min_delta=0.0001,
                verbose=True,
            )
        )

    return callbacks


def create_trainer(
    args: argparse.Namespace,
) -> pl.Trainer:

    use_cuda = torch.cuda.is_available()

    if not use_cuda:
        raise RuntimeError(
            "CUDA GPU is not available. Training stopped to prevent "
            "an unintended CPU training run."
        )

    precision = select_precision(args.precision, use_cuda)

    print("\nTRAINING HARDWARE")
    print("-" * 50)
    print(f"Accelerator: GPU")
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"Precision:   {precision}")

    logger = CSVLogger(
        save_dir=str(args.output_dir / "logs"),
        name="trinetra",
    )

    return pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        precision=precision,

        accumulate_grad_batches=args.accumulate_grad_batches,

        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",

        callbacks=create_callbacks(args),
        logger=logger,

        default_root_dir=str(args.output_dir),

        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,

        deterministic=True,
        benchmark=False,

        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )


def main() -> None:
    args = parse_args()

    validate_arguments(args)
    validate_split_manifest(args.index)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    torch.set_float32_matmul_precision("high")

    print("\n" + "=" * 60)
    print("TRINETRA CHANGE DETECTION TRAINING")
    print("=" * 60)

    print(f"PyTorch: {torch.__version__}")
    print(f"Lightning: {pl.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Gradient accumulation: {args.accumulate_grad_batches}")

    effective_batch_size = (
        args.batch_size * args.accumulate_grad_batches
    )

    print(
        "Effective training batch size: "
        f"{effective_batch_size}"
    )

    train_loader, val_loader = create_dataloaders(args)

    print("\nInitializing Lightning module...")

    model = TrinetraLightningModule(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        threshold=args.threshold,
        max_epochs=args.epochs,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    trainer = create_trainer(args)

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=(
            str(args.resume_from)
            if args.resume_from is not None
            else None
        ),
    )

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    checkpoint_callback = next(
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ModelCheckpoint)
    )

    print("Best checkpoint:", checkpoint_callback.best_model_path)
    print("Best validation loss:", checkpoint_callback.best_model_score)
    print("Last checkpoint:", checkpoint_callback.last_model_path)
    print("Output directory:", args.output_dir)


if __name__ == "__main__":
    main()
