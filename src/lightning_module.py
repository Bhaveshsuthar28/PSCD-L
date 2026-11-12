
from __future__ import annotations

from typing import Any

import torch
import lightning.pytorch as pl

from src.models.decoder import ChangeDetectionDecoder
from src.models.losses import CombinedChangeDetectionLoss


class TrinetraLightningModule(pl.LightningModule):
    """
    Trinetra change-detection training module.

    Inputs:
        t0:   [B, 768, 37, 37]
        t1:   [B, 768, 37, 37]
        mask: [B, 1, 512, 512]

    Output:
        logits: [B, 1, 512, 512]
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        threshold: float = 0.5,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = ChangeDetectionDecoder()

        self.criterion = CombinedChangeDetectionLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

        self.threshold = threshold

        self.register_buffer(
            "_val_tp", torch.zeros((), dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_val_fp", torch.zeros((), dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_val_fn", torch.zeros((), dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_val_tn", torch.zeros((), dtype=torch.float64),
            persistent=False,
        )

    def forward(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(t0, t1)

    @staticmethod
    def _validate_batch(batch: dict[str, Any]) -> None:
        required_keys = {"t0", "t1", "mask"}
        missing = required_keys - batch.keys()

        if missing:
            raise ValueError(
                f"Batch is missing required keys: {missing}"
            )

        t0 = batch["t0"]
        t1 = batch["t1"]
        mask = batch["mask"]

        if t0.ndim != 4 or t1.ndim != 4:
            raise ValueError(
                "Feature tensors must have shape [B, C, H, W]."
            )

        if mask.ndim != 4:
            raise ValueError(
                "Mask tensor must have shape [B, C, H, W]."
            )

        if t0.shape != t1.shape:
            raise ValueError(
                "Reference and current feature shapes differ."
            )

        if t0.shape[1:] != (768, 37, 37):
            raise ValueError(
                f"Unexpected feature shape: {tuple(t0.shape)}"
            )

        if mask.shape[1:] != (1, 512, 512):
            raise ValueError(
                f"Unexpected mask shape: {tuple(mask.shape)}"
            )

        if t0.shape[0] != mask.shape[0]:
            raise ValueError(
                "Feature and mask batch sizes differ."
            )

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        self._validate_batch(batch)

        t0 = batch["t0"]
        t1 = batch["t1"]
        masks = batch["mask"]

        logits = self(t0, t1)

        losses = self.criterion.compute_components(
            logits,
            masks,
        )

        total_loss = losses["total"]

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=t0.shape[0],
        )

        self.log(
            "train_bce",
            losses["bce"],
            on_step=False,
            on_epoch=True,
            batch_size=t0.shape[0],
        )

        self.log(
            "train_dice_loss",
            losses["dice"],
            on_step=False,
            on_epoch=True,
            batch_size=t0.shape[0],
        )

        return total_loss

    def on_validation_epoch_start(self) -> None:
        self._val_tp.zero_()
        self._val_fp.zero_()
        self._val_fn.zero_()
        self._val_tn.zero_()

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        self._validate_batch(batch)

        t0 = batch["t0"]
        t1 = batch["t1"]
        masks = batch["mask"]

        logits = self(t0, t1)

        losses = self.criterion.compute_components(
            logits,
            masks,
        )

        self.log(
            "val_loss",
            losses["total"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=t0.shape[0],
        )

        self.log(
            "val_bce",
            losses["bce"],
            on_step=False,
            on_epoch=True,
            batch_size=t0.shape[0],
        )

        self.log(
            "val_dice_loss",
            losses["dice"],
            on_step=False,
            on_epoch=True,
            batch_size=t0.shape[0],
        )

        probabilities = torch.sigmoid(logits.float())
        predictions = probabilities >= self.threshold
        targets = masks >= 0.5

        self._val_tp += (
            predictions & targets
        ).sum().double()

        self._val_fp += (
            predictions & ~targets
        ).sum().double()

        self._val_fn += (
            ~predictions & targets
        ).sum().double()

        self._val_tn += (
            ~predictions & ~targets
        ).sum().double()

        return losses["total"]

    def on_validation_epoch_end(self) -> None:
        epsilon = 1e-8

        tp = self._val_tp
        fp = self._val_fp
        fn = self._val_fn
        tn = self._val_tn

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = (
            2.0 * precision * recall
            / (precision + recall + epsilon)
        )

        iou = tp / (tp + fp + fn + epsilon)

        dice = (
            2.0 * tp
            / (2.0 * tp + fp + fn + epsilon)
        )

        accuracy = (
            tp + tn
        ) / (
            tp + tn + fp + fn + epsilon
        )

        self.log(
            "val_precision",
            precision.float(),
            sync_dist=True,
        )

        self.log(
            "val_recall",
            recall.float(),
            sync_dist=True,
        )

        self.log(
            "val_f1",
            f1.float(),
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val_iou",
            iou.float(),
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val_dice",
            dice.float(),
            sync_dist=True,
        )

        self.log(
            "val_accuracy",
            accuracy.float(),
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.learning_rate * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
