
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LossWeights:
    bce: float = 1.0
    dice: float = 1.0


class BinaryDiceLoss(nn.Module):
    """
    Dice loss for binary change detection.

    Accepts raw logits and binary targets.
    Sigmoid is applied internally.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()

        if smooth <= 0:
            raise ValueError("smooth must be greater than zero.")

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.float()
        targets = targets.float()

        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (probabilities * targets).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return (1.0 - dice_score).mean()


class CombinedChangeDetectionLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice loss.

    Input:
        logits:  [B, 1, H, W] raw decoder logits
        targets: [B, 1, H, W] binary masks in [0, 1]

    Output:
        Scalar combined loss.

    Formula:
        total = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights cannot be negative.")

        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight must be positive.")

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

        self.bce_loss = nn.BCEWithLogitsLoss()

        self.dice_loss = BinaryDiceLoss(
            smooth=dice_smooth
        )

    @staticmethod
    def _validate_inputs(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:

        if logits.ndim != 4:
            raise ValueError(
                f"Expected logits [B, C, H, W], got {tuple(logits.shape)}"
            )

        if targets.ndim != 4:
            raise ValueError(
                f"Expected targets [B, C, H, W], got {tuple(targets.shape)}"
            )

        if logits.shape != targets.shape:
            raise ValueError(
                "Logits and targets must have identical shapes. "
                f"Got {tuple(logits.shape)} and {tuple(targets.shape)}"
            )

        if logits.shape[1] != 1:
            raise ValueError(
                f"Expected one output channel, got {logits.shape[1]}"
            )

        if not logits.is_floating_point():
            raise TypeError("Logits must be floating-point tensors.")

        if not torch.isfinite(logits).all():
            raise ValueError("Logits contain NaN or Inf values.")

        if not torch.isfinite(targets).all():
            raise ValueError("Targets contain NaN or Inf values.")

        if targets.min() < 0 or targets.max() > 1:
            raise ValueError("Targets must contain values in [0, 1].")

    def compute_components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns individual losses and their weighted total.
        """

        self._validate_inputs(logits, targets)

        targets = targets.to(
            device=logits.device,
            dtype=torch.float32,
        )

        logits = logits.float()

        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
        )

        return {
            "total": total,
            "bce": bce,
            "dice": dice,
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        return self.compute_components(
            logits,
            targets,
        )["total"]
